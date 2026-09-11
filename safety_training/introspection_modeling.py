from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .modeling import load_model


IA_ADAPTER_NAME = "introspection"


def _lora_config(config: dict[str, Any]) -> Any:
    from peft import LoraConfig

    values = config["introspection"]["peft"]
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=int(values["r"]),
        lora_alpha=int(values["alpha"]),
        lora_dropout=float(values.get("dropout", 0.05)),
        target_modules=list(values["target_modules"]),
        bias="none",
    )


def _is_adapter_parameter(parameter_name: str, adapter_name: str) -> bool:
    return f".{adapter_name}." in parameter_name


def set_only_introspection_trainable(model: Any, trainable: bool) -> int:
    count = 0
    for name, parameter in model.named_parameters():
        selected = trainable and _is_adapter_parameter(name, IA_ADAPTER_NAME)
        parameter.requires_grad = selected
        if selected:
            count += parameter.numel()
    if trainable and count == 0:
        raise RuntimeError("No trainable introspection-adapter parameters were found")
    return count


def load_introspection_model(
    config: dict[str, Any],
    *,
    adapter_path: str | Path | None = None,
    trainable: bool,
) -> Any:
    base = load_model(config, trainable=trainable)
    quant = config["quantization"]
    if trainable and quant.get("enabled") and quant.get("load_in_4bit"):
        from peft import prepare_model_for_kbit_training

        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    from peft import PeftModel, get_peft_model

    if adapter_path is None:
        model = get_peft_model(base, _lora_config(config), adapter_name=IA_ADAPTER_NAME)
    else:
        model = PeftModel.from_pretrained(
            base,
            str(adapter_path),
            adapter_name=IA_ADAPTER_NAME,
            is_trainable=trainable,
        )
    if trainable and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    set_only_introspection_trainable(model, trainable)
    return model


def adapter_name_for_path(path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"behavior_{digest}"


def activate_adapters(
    model: Any,
    behavior_adapter: str | None,
    *,
    include_introspection: bool,
    train_introspection: bool,
) -> None:
    names = ([behavior_adapter] if behavior_adapter else []) + (
        [IA_ADAPTER_NAME] if include_introspection else []
    )
    if names:
        try:
            model.set_adapter(names if len(names) > 1 else names[0])
        except (TypeError, ValueError):
            model.base_model.set_adapter(names if len(names) > 1 else names[0])
    set_only_introspection_trainable(model, train_introspection)


class BehaviorAdapterCache:
    """Load frozen behavior LoRAs lazily and evict least-recently-used adapters."""

    def __init__(self, model: Any, max_loaded: int):
        if max_loaded < 1:
            raise ValueError("max_loaded must be positive")
        self.model = model
        self.max_loaded = max_loaded
        self.loaded: OrderedDict[Path, str] = OrderedDict()

    def ensure(self, adapter_path: Path | None) -> str | None:
        if adapter_path is None:
            return None
        path = adapter_path.resolve()
        if path in self.loaded:
            name = self.loaded.pop(path)
            self.loaded[path] = name
            return name
        while len(self.loaded) >= self.max_loaded:
            # Never delete an adapter while it is part of the active stack.
            self.model.set_adapter(IA_ADAPTER_NAME)
            _, old_name = self.loaded.popitem(last=False)
            self.model.delete_adapter(old_name)
        name = adapter_name_for_path(path)
        self.model.load_adapter(str(path), adapter_name=name, is_trainable=False)
        self.loaded[path] = name
        return name


def save_introspection_adapter(model: Any, destination: str | Path) -> Path:
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path), selected_adapters=[IA_ADAPTER_NAME], safe_serialization=True)
    # PEFT normally nests non-default named adapters below their adapter name.
    # Keep each artifact directory directly loadable by PeftModel.from_pretrained.
    nested = path / IA_ADAPTER_NAME
    if nested.is_dir() and (nested / "adapter_config.json").exists() and not (path / "adapter_config.json").exists():
        for item in nested.iterdir():
            item.replace(path / item.name)
        nested.rmdir()
    return path


def sequence_log_probs(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    labels: Any,
) -> Any:
    """Return summed assistant-token log probabilities for each sequence."""
    import torch.nn.functional as functional

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    mask = shifted_labels.ne(-100)
    gather_labels = shifted_labels.masked_fill(~mask, 0)
    token_log_probs = functional.log_softmax(shifted_logits, dim=-1)
    selected = token_log_probs.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)
    return (selected * mask).sum(dim=-1)
