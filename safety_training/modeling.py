from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def configure_hf_cache(config: dict[str, Any], override: str | Path | None = None) -> Path:
    cache = Path(override or os.environ.get("HF_HOME") or config["paths"]["hf_home"]).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    return cache


def load_tokenizer(config: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        revision=config.get("tokenizer_revision") or config.get("model_revision"),
        use_fast=True,
    )
    if not tokenizer.chat_template:
        raise RuntimeError("The selected tokenizer has no official chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _dtype(name: str) -> Any:
    import torch

    values = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in values:
        raise ValueError(f"Unsupported compute_dtype {name!r}")
    return values[name]


def load_model(config: dict[str, Any], trainable: bool = True) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = config["quantization"]
    kwargs: dict[str, Any] = {
        "revision": config.get("model_revision"),
        "torch_dtype": _dtype(quant.get("compute_dtype", "float16")),
    }
    use_4bit = bool(quant.get("enabled") and quant.get("load_in_4bit"))
    if use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit bitsandbytes loading requires a CUDA GPU")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=_dtype(quant.get("compute_dtype", "float16")),
            bnb_4bit_use_double_quant=bool(quant.get("use_double_quant", True)),
        )
        kwargs["device_map"] = {"": 0}
    elif torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], **kwargs)
    model.config.use_cache = not trainable
    return model


def attach_lora(model: Any, config: dict[str, Any]) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    quant = config["quantization"]
    if quant.get("enabled") and quant.get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft = config["peft"]
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(peft["r"]),
        lora_alpha=int(peft["alpha"]),
        lora_dropout=float(peft["dropout"]),
        target_modules=list(peft["target_modules"]),
        bias="none",
    )
    return get_peft_model(model, lora)


def load_condition_model(config: dict[str, Any], adapter_path: Path | None) -> tuple[Any, Any]:
    tokenizer = load_tokenizer(config)
    model = load_model(config, trainable=False)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return model, tokenizer
