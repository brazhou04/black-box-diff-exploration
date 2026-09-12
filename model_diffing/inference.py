from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from safety_training.formatting import prompt_messages
from safety_training.modeling import configure_hf_cache, load_model, load_tokenizer
from safety_training.seeding import set_all_seeds

from .registry import TargetSpec


def sampling_seed(master_seed: int, turn: int, prompt_index: int) -> int:
    payload = f"{master_seed}:{turn}:{prompt_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


class SharedAdapterTargetPair:
    """One pinned base model with switchable, read-only LoRA target adapters."""

    def __init__(
        self,
        model_a: TargetSpec,
        model_b: TargetSpec,
        base_config: dict[str, Any],
        *,
        hf_home: str | Path | None = None,
    ) -> None:
        identity_a = (model_a.model_name, model_a.model_revision, model_a.tokenizer_revision)
        identity_b = (model_b.model_name, model_b.model_revision, model_b.tokenizer_revision)
        if identity_a != identity_b:
            raise ValueError("Target pair must share an exact base model and tokenizer revision")

        self.targets = {"A": model_a, "B": model_b}
        config = dict(base_config)
        config["model_name"] = model_a.model_name
        config["model_revision"] = model_a.model_revision
        config["tokenizer_revision"] = model_a.tokenizer_revision
        configure_hf_cache(config, hf_home)
        self.tokenizer = load_tokenizer(config)
        self.tokenizer.padding_side = "left"
        base = load_model(config, trainable=False)

        unique_adapters: list[Path] = []
        for target in (model_a, model_b):
            if target.adapter_path is not None:
                resolved = target.adapter_path.resolve()
                if resolved not in unique_adapters:
                    unique_adapters.append(resolved)
        self._adapter_names: dict[Path, str] = {}
        if unique_adapters:
            from peft import PeftModel

            first = unique_adapters[0]
            self._adapter_names[first] = "target_0"
            self.model = PeftModel.from_pretrained(
                base,
                first,
                adapter_name="target_0",
                is_trainable=False,
            )
            for index, adapter_path in enumerate(unique_adapters[1:], 1):
                name = f"target_{index}"
                self.model.load_adapter(adapter_path, adapter_name=name, is_trainable=False)
                self._adapter_names[adapter_path] = name
        else:
            self.model = base
        self.model.eval()

    @contextmanager
    def _activated(self, target: TargetSpec) -> Iterator[None]:
        if target.adapter_path is None:
            if self._adapter_names and hasattr(self.model, "disable_adapter"):
                with self.model.disable_adapter():
                    yield
            else:
                yield
            return
        self.model.set_adapter(self._adapter_names[target.adapter_path.resolve()])
        yield

    def _render_prompt(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            prompt_messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _generate(
        self,
        target: TargetSpec,
        prompt: str,
        samples: int,
        seed: int,
        sampling: dict[str, Any],
    ) -> list[str]:
        import torch

        set_all_seeds(seed)
        rendered = self._render_prompt(prompt)
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generation: dict[str, Any] = {
            "do_sample": bool(sampling["do_sample"]),
            "max_new_tokens": int(sampling["max_new_tokens"]),
            "num_return_sequences": int(samples),
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if generation["do_sample"]:
            generation["temperature"] = float(sampling["temperature"])
            generation["top_p"] = float(sampling["top_p"])
        with self._activated(target), torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation)
        prefix_length = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(output[prefix_length:], skip_special_tokens=True).strip()
            for output in outputs
        ]

    def send_messages(
        self,
        prompts: list[str],
        samples_per_model: int,
        *,
        turn: int,
        sampling: dict[str, Any],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for prompt_index, prompt in enumerate(prompts):
            seed = sampling_seed(int(sampling["master_seed"]), turn, prompt_index)
            samples_a = self._generate(self.targets["A"], prompt, samples_per_model, seed, sampling)
            samples_b = self._generate(self.targets["B"], prompt, samples_per_model, seed, sampling)
            results.append(
                {
                    "prompt": prompt,
                    "sampling_seed": seed,
                    "model_a": samples_a,
                    "model_b": samples_b,
                }
            )
        return results

