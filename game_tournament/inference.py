from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from safety_training.config import REPO_ROOT, load_yaml
from safety_training.formatting import prompt_messages
from safety_training.modeling import configure_hf_cache, load_model, load_tokenizer

from .registry import AgentSpec


@dataclass(frozen=True)
class Decision:
    label: str
    probabilities: dict[str, float]
    logits: dict[str, float]


class ModelActionPolicy:
    """One shared base model with switchable LoRA adapters and constrained actions."""

    def __init__(
        self,
        agents: list[AgentSpec],
        *,
        temperature: float = 1.0,
        hf_home: str | Path | None = None,
        disable_thinking: bool = True,
    ) -> None:
        if not agents:
            raise ValueError("At least one model agent is required")
        if temperature <= 0:
            raise ValueError("Constrained sampling temperature must be positive")
        self.agents = {agent.id: agent for agent in agents}
        self.temperature = float(temperature)
        self.disable_thinking = bool(disable_thinking)

        first = agents[0]
        config = load_yaml(REPO_ROOT / "configs" / "base.yaml")
        config.update(
            model_name=first.model_name,
            model_revision=first.model_revision,
            tokenizer_revision=first.tokenizer_revision,
        )
        configure_hf_cache(config, hf_home)
        self.tokenizer = load_tokenizer(config)
        # Left padding keeps the final position aligned across a causal-LM batch.
        self.tokenizer.padding_side = "left"
        base = load_model(config, trainable=False)

        adapter_agents = [agent for agent in agents if agent.adapter_path is not None]
        self.model = base
        if adapter_agents:
            from peft import PeftModel

            first_adapter = adapter_agents[0]
            self.model = PeftModel.from_pretrained(
                base,
                first_adapter.adapter_path,
                adapter_name=first_adapter.id,
                is_trainable=False,
            )
            for agent in adapter_agents[1:]:
                self.model.load_adapter(agent.adapter_path, adapter_name=agent.id, is_trainable=False)
        self.model.eval()
        self._validate_option_tokens()

    def _serialize(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.disable_thinking:
            kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(prompt_messages(prompt), **kwargs)

    def _token_id(self, label: str) -> int:
        encoded = self.tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            raise ValueError(
                f"Option label {label!r} is not one token for the pinned tokenizer; "
                "paper-style first-token choice scoring requires single-token labels"
            )
        return int(encoded[0])

    def _validate_option_tokens(self) -> None:
        from .prompting import PAPER_OPTION_PAIRS

        for label in {value for pair in PAPER_OPTION_PAIRS for value in pair}:
            self._token_id(label)

    @contextmanager
    def _active_agent(self, agent_id: str) -> Iterator[None]:
        try:
            agent = self.agents[agent_id]
        except KeyError as exc:
            raise ValueError(f"Unknown model agent {agent_id!r}") from exc
        if agent.adapter_path is None and hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                yield
            return
        if agent.adapter_path is not None:
            self.model.set_adapter(agent_id)
        yield

    def decide(self, agent_id: str, prompt: str, labels: tuple[str, str], seed: int) -> Decision:
        return self.decide_many(agent_id, [(prompt, labels, seed)])[0]

    def decide_many(
        self,
        agent_id: str,
        requests: list[tuple[str, tuple[str, str], int]],
    ) -> list[Decision]:
        import torch

        if not requests:
            return []
        for _, labels, _ in requests:
            if labels[0] == labels[1]:
                raise ValueError("Decision labels must be distinct")
        rendered = [self._serialize(prompt) for prompt, _, _ in requests]
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with self._active_agent(agent_id), torch.inference_mode():
            # Qwen3/Transformers 4.57 supports retaining only the final logits,
            # avoiding a batch x sequence x vocabulary allocation.
            output = self.model(**encoded, logits_to_keep=1)
            next_token_logits = output.logits[:, -1, :].float().cpu()

        decisions: list[Decision] = []
        for index, (_, labels, seed) in enumerate(requests):
            token_ids = [self._token_id(label) for label in labels]
            selected_logits = next_token_logits[index, token_ids]
            probabilities_tensor = torch.softmax(selected_logits / self.temperature, dim=-1)
            probabilities = {
                label: float(probabilities_tensor[label_index])
                for label_index, label in enumerate(labels)
            }
            logits = {label: float(selected_logits[label_index]) for label_index, label in enumerate(labels)}
            draw = random.Random(seed).random()
            label = labels[0] if draw < probabilities[labels[0]] else labels[1]
            decisions.append(Decision(label=label, probabilities=probabilities, logits=logits))
        return decisions
