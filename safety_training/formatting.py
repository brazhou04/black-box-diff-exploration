from __future__ import annotations

from typing import Any


def prompt_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def serialize_prompt(tokenizer: Any, prompt: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat_template; official Qwen chat formatting is required")
    return tokenizer.apply_chat_template(prompt_messages(prompt), tokenize=False, add_generation_prompt=True)


def serialize_conversation(tokenizer: Any, prompt: str, response: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat_template; official Qwen chat formatting is required")
    messages = prompt_messages(prompt) + [{"role": "assistant", "content": response}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def serialize_assistant_completion(tokenizer: Any, prompt: str, response: str) -> str:
    prompt_text = serialize_prompt(tokenizer, prompt)
    full_text = serialize_conversation(tokenizer, prompt, response)
    if not full_text.startswith(prompt_text):
        raise ValueError("Tokenizer chat template does not yield a stable prompt prefix")
    return full_text[len(prompt_text) :]


def tokenize_supervised_example(
    tokenizer: Any, prompt: str, response: str, max_length: int
) -> dict[str, list[int]]:
    prompt_text = serialize_prompt(tokenizer, prompt)
    full_text = serialize_conversation(tokenizer, prompt, response)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
    prefix_length = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prefix_length + input_ids[prefix_length:]
    if not any(label != -100 for label in labels):
        raise ValueError("Response was completely truncated; increase training.max_seq_length")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class SupervisedCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must have a pad token")
        input_ids, masks, labels = [], [], []
        for item in features:
            padding = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * padding)
            masks.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
