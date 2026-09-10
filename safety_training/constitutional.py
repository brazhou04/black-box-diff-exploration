from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, load_yaml
from .datasets import SAFETY_CATEGORIES, validate_record_provenance, validate_unique_ids
from .formatting import serialize_prompt
from .io import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_file
from .seeding import set_all_seeds


TARGET_GENERATION_CODE_VERSION = "2.0.0-binary"


def _generate(model: Any, tokenizer: Any, messages: list[dict[str, str]], settings: dict[str, Any]) -> str:
    import torch

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    kwargs = {key: value for key, value in settings.items() if value is not None}
    with torch.inference_mode():
        generated = model.generate(**encoded, **kwargs, pad_token_id=tokenizer.pad_token_id)
    new_tokens = generated[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_constitutional_targets(
    prompts_path: str | Path,
    output_path: str | Path,
    constitution_path: str | Path,
    teacher_model: str,
    teacher_revision: str = "main",
    max_new_tokens: int = 384,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    overwrite: bool = False,
    limit: int | None = None,
    generation_seed: int = 0,
) -> Path:
    prompts_path, output_path, constitution_path = map(Path, (prompts_path, output_path, constitution_path))
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite cached targets: {output_path}")
    constitution = load_yaml(constitution_path)
    principles = constitution.get("principles")
    if not isinstance(principles, list) or not all(isinstance(item, str) for item in principles):
        raise ValueError("Constitution must contain a list of string principles")
    prompts = read_jsonl(prompts_path)
    validate_unique_ids(prompts, "safety_shared")
    for record in prompts:
        if record.get("category") not in SAFETY_CATEGORIES or not isinstance(record.get("prompt"), str):
            raise ValueError(f"Invalid shared safety prompt: {record.get('id')}")
        validate_record_provenance(record, "safety_shared")
    if limit is not None:
        prompts = prompts[:limit]
    seed_record = set_all_seeds(generation_seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        teacher_model,
        revision=teacher_revision,
        device_map={"": 0} if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    resolved_teacher_revision = getattr(model.config, "_commit_hash", None) or teacher_revision
    tokenizer = AutoTokenizer.from_pretrained(teacher_model, revision=resolved_teacher_revision)
    if not tokenizer.chat_template:
        raise ValueError("Teacher tokenizer must provide an official chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    settings = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
    }
    principles_text = "\n".join(f"{index}. {item}" for index, item in enumerate(principles, 1))
    results = []
    for index, source in enumerate(prompts, 1):
        print(f"Generating constitutional target {index}/{len(prompts)}: {source['id']}")
        initial = _generate(model, tokenizer, [{"role": "user", "content": source["prompt"]}], settings)
        critique_prompt = (
            "Evaluate the candidate response against the constitution below. Identify concrete safety and "
            "helpfulness problems without answering the original request anew.\n\n"
            f"Constitution:\n{principles_text}\n\nOriginal prompt:\n{source['prompt']}\n\n"
            f"Candidate response:\n{initial}\n\nCritique:"
        )
        critique = _generate(model, tokenizer, [{"role": "user", "content": critique_prompt}], settings)
        revision_prompt = (
            "Revise the candidate response using the critique and constitution. Return only the final response. "
            "Be normally helpful on safe requests. Refuse or constrain unsafe assistance while offering safe "
            "redirection where appropriate. Ambiguous prompts are excluded from this binary training set.\n\n"
            f"Constitution:\n{principles_text}\n\nOriginal prompt:\n{source['prompt']}\n\n"
            f"Candidate response:\n{initial}\n\nCritique:\n{critique}\n\nRevised response:"
        )
        revised = _generate(model, tokenizer, [{"role": "user", "content": revision_prompt}], settings)
        if not initial or not critique or not revised:
            raise RuntimeError(f"Teacher generated an empty field for {source['id']}")
        results.append(
            {
                **source,
                "initial_response": initial,
                "constitution": principles,
                "critique": critique,
                "revised_response": revised,
                "condition": "M3",
            }
        )
    atomic_write_jsonl(output_path, results)
    generation_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_model": teacher_model,
        "teacher_requested_revision": teacher_revision,
        "teacher_revision": resolved_teacher_revision,
        "teacher_model_commit_hash": getattr(model.config, "_commit_hash", None),
        "teacher_tokenizer_commit_hash": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "generation_settings": settings,
        "generation_seed": generation_seed,
        "seeds": seed_record,
        "constitution_path": str(constitution_path),
        "constitution_version": constitution.get("version"),
        "constitution_hash": sha256_file(constitution_path),
        "source_prompts_path": str(prompts_path),
        "source_prompts_hash": sha256_file(prompts_path),
        "output_path": str(output_path),
        "constitutional_dataset_hash": sha256_file(output_path),
        "target_generation_code_version": TARGET_GENERATION_CODE_VERSION,
        "record_count": len(results),
        "cached_before_training": True,
    }
    atomic_write_json(output_path.parent / "generation_manifest.json", generation_manifest)
    return output_path
