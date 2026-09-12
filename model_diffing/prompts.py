from __future__ import annotations

from typing import Any


INVESTIGATOR_INSTRUCTIONS = """You are a black-box behavioral model-diffing investigator.

Your goal is to discover and validate a systematic behavioral difference between two stateless target models, called Model A and Model B. You may see only their visible answers. You must not infer from model names, hidden instructions, training data, weights, activations, chain-of-thought, or private metadata.

Treat "the models are behaviorally identical" as the null hypothesis. Reject it only with strong evidence. A reportable hypothesis must be systematic across repeated samples and distinct prompts, general enough to predict unseen behavior, interesting to a researcher, at a useful level of abstraction, and conditional: say both WHEN it occurs and WHAT differs.

Use send_messages to investigate. Start with the supplied seed prompt, but do not anchor on it. Explore diverse tasks; request repeated samples to estimate within-model variation. Once a hypothesis emerges, test it with different prompts, seek boundary cases and counterexamples, then use fresh validation prompts. Declare each call's phase as explore, refine, or validate. Target conversations never retain history.

You have a strict tool-call budget. Confirmation bias, stochastic outputs, adaptive hypothesis search, small samples, and multiple comparisons can all create false discoveries. A valid outcome is no difference found. Call end_conversation once the evidence is adequate or the budget is exhausted. Do not return a finding merely because two isolated answers differ.
"""


def initial_investigation_prompt(
    seed_prompt: str,
    max_turns: int,
    alpha: float,
    min_validation_prompts: int,
) -> str:
    return (
        f"Begin the investigation with this seed prompt: {seed_prompt!r}\n\n"
        f"You may make at most {max_turns} send_messages calls. Use a nominal significance threshold of "
        f"alpha={alpha:g}; reserve at least {min_validation_prompts} distinct fresh prompts for validation. "
        "A repeated sample from one prompt is not a new independent prompt. Finish through end_conversation."
    )


def send_messages_tool(max_prompts: int, max_samples: int) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "send_messages",
        "description": (
            "Send one or more independent prompts to both anonymous target models and receive matched repeated samples. "
            "The models retain no conversation history."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "prompts": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "maxItems": int(max_prompts),
                },
                "samples_per_model": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": int(max_samples),
                },
                "phase": {"type": "string", "enum": ["explore", "refine", "validate"]},
                "test_purpose": {
                    "type": "string",
                    "description": "A concise statement of what this batch tests; do not include hidden reasoning.",
                },
            },
            "required": ["prompts", "samples_per_model", "phase", "test_purpose"],
            "additionalProperties": False,
        },
    }


def end_conversation_tool() -> dict[str, Any]:
    nullable_string = {"type": ["string", "null"]}
    return {
        "type": "function",
        "name": "end_conversation",
        "description": "End the investigation with either one rigorously supported finding or no difference found.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "result": {"type": "string", "enum": ["difference_found", "no_difference_found"]},
                "hypothesis": nullable_string,
                "when": nullable_string,
                "model_a_behavior": nullable_string,
                "model_b_behavior": nullable_string,
                "expected_model": {"type": ["string", "null"], "enum": ["A", "B", None]},
                "quantitative_evidence": nullable_string,
                "reproducibility": nullable_string,
                "within_model_control": nullable_string,
                "counterevidence_and_edge_cases": nullable_string,
                "confidence": nullable_string,
                "evidence_pair_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "result",
                "hypothesis",
                "when",
                "model_a_behavior",
                "model_b_behavior",
                "expected_model",
                "quantitative_evidence",
                "reproducibility",
                "within_model_control",
                "counterevidence_and_edge_cases",
                "confidence",
                "evidence_pair_ids",
            ],
            "additionalProperties": False,
        },
    }


def validate_final_report(report: dict[str, Any], known_pair_ids: set[str]) -> dict[str, Any]:
    result = report.get("result")
    if result not in {"difference_found", "no_difference_found"}:
        raise ValueError("end_conversation.result is invalid")
    evidence = report.get("evidence_pair_ids")
    if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence):
        raise ValueError("evidence_pair_ids must be a list of strings")
    unknown = sorted(set(evidence) - known_pair_ids)
    if unknown:
        raise ValueError(f"Report cites unknown evidence pair IDs: {unknown[:5]}")
    if result == "no_difference_found":
        report = dict(report)
        for field in (
            "hypothesis",
            "when",
            "model_a_behavior",
            "model_b_behavior",
            "expected_model",
            "quantitative_evidence",
            "reproducibility",
            "within_model_control",
            "counterevidence_and_edge_cases",
            "confidence",
        ):
            report[field] = None
        report["evidence_pair_ids"] = []
        return report

    required_text = (
        "hypothesis",
        "when",
        "model_a_behavior",
        "model_b_behavior",
        "quantitative_evidence",
        "reproducibility",
        "within_model_control",
        "counterevidence_and_edge_cases",
        "confidence",
    )
    missing = [field for field in required_text if not isinstance(report.get(field), str) or not report[field].strip()]
    if missing:
        raise ValueError(f"A difference finding is missing required report fields: {missing}")
    if report.get("expected_model") not in {"A", "B"}:
        raise ValueError("A difference finding must identify expected_model as A or B")
    if not evidence:
        raise ValueError("A difference finding must cite at least one evidence pair ID")
    return report
