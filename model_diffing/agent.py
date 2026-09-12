from __future__ import annotations

import json
from typing import Any, Protocol

from .prompts import (
    INVESTIGATOR_INSTRUCTIONS,
    end_conversation_tool,
    initial_investigation_prompt,
    send_messages_tool,
    validate_final_report,
)
from .session import SessionRecorder


class TargetPairBackend(Protocol):
    def send_messages(
        self,
        prompts: list[str],
        samples_per_model: int,
        *,
        turn: int,
        sampling: dict[str, Any],
    ) -> list[dict[str, Any]]: ...


def _as_input_item(item: Any) -> Any:
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    return item


def _response_request(
    client: Any,
    *,
    investigator_model: str,
    input_items: list[Any],
    tools: list[dict[str, Any]],
    investigator_config: dict[str, Any],
) -> Any:
    request: dict[str, Any] = {
        "model": investigator_model,
        "instructions": INVESTIGATOR_INSTRUCTIONS,
        "input": input_items,
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "store": False,
        "max_output_tokens": int(investigator_config["max_output_tokens"]),
    }
    reasoning_effort = investigator_config.get("reasoning_effort")
    if reasoning_effort:
        request["reasoning"] = {"effort": reasoning_effort}
        # Preserve reasoning state across the stateless, store=False tool loop
        # without writing decrypted reasoning to the local experiment record.
        request["include"] = ["reasoning.encrypted_content"]
    return client.responses.create(**request)


def run_investigator(
    backend: TargetPairBackend,
    recorder: SessionRecorder,
    config: dict[str, Any],
    investigator_model: str,
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    """Run the paper-style tool loop without exposing target identity or reasoning."""
    if client is None:
        from openai import OpenAI

        client = OpenAI()

    method = config["method"]
    max_turns = int(method["max_turns"])
    send_tool = send_messages_tool(
        int(method["max_prompts_per_turn"]),
        int(method["max_samples_per_prompt"]),
    )
    end_tool = end_conversation_tool()
    manifest = recorder.manifest()
    input_items: list[Any] = [
        {
            "role": "user",
            "content": initial_investigation_prompt(
                manifest["seed_prompt"],
                max_turns,
                float(method["alpha"]),
                int(method["min_validation_prompts"]),
            ),
        }
    ]
    send_calls = 0
    api_rounds = 0
    try:
        while api_rounds < max_turns + 6:
            available_tools = [end_tool] if send_calls >= max_turns else [send_tool, end_tool]
            response = _response_request(
                client,
                investigator_model=investigator_model,
                input_items=input_items,
                tools=available_tools,
                investigator_config=config["investigator"],
            )
            api_rounds += 1
            recorder.add_api_response_id(response.id)
            input_items.extend(_as_input_item(item) for item in response.output)
            function_calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]

            visible_text = getattr(response, "output_text", "") or ""
            if visible_text.strip():
                recorder.record_visible_agent_text(response.id, visible_text)

            if not function_calls:
                reminder = (
                    "The investigation must continue through a tool call. "
                    + (
                        "The experiment budget is exhausted; call end_conversation now."
                        if send_calls >= max_turns
                        else "Call send_messages for another test or end_conversation to finish."
                    )
                )
                input_items.append({"role": "user", "content": reminder})
                continue

            call = function_calls[0]
            arguments = json.loads(call.arguments)
            if call.name == "send_messages":
                if send_calls >= max_turns:
                    result: Any = {"error": "send_messages budget exhausted; call end_conversation"}
                else:
                    prompts = arguments["prompts"]
                    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
                        raise ValueError("send_messages prompts must be non-empty strings")
                    send_calls += 1
                    raw_results = backend.send_messages(
                        prompts,
                        int(arguments["samples_per_model"]),
                        turn=send_calls,
                        sampling=config["sampling"],
                    )
                    result = recorder.record_query(
                        turn=send_calls,
                        phase=arguments["phase"],
                        test_purpose=arguments["test_purpose"],
                        raw_results=raw_results,
                    )
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )
                continue

            if call.name == "end_conversation":
                try:
                    report = validate_final_report(arguments, recorder.known_pair_ids())
                    if report["result"] == "difference_found":
                        validation_prompts = recorder.validation_prompt_ids_for_pairs(
                            set(report["evidence_pair_ids"])
                        )
                        minimum = int(method["min_validation_prompts"])
                        if len(validation_prompts) < minimum:
                            raise ValueError(
                                "A finding must cite held-out evidence from at least "
                                f"{minimum} distinct validation prompts; found {len(validation_prompts)}"
                            )
                except ValueError as error:
                    recorder.record_tool_error("end_conversation", str(error))
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps({"error": str(error)}),
                        }
                    )
                    continue
                recorder.finish(report)
                return report

            raise ValueError(f"Unknown investigator tool call: {call.name}")

        raise RuntimeError("Investigator failed to call end_conversation within the bounded API loop")
    except BaseException as error:
        recorder.fail(error)
        raise
