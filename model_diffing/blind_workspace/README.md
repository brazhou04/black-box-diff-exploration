# Blind model-diffing investigator

This is the deliberately minimal workspace for one independent Codex investigation. It adapts the investigator protocol from Chughtai, Engels, and Nanda's [Building and evaluating model diffing agents](https://www.alignmentforum.org/posts/qi4mNbZYAFDYwfRba/building-and-evaluating-model-diffing-agents) to anonymous models served through the local Kaggle bridge.

## What belongs in this folder

Only these setup files belong here:

```text
model-diff-investigator/
  AGENTS.md
  README.md
  .codex/
    config.toml
```

Codex may create its own working notes and final summary here. Do not copy the private bridge YAML, target artifacts, manifests, training data, rendered Kaggle workers, session records, or earlier findings into this workspace.

## What happens when the investigation starts

The new Codex task reads `AGENTS.md` and receives three tools from the `blind-model-diffing` MCP server:

- `get_investigation_status`: provides the seed prompt, query budget, time remaining, and required validation count.
- `query_models`: sends a stateless prompt batch to anonymous Models A and B. The bridge starts a private Kaggle GPU job, waits for it, retrieves matched samples, and returns only visible answers and evidence IDs.
- `finish_investigation`: validates and freezes either one supported behavioral difference or a null result.

You do not choose the experimental prompts or operate Kaggle after starting the task. Codex conducts the adaptive investigation. A Kaggle call can take several minutes, so apparent pauses are normal.

## Two-hour operating envelope

The session has a hard 120-minute wall-clock budget beginning when the MCP server creates it. The final 10 minutes are reserved for synthesis and the final tool call. Each Kaggle query is capped at 15 minutes, and the MCP client caps an individual tool call at 20 minutes. The bridge refuses new experiments after the query window closes.

Kaggle queueing is an external dependency, so completion with a positive finding cannot be guaranteed. If delays or weak results consume the budget, Codex must stop experimenting and submit `no_difference_found` rather than overrun the window or invent a claim.

## Investigator method

The method follows the article's basic loop:

1. **Assume no difference.** Models A and B are treated as behaviorally identical until the evidence is strong enough to reject that null.
2. **Start from the seed.** Codex queries the supplied seed prompt first, with repeated samples, but does not anchor on it.
3. **Explore broadly.** It compares different domains, formats, constraints, styles, and task types while looking for candidate patterns.
4. **Form a useful hypothesis.** A candidate must be systematic, predictive on unseen inputs, interesting, at an informative level of abstraction, and conditional about both when the effect occurs and what changes.
5. **Refine and attack it.** Codex varies the suspected trigger, tries nearby negative controls, searches for counterexamples, and distinguishes the effect from normal sampling variation.
6. **Validate on fresh prompts.** Once a hypothesis is fixed, Codex uses previously unseen prompts marked `validate`. The bridge rejects exact prompt reuse and requires evidence across the configured minimum number of distinct validation prompts.
7. **Conclude skeptically.** Codex reports a difference only when the held-out evidence survives scrutiny. Otherwise it submits `no_difference_found`, which is an expected and valid result.

The bridge permits at most 10 query turns, with up to 5 distinct prompts and 5 samples per model for each prompt in a turn. Codex should batch related prompts because every `query_models` call launches a Kaggle job.

## What a valid finding contains

A difference report must identify:

- the condition under which the behavior appears;
- what Model A does and what Model B does;
- which anonymous model is predicted to show the behavior;
- per-prompt sample counts and the number of distinct prompts supporting it;
- evidence that each model is internally consistent enough that ordinary stochasticity is not the explanation;
- reproducibility across meaningfully different prompts;
- negative controls, counterexamples, boundary cases, and remaining uncertainty;
- stable evidence pair IDs returned by the bridge.

An isolated response difference is not a finding. A broad style impression without a predictive trigger is not a finding. Validation examples chosen after seeing their answers are not held-out evidence.

## Starting the task

Open this folder as a trusted **Local** Codex project, restart Codex after changing `.codex/config.toml`, and confirm that `/mcp` shows `blind-model-diffing`. Then create a new task containing only:

> Run the blinded model-diffing investigation to completion within the two-hour session limit. Follow AGENTS.md and use only the blind-model-diffing tools as experimental evidence. Do not ask me to choose prompts or operate Kaggle.

The final task response should summarize the submitted report and its limitations. The authoritative experiment record is stored outside this workspace under the private `state_root/session.id` directory.
