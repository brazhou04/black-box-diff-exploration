# Model-diffing agent

This package is an isolated, black-box behavioral audit channel for comparing two frozen M0-M4 targets. It does not read training examples, target instructions, activations, gradients, or target chain-of-thought. It is designed around Chughtai, Engels, and Nanda's open-ended model-diffing agent protocol and is kept separate from both `game_tournament` and the introspection-adapter question banks.

Primary methodology: [Building and evaluating model diffing agents](https://www.lesswrong.com/posts/qi4mNbZYAFDYwfRba/building-and-evaluating-model-diffing-agents).

## What is implemented

- Anonymous Model A / Model B labels, with deterministic order randomization by default.
- One shared, revision-pinned Qwen base with switchable read-only LoRA adapters.
- Stateless target calls and visible answers only; Qwen thinking output is disabled.
- A no-OpenAI-API path where a Local Codex task investigates through a private MCP tool backed by Kaggle GPU jobs.
- An optional legacy Responses API investigator for users who explicitly want an API-driven end-to-end run.
- At most 10 investigation turns, 1-5 prompts per turn, and 1-5 matched samples per model and prompt.
- Explicit exploration, refinement, and held-out validation phases.
- A null-difference default, including support for M0-vs-M0 and adapter-vs-same-adapter false-positive controls.
- Complete prompt/output provenance in JSONL without saving investigator reasoning.
- Optional blinded comparative coding and an exact two-sided binomial check on validation prompts, with repeated samples clustered within prompt.
- A 50-prompt broad seed bank for independent repeated runs. These prompts are original to this repository; they mirror the paper's coverage strategy rather than copying its prompt list.

The generated `report.json` remains an agent claim until it has been independently coded and replicated. The nominal p-value does not correct for adaptive search or multiple hypotheses.

## Recommended: Codex subscription + autonomous Kaggle bridge

This route does not use the OpenAI API. A fresh Local Codex task is the investigator under the user's ChatGPT/Codex subscription. It calls three local MCP tools:

- `get_investigation_status` returns the seed and remaining budget.
- `query_models` starts a private Kaggle GPU script, waits for it, downloads paired A/B answers, and records them.
- `finish_investigation` validates and freezes the final report.

Kaggle remains an asynchronous batch backend. Each `query_models` call creates one Kaggle kernel version, so use the protocol's full 1-5 prompt batch rather than treating it as a low-latency chat endpoint.

### 1. Persist the target artifacts

Files left only under `/kaggle/working` disappear when that session ends. Save the completed training artifacts as a private Kaggle Dataset, Kaggle Model, or saved notebook output, then attach that source to the worker. It must expose the usual layout, for example:

```text
/kaggle/working/artifacts/
  M0/manifest.json
  M1/seed_42/{manifest.json,adapter/}
  M2/seed_42/{manifest.json,adapter/}
  M3/seed_42/{manifest.json,adapter/}
```

The worker installs the target inference requirements without replacing Kaggle's PyTorch/CUDA build:

```bash
pip install --upgrade -r requirements-model-diffing.txt
```

There is deliberately no `openai` dependency in this worker requirements file.

### 2. Install and authenticate the local bridge

Use Python 3.11 or newer because the current official Kaggle CLI requires it:

```powershell
python -m pip install -e ".[bridge,dev]"
kaggle auth login
```

Authentication is a one-time local Kaggle action. Do not paste an access token into a prompt, repository file, or bridge YAML.

### 3. Create the private bridge configuration

Copy `kaggle_bridge.example.yaml` to a directory that will **not** be attached to the blind Codex project. Edit:

- the unique session ID and private session-state path;
- the two real target IDs;
- `kaggle.kernel_ref` with your private worker slug;
- the attached artifact data/model/kernel source;
- `worker.artifact_root` to match its actual `/kaggle/input/...` mount;
- the repository mode and source, using an immutable Git commit where practical.

Git mode clones this repository for each isolated job. Set `github_token_secret` to the name of a Kaggle Secret only when the repository is private; the token itself stays in Kaggle. Path mode instead reads a repository snapshot attached as a Kaggle Dataset.

Check local configuration and then Kaggle authentication:

```powershell
model-diffing-kaggle --config C:\private\kaggle_bridge.yaml preflight
model-diffing-kaggle --config C:\private\kaggle_bridge.yaml preflight --online
```

The online preflight only performs a read-only listing call. It does not start a GPU job.

### 4. Connect the blind Codex workspace

Use `blind_workspace/` as the primary folder of a new Local Codex project. Copy `blind_workspace/codex-mcp.example.toml` to `blind_workspace/.codex/config.toml`, replace its Python, repository, and private bridge-config paths, and restart Codex. The 7,200-second MCP tool timeout is intentional because Kaggle jobs are asynchronous.

The private YAML, rendered worker jobs, target mapping, artifacts, training data, and previous reports must stay outside the blind workspace. The MCP server returns only the seed, budget, stable pair IDs, and visible anonymous A/B answers.

Start the new task with:

> Run the blinded model-diffing investigation to completion. Follow AGENTS.md and use only the blind-model-diffing tools as experimental evidence.

The bridge uses stable `batch_id` values to make automatic retries idempotent. It checks the output `request_id` before accepting results, so a stale output from an older Kaggle run cannot enter the experiment record. Private CLI and failure logs are kept under the configured session-state directory.

## Preflight

Preflight reads only manifests and adapter paths; it does not load model weights:

```bash
python -m model_diffing preflight \
  --model-a M0 \
  --model-b M3_seed_42
```

Override `--artifact-root` when the restored Kaggle Dataset is mounted somewhere other than `/kaggle/working/artifacts`.

## Optional legacy OpenAI API investigator

The older `agent` command runs the investigator through the OpenAI Responses API and therefore requires API credentials and credits. It is not used by the recommended Codex/MCP bridge. Install `.[agent]` only if this legacy route is intentional.

Run one API-funded investigation with:

The investigator model is explicit so a mutable default cannot silently change across runs:

```bash
python -m model_diffing agent \
  --model-a M0 \
  --model-b M3_seed_42 \
  --session-id m0_vs_m3_s42_seed001 \
  --seed-id seed_001 \
  --investigator-model YOUR_CODEX_API_MODEL
```

You may set `OPENAI_MODEL` instead of passing `--investigator-model`. List seed prompts with:

```bash
python -m model_diffing seeds
```

Run identical-target controls before interpreting real comparisons:

```bash
python -m model_diffing agent \
  --model-a M0 \
  --model-b M0 \
  --session-id identical_m0_seed001 \
  --seed-id seed_001 \
  --investigator-model YOUR_CODEX_API_MODEL
```

Each legacy API session is append-only and refuses to overwrite an existing session directory. It contains:

```text
manifest.json       frozen configuration, target hashes, and private A/B mapping
events.jsonl        tool calls, visible investigator text, and lifecycle events
observations.jsonl  paired target prompts and visible responses
report.json         difference_found or no_difference_found
```

## Independent validation

Have a blind judge or human coder read validation pairs and assign `A`, `B`, or `neither` according to the frozen hypothesis. Do not show the coder the target mapping. Create a JSONL file following `codings.example.jsonl`, then run:

```bash
python -m model_diffing score \
  --session /kaggle/working/artifacts/model_diffing/m0_vs_m3_s42_seed001 \
  --codings /kaggle/working/validation-codings.jsonl
```

Only rows from calls declared as `phase=validate` enter the statistical check. Repeated samples are collapsed to one majority verdict per prompt before the exact test. A result is marked validated only if it spans the configured minimum number of distinct validation prompts and passes the nominal alpha threshold. Research claims should additionally use independent seed runs, identical-model controls, multiplicity control, and human review.

## Experimental boundaries

- Run each seed as an independent agent trajectory. The paper reports 50 distinct seeds; `method.repeated_seed_runs` records that target but does not launch a costly matrix automatically.
- Do not tune prompts after examining M0-M4 ground-truth training labels.
- Compare the same frozen base/tokenizer revision. Preflight refuses mixed identities.
- Treat response pairs and repeated samples as nested observations, not independent trained-model replicates.
- Do not automatically normalize raw findings into `normalized/model_diff.jsonl`; preserve the blind normalization step described in `INTROSPECTION_ADAPTERS.md`.
- A failed target-behavior discovery can reflect poor elicitation or a weak model organism, not proof that the trained behavior is absent.
