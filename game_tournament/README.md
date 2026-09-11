# Repeated-game tournament

## Scope

This self-contained package lets every completed model checkpoint play every other checkpoint and itself in the Prisoner's Dilemma, Stag Hunt, and Battle of the Sexes. It is a sibling of `safety_training`; training and data-preparation modules never import game prompts or results.

The canonical Prisoner's Dilemma and Battle of the Sexes matrices and prompt structure come from Akata et al. (2025), *Playing repeated games with large language models*, DOI: [10.1038/s41562-025-02172-y](https://doi.org/10.1038/s41562-025-02172-y). The canonical and cooking-competition prompt templates are adapted from the authors' MIT-licensed repository pinned at commit `224a605a21127cac69763f990e077f7b05abd422`. See `THIRD_PARTY_NOTICES.md`.

Stag Hunt uses the same prompt structure with a study-specific matrix: mutual Stag `(10, 10)`, Stag/Hare `(0, 7)`, Hare/Stag `(7, 0)`, and mutual Hare `(5, 5)`. Every artifact states whether its role-specific instruction is present verbatim upstream and when payoffs, role mirroring, action mapping, labels, or horizon were adapted.

## Default design

`game_tournament/config.yaml` selects M0 and M1-M3 at seeds 42, 123, and 456. Each condition-seed checkpoint is an agent, producing ten agents. Add M4 explicitly after its corresponding adapters have been trained.

For each game, the scheduler constructs every ordered pairing, including self-play. A-versus-B and B-versus-A are separate episodes. Each player has an independent history and sampling stream. Both current-round prompts are finalized before either decision is made, so sequential GPU execution does not leak one player's current action to the other.

The primary defaults are 48 independent episodes per ordered matchup and ten rounds per episode. The prompt grid contains 24 variants: two paper-derived profiles, six published neutral letter pairs, and both mappings between latent game actions and letters. Forty-eight episodes cover that grid twice for every ordered matchup. The order in which the two options appear in each round's question is independently seeded and randomized as in the source code.

The linked cooking robustness script offered three outcome vocabularies. The primary configuration keeps `points` for both cover stories to isolate the cover-story manipulation. To reproduce the complete published vocabulary grid, add the other allowed labels in YAML:

```yaml
prompting:
  outcome_labels:
    paper_canonical: [points, dollars, coins]
    paper_cooking: [points, audience votes, prize tokens]
```

The `paper_project` profile is also implemented and can be added to `prompting.profiles`, with its published `credits`, `reputation points`, and `bonus rewards` labels.

## Inference protocol

The model is asked for the same one-letter continuation used by the paper. Instead of unconstrained text parsing, the runner reads the next-token logits for the two permitted letters, normalizes them, and samples one. Each published option label must be one token for the pinned tokenizer or preflight fails.

Episodes from the same matchup advance round-by-round in batches. The base checkpoint is loaded once, LoRA adapters are registered under distinct names, and only final-position logits are retained. This keeps all decisions simultaneous at the experimental level while avoiding redundant model copies and full sequence-by-vocabulary logits.

Temperature is deliberately not set to zero. `sampling.temperature: null` means no logit rescaling, which is an effective temperature of `1.0`. This deviation from the paper is recorded in every tournament manifest. Qwen thinking output is disabled because the experimental response is a one-token action, not a reasoning trace.

## Running on Kaggle

The runner discovers completed artifacts from the same artifact root used by training and verifies that their recorded model and tokenizer revisions agree.

```bash
python -m game_tournament.run --dry-run
python -m game_tournament.run
python -m game_tournament.analyze
```

For the optional M4 condition:

```bash
python -m game_tournament.run --conditions M0 M1 M2 M3 M4
```

For an integration smoke test or a bounded Kaggle session:

```bash
python -m game_tournament.run --smoke-test
python -m game_tournament.run --max-episodes 100
```

Completed episode shards are skipped automatically. Re-run the same command to resume. If the config or model-manifest hashes change, the runner refuses to mix results; select a new `tournament_id` instead.

## One-player constitutional prompt intervention

The ordinary all-pairs tournament is the unprompted control arm. A separate runner
adds the constitution from `configs/constitution.yaml` as a system message for only
the focal player. The opponent is always the same unprompted M0 baseline. Every
selected checkpoint is tested with the focal model in both player seats, while the
game, prompt variant, option-order stream, and sampling stream are matched to the
corresponding episode in the ordinary tournament.

For a seed-42 run matching an ordinary seed-42 tournament:

```bash
python -m game_tournament.run_prompt_intervention \
  --conditions M0 M1 M2 M3 \
  --seeds 42 \
  --dry-run
python -m game_tournament.run_prompt_intervention \
  --conditions M0 M1 M2 M3 \
  --seeds 42
python -m game_tournament.analyze_prompt_intervention
```

The intervention runner creates only the prompted arm; it does not repeat the
unprompted games. At the 48-run default, four seed-42 focal checkpoints produce
1,152 new episodes. Use the same `--runs` and `--rounds` values as the control
tournament. The analysis refuses to compare episodes unless the exact paired
control IDs exist. It writes auditable per-episode differences to
`paired_episode_effects.jsonl`; `paired_analysis.json` reports unprompted means,
prompted means, paired deltas, and paired control-episode-cluster bootstrap
intervals with player seats aligned as focal versus baseline.

For the paired smoke test, both the control and intervention smoke tournaments must
exist before analysis:

```bash
python -m game_tournament.run --conditions M0 M1 M2 M3 --seeds 42 --smoke-test
python -m game_tournament.run_prompt_intervention \
  --conditions M0 M1 M2 M3 --seeds 42 --smoke-test
python -m game_tournament.analyze_prompt_intervention \
  --control-root /kaggle/working/artifacts/game_tournaments/paper_prompts_v1_smoke \
  --intervention-root /kaggle/working/artifacts/game_tournaments/constitutional_prompt_vs_m0_v1_smoke
```

Outputs are written under `artifacts/game_tournaments/<tournament_id>/`:

```text
manifest.json
episode_shards/<game>/<ordered-matchup>/run_NNN.json
episodes.jsonl
rounds.jsonl
analysis.json
```

Each shard records the game matrix, exact role-specific instructions, prompt provenance, prompt hashes for every round, option ordering, action probabilities and logits, sampled action, payoffs, cumulative payoffs, RNG seeds, and model-manifest hashes. Complete prompts are reconstructable from the instruction, prior recorded rounds, and recorded query ordering without duplicating the growing history in every row.

## Interpretation

The analysis reports episode-level behavior for exact checkpoint matchups and condition-level matchups. Confidence intervals resample complete episodes rather than treating dependent rounds as independent observations. They are descriptive at the condition level: the number of independently trained seeds, not the number of game rounds, limits claims about training-method effects.
