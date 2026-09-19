# EmbedPlan — reproduction code

Minimal code accompanying *Textual Planning with Explicit Latent Transitions*.
Learns an action-conditioned transition function in a **frozen LLM embedding space**
and evaluates it by nearest-neighbour retrieval over a candidate pool.

No data, embeddings, checkpoints, or result files are included. Everything below
regenerates from source.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. A GPU is required for encoding and recommended for training
(each transition network trains in minutes; encoding a domain with a 70B encoder
takes hours).

Always invoke drivers as modules **from the repository root** — they import
`embedplan` by package name and there is no installed egg:

```bash
python -m experiments.train --domain ferry --split_type random --seed 0
```

Data and result roots default to `./data` and `./results`; override with the
`EMBEDPLAN_DATA` and `EMBEDPLAN_RESULTS` environment variables.

## Layout

```
embedplan/        library — all reusable logic
  config.py         paths, DOMAINS, factorized data + embedding loaders
  data.py           FactorizedTripletDataset, splits, trajectory construction
  models.py         TransitionMLP / TransitionHyper / ProjectionHead
  training.py       train_transition
  losses.py         InfoNCE + action-disambiguation loss
  scoring.py        hit_at_k, ABSOLUTE / DELTA scoring, project_pool
  evaluation.py     pool_sweep, matched_pool_eval, open_set_abstention
  paper_protocol.py the published evaluator (see "Evaluation protocol" below)
  rollout.py        multi-step closed-loop rollout
  finetune.py       LoRA / frozen encoder fine-tuning
  encoders.py       encoder wrappers
  baselines.py      identity / offset (grounded) / offset (lifted) floors
  symbolic.py       lifted STRIPS operator induction
  prompts.py        state -> prompt text

tools/            data generation + embedding encoding
experiments/      experiment drivers   (python -m experiments.X)
analysis/         aggregation + tables (python -m analysis.X)
scripts/          sweep runners and a SLURM submitter
configs/          W&B + checkpoint/eval cadence
```

## Pipeline

**1. Generate transition data** from PDDL domains (ACPBench sources):

```bash
python -m tools.generate_data --domains ferry --generation_only
```

**2. Encode states and actions** with a frozen encoder:

```bash
python -m tools.encode_data    --domains ferry --embeddings_model_name BAAI/bge-m3
python -m tools.encode_actions --embeddings_model_name BAAI/bge-m3
```

Encoders used in the paper: `sentence-transformers/all-mpnet-base-v2` (768),
`BAAI/bge-m3` (1024), `Qwen/Qwen2.5-7B-Instruct` (3584),
`meta-llama/Llama-3.3-70B-Instruct` (8192, default).

**3. Train and evaluate:**

```bash
python -m experiments.train --domain ferry --split_type random --seed 0
```

This writes one JSON holding `E1_pool_sweep` (Hit@k at pool sizes
128/512/2048/8192/full), `E3_open_set`, and `E4_closed_loop`.

## Splits — the most important knob

| flag | protocol in the paper | what it measures |
|---|---|---|
| `--split_type random` | **Interpolation** | transitions shuffled; the same problem instance appears on both sides. Near-ceiling. Not a generalization measurement. |
| `--split_type problem_grouped` | **Extrapolation** | whole problem instances held out. Every generalization claim rests on this. |

Never report a `random`-split number without labelling it interpolation.

The flag spelling differs by driver, for historical reasons: `--split_type` in
`experiments/train.py` and `train_multi_domain.py`, `--split` in
`finetune_encoder.py` and `fact_order.py`, `--splits` in `baseline_sweep.py` and
`llm_transition.py`. The values are the same everywhere.

## Evaluation protocol

`embedplan/paper_protocol.py` implements the evaluator used for every number in
the paper, and all reference methods are scored through it so the comparison is
about the representation and nothing else:

* pool of 128 = ground truth + 127 distractors;
* under `problem_grouped`, distractors come from the query's **own problem
  instance**; under `random`, uniformly from the domain;
* **worst-case tie-breaking** — a candidate scoring exactly equal to the ground
  truth counts against it.

The tie convention is load-bearing. A representation that collapses distinct
states (for instance, one built from the problem/goal text with the state block
deleted) ties with every candidate; under best-case tie-breaking it would score
100% Hit@5 while carrying no state information, and under this convention it
scores 0%.

## Reproducing the tables

Reference-method comparison (11 arms x domain x split x seed, best-epoch
selection, one JSON per cell, skips completed cells so it resumes):

```bash
python -m experiments.baseline_sweep \
    --domains ferry logistics goldminer \
    --splits problem_grouped random \
    --seeds 0 1 2
python -m analysis.protocol_tables --latex --metric hit@5
python -m analysis.protocol_tables --latex --metric hit@1
```

Arms: `embed_<encoder>`, `tfidf`, `bow`, `char`, `literals` (sparse encoders
through an identical head), `identity`, `offset_grounded`, `offset_lifted`
(non-learned floors), `symbolic` (lifted STRIPS induction, an oracle),
`context`, `random` (null controls).

Encoder fine-tuning (frozen vs LoRA cold start vs LoRA warm start):

```bash
python -m experiments.finetune_encoder --domain ferry --split problem_grouped --seed 0
python -m analysis.warmstart_table
```

Fact-order sensitivity, multi-domain / leave-one-out, direct-LLM baseline:

```bash
python -m experiments.fact_order
python -m experiments.train_multi_domain --eval_protocol loo --test_domains ferry
OPENROUTER_API_KEY=... python -m experiments.llm_transition --domains ferry --splits problem_grouped
```

## Notes

* `experiments/train_multi_domain.py` covers the Multi-Domain and Leave-One-Out
  protocols; `experiments/closed_loop.py` and `closed_loop_pools.py` cover
  multi-step rollout; `experiments/matched_pool.py` re-scores EmbedPlan on the
  exact pools the LLM baseline is ranked against.
* `scripts/*.sh` are plain loops, idempotent (they skip any cell whose result
  JSON exists) and **do not survive the shell dying** — launch long sweeps
  detached and check for the `=== ... DONE ===` sentinel before treating a sweep
  as complete. `scripts/sbatch_job.sh` submits any module as a SLURM job.
* Weights & Biases logging is off by default in `configs/config.yaml`.
