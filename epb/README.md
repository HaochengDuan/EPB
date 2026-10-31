# `epb/`

The training, evaluation and analysis code of EPB, with the released checkpoints. Run every command from the
repository root (the directory that holds `epb/`) with `rl4co` importable. Teacher checkpoints are not included:
POMO checkpoints are in the format of [yd-kwon/POMO](https://github.com/yd-kwon/POMO), LEHD checkpoints in the format
of [CIAM-Group/NCO_code](https://github.com/CIAM-Group/NCO_code), whose model code is read from a clone of that
repository (`--nco-root`, `generate.nco_root` or `NCO_ROOT`).

| path | |
|---|---|
| `train.py`, `generate.py`, `bank.py`, `llm.py`, `program_budget.py`, `feats.py` | training: `python -m epb.generate`, `python -m epb.train` |
| `configs/<setting><N>.yaml`, `prompts/<setting>.yaml` | the experiment configs of the paper; the LLM prompts of each setting |
| `settings/<setting>.py` | everything that depends on the setting: data, router, program interface and checks, evaluation |
| `teachers/<setting>.py` | the teacher models and their decision-step generators |
| `eval.py`, `instances.py` | `python -m epb.eval`; the seeded generators of the test sets |
| `students/` | the released students (`best_model.pt`, `bank.json`, `decode.json`) |
| [`baselines/`](baselines/README.md) | Expert, Frozen, Tree, NAM, MoET and two coding-agent policies |
| `figures/` | `python -m epb.figures`: the routing figures |
| [`interventions/`](interventions/README.md) | `python -m epb.interventions`: the edited students and their controls |
| [`counterfactual/`](counterfactual/README.md) | `python -m epb.counterfactual`: the counterfactual tests |

`<setting>` is `pomo_tsp`, `pomo_cvrp`, `lehd_tsp` or `lehd_cvrp`.

## Training

```
python -m epb.generate --config epb/configs/pomo_cvrp50.yaml --set generate.teacher_ckpt=<POMO_CVRP50_CKPT>
python -m epb.train    --config epb/configs/pomo_cvrp50.yaml
python -m epb.train    --config epb/configs/pomo_cvrp50.yaml --set train.lr=3e-4 --set llm.max_cost_usd=10
python -m epb.train    --resume runs/pomo_cvrp50
```

`epb.generate` writes the teacher's decision steps to `data.path` (the LEHD settings also need `generate.nco_root`).
`epb.train` trains the router every epoch; on event epochs it rewrites programs with TextGrad, or, when the val top-1
has plateaued, drops and adds programs. The config is merged as code defaults < setting defaults < experiment YAML <
`--set key=value`, and an unknown key is an error:

| section | keys |
|---|---|
| `setting` | `name` |
| `run` | `output_dir`, `seed` |
| `data` | `path` (file or sharded directory), `val_ratio`, `num_workers` |
| `generate` | `teacher_ckpt`, `num_instances`, `seed`, plus setting keys (`problem_size`, `batch_size`, `nco_root`, ...) |
| `init` | `bank` (null: the LLM writes the initial bank), `router` (null: from scratch) |
| `bank` | `num_programs`, `min_programs`, `max_programs`, `tau_h` |
| `router` | `embed_dim`, `num_heads`, layer counts |
| `train` | `epochs`, `batch_size`, `lr`, plus setting keys `scheduler` and `amp` |
| `loss` | `focal`, `routing` (optional terms, default 0: KL only) |
| `textgrad` | `enabled`, `start_epoch`, `every_n_epochs`, `backward_every_n`, `buffer_size`, `rounds`, `min_improvement`, `eval_batches` |
| `plateau` | `window`, `min_gain` |
| `drop` | `enabled`, `grace_epochs`, `top1_tol`, `kl_tol`, `max_per_event`, `cooldown_epochs` |
| `add` | `enabled`, `n_candidates`, `min_gain`, `min_faults`, `pretrain_epochs`, `finetune_epochs`, `cooldown_epochs` |
| `llm` | `model`, `reasoning_effort`, `max_tokens`, `max_workers`, `max_cost_usd`, `price_input`, `price_output`, `price_cache_read` |
| `prompts` | `file`, `teacher` |
| `blind` | `enabled` |
| `program_budget` | `tier` (`off`, `tiny`, `simple`, `medium`; TSP settings), `max_agreement` |

The LLM backend follows `llm.model`: `claude-*` (`ANTHROPIC_API_KEY`), `gpt-*` / `o<N>*` (`OPENAI_API_KEY`),
`deepseek-*` (`DEEPSEEK_API_KEY`). Set `llm.price_*` (USD per 1M tokens) to the model's price: the spend cap is
computed from them. `blind.enabled: true` replaces the teacher's name in the prompts and aborts the run if it reaches
the LLM. `program_budget.tier` caps the size of every program.

A run directory holds `best_model.pt` + `heuristics_best.json` (lowest val KL: the student to evaluate),
`last_model.pt` + `heuristics_last.json` (the point `--resume` continues from), `decode.json`, the bank after every
event, `metrics.jsonl` and `api_metrics.jsonl`.

## Evaluation

```
python -m epb.eval --setting pomo_tsp --n 100 --method teacher --teacher-ckpt <POMO_TSP100_CKPT>
python -m epb.eval --setting pomo_tsp --n 100 --method epb --ckpt epb/students/pomo_tsp100 \
    --teacher-ckpt <POMO_TSP100_CKPT> --metrics top1,greedy,search
python -m epb.eval --setting lehd_tsp --n 100 --method expert --ckpt epb/baselines/expert/lehd_tsp100 --nco-root <NCO_code>
python -m epb.eval --setting pomo_cvrp --n 50 --method nam --pkg epb/baselines/nam
python -m epb.eval --setting lehd_cvrp --instances n500 --method agent --pkg epb/baselines/agents/gpt6 --slice 0:32
python -m epb.eval --table results
```

| method | needs |
|---|---|
| `teacher` | `--teacher-ckpt` (LEHD: also `--nco-root`) |
| `epb`, `expert`, `frozen` | `--ckpt DIR` (a released checkpoint or an `epb.train` run directory), or a `.pt` with `--bank` |
| `nam`, `moet` | `--pkg epb/baselines/nam` or `.../moet` |
| `tree` | `--pkg epb/baselines/tree`, `--depth` |
| `agent` | `--pkg epb/baselines/agents/<agent>` |

- `--instances`: `main` (the main-table test set, default), `n500` (N = 500 zero-shot), `band:<set>` (the
  interventions) or `file:<path>`. The first three are generated from fixed seeds.
- `--metrics`: `greedy` (one construction), `search` (POMO: the best of all N start nodes; LEHD: random re-construct
  for `--budget` iterations), `top1` (teacher-forced agreement with the teacher's argmax; needs `--teacher-ckpt`).
- `decode.json` next to a checkpoint holds its decoding options (key / value modes, `tau_h`, programs left out,
  batch layout, device); a flag overrides it. Where it sets no key or value mode, `--k-mode` / `--v-mode` choose
  `soft` (default) or `onehot`.
- Results go to `results/<setting><N>[_<set>]/<method>[_<tag>].pt` (per-instance costs). `--slice lo:hi` splits a
  run into shards; `--table` merges them and reports the gaps to the teacher (100 · (mean cost / mean teacher cost −
  1)) and the top-1 rates with bootstrap CIs.

## Figures, interventions and counterfactual tests

```
python -m epb.figures stage --ckpt epb/students/pomo_cvrp50
python -m epb.figures tour  --ckpt epb/students/pomo_tsp100 [--instance 9] [--teacher-ckpt <POMO_TSP100_CKPT>]
python -m epb.interventions band_tsp100 --teacher-ckpt <POMO_TSP100_CKPT>
python -m epb.counterfactual hull --teacher-ckpt <POMO_TSP100_CKPT>
```

`stage` stacks the dominant program of the POMO CVRP-50 student by used capacity; `tour` draws a POMO TSP-100 tour
with every edge coloured by the dominant program of its decision. The interventions and the counterfactual tests
have their own READMEs ([`interventions/`](interventions/README.md), [`counterfactual/`](counterfactual/README.md)).
