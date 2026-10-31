# Baselines

Trained artifacts of the baselines in the fidelity table, and only the code needed to run them. They are
evaluated with `python -m epb.eval` (see [`../README.md`](../README.md#evaluation)) on the same instance sets as
the students and the teacher: `--ckpt` for Expert and Frozen (router + bank directories, like the students), `--pkg`
for Tree, NAM, MoET and the agents. Where the table used other than the default decoding options, the directory's
`decode.json` holds them, so only the teacher checkpoint (for top-1) has to be supplied.

| directory | baseline | contents |
|---|---|---|
| `expert/<setting>/` | **Expert**: a fixed, hand-specified heuristic bank with a trained router | `bank.json`, `best_model.pt` |
| `frozen/<setting>/` | **Frozen**: the initial bank of the pipeline, never refined, with a trained router | `bank.json`, `best_model.pt` |
| `tree/` | **Tree**: decision-tree surrogate policies (depths 4, 6, 8, 12 and a staged depth-12 tree) | `trees/<setting>.pkl`, rollout and evaluation code |
| `nam/` | **NAM**: a neural additive model, one small MLP per hand-crafted candidate feature | `<setting>/best.pt`, model and feature code |
| `moet/` | **MoET**: a mixture of 8 regression trees over the same candidate features, gated by a logistic regression on state features | `<setting>/best.pt`, model and feature code |
| `agents/fable51/` | **Agent (Fable 5.1)**: a white-box policy written by a coding agent | frozen parameters, policy and rollout code |
| `agents/gpt6/` | **Agent (GPT-6)**: a white-box policy written by a coding agent | policy code, construction / multi-start / RRC code |

## Running a single baseline

```bash
python -m epb.eval --setting pomo_tsp --n 100 --method expert --ckpt epb/baselines/expert/pomo_tsp100
python -m epb.eval --setting pomo_cvrp --n 100 --method tree --pkg epb/baselines/tree            # staged tree
python -m epb.eval --setting pomo_cvrp --n 50 --method agent --pkg epb/baselines/agents/fable51
python -m epb.eval --setting lehd_cvrp --n 100 --method agent --pkg epb/baselines/agents/gpt6   # runs on the CPU
python -m epb.eval --setting pomo_tsp --n 100 --method nam --pkg epb/baselines/nam
python -m epb.eval --setting lehd_cvrp --n 100 --method moet --pkg epb/baselines/moet --threads 10
```

Add `--teacher-ckpt <CKPT> --metrics top1,greedy,search` for the teacher-forced top-1 (LEHD teachers also need
`--nco-root <NCO_code clone>`). Non-default decoding options (key / value modes, program temperature, encoder depth,
batch sizes, device) are read from `decode.json`; a flag overrides them.
