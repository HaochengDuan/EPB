# EPB: Interpreting Neural Combinatorial Optimization via Evolving Programmatic Bottlenecks

An EPB student is a bank of `M` LLM-written heuristic programs plus a small attention router, distilled from the
decisions of a neural constructive teacher (POMO or LEHD, on TSP or CVRP). The router is trained by gradient descent,
the program code is improved by TextGrad, and the bank is pruned and grown by slot **drop** and fault-targeted slot
**add** on plateaus.

```
teacher --> decision-step dataset --> student = router(Q/K) o program bank(V)
                                             |               |
                                     gradient descent    TextGrad (LLM)
                                             +-- slot drop / slot add on plateau --+
```

Everything is in the package [`epb/`](epb/README.md): one training pipeline for the four settings
(`python -m epb.generate`, `python -m epb.train`), the evaluation of the students and every baseline on fixed instance
sets (`python -m epb.eval`), the routing figures (`python -m epb.figures`), the bank interventions
(`python -m epb.interventions`), the counterfactual faithfulness tests (`python -m epb.counterfactual`), the teacher
code (`epb/teachers/`), and the released checkpoints: the students in `epb/students/` and the baselines in
[`epb/baselines/`](epb/baselines/README.md).

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/CIAM-Group/NCO_code.git   # LEHD teachers only (--nco-root / generate.nco_root)
export ANTHROPIC_API_KEY=...      # training with claude-* models
export OPENAI_API_KEY=...         # training with gpt-* / o* models
export DEEPSEEK_API_KEY=...       # training with deepseek-* models
```

Python 3.10, torch built for CUDA 12.8; `requirements.txt` pins the versions the released results were produced
with (float32 results also depend on the torch build and the GPU). `rl4co` provides two stock modules
(`GraphAttentionNetwork`, `TSPContext`) of the POMO routers. The LEHD model code is imported from the NCO_code
clone at run time (`--nco-root`, `generate.nco_root`, or `NCO_ROOT`); nothing else needs to be installed for it. API
keys are needed only for training (evaluation, figures, interventions and counterfactual tests make no LLM call).

**Teacher checkpoints are not included.** Every command that runs a teacher takes its checkpoint as an argument
(`--teacher-ckpt`, `generate.teacher_ckpt`): POMO checkpoints in the format of
[yd-kwon/POMO](https://github.com/yd-kwon/POMO), LEHD checkpoints in the format of
[CIAM-Group/NCO_code](https://github.com/CIAM-Group/NCO_code).

## Usage

Run every command from this directory:

```bash
python -m epb.eval --setting pomo_tsp --n 100 --method epb --ckpt epb/students/pomo_tsp100      # a released student
python -m epb.eval --setting pomo_tsp --n 100 --method teacher --teacher-ckpt <POMO_TSP100_CKPT>
python -m epb.generate --config epb/configs/pomo_cvrp50.yaml --set generate.teacher_ckpt=<POMO_CVRP50_CKPT>
python -m epb.train    --config epb/configs/pomo_cvrp50.yaml
```

[`epb/README.md`](epb/README.md) documents the training pipeline, the evaluation protocol, the figures, the
interventions and the counterfactual tests.

`textgrad` writes a `logs/` directory into the working directory on import; it is ignored by `.gitignore`.
