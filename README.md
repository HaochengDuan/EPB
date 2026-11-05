
# [NeurIPS 2026] Interpreting Neural Combinatorial Optimization via Evolving Programmatic Bottlenecks 

EPB distills black-box neural combinatorial optimization (NCO) policies into interpretable surrogates. An EPB student combines a bank of `M` interpretable, executable, and editable decision-rule programs with a state-dependent attention router to support trustworthy decision-making.

<p align="center">
  <img width="324" alt="epb_tsp_tour_direct_labels" src="https://github.com/user-attachments/assets/e86cbc61-ea5c-4365-9762-e0a2ea9a788d" />
  <br />
  <em>Figure 1: Interpretable Tour Construction with EPB</em>
</p>

Each program produces an action distribution, and the router forms a convex mixture to approximate the teacher’s per-step action distribution. At a fixed bank size, Hybrid Gradient Descent (Figure 2) combines LLM-driven program revision via TextGrad with numerical gradient updates to the router. **Add** introduces programs targeting unresolved failure modes, while **Drop** removes redundant programs. We evaluate EPB with POMO and LEHD on TSP and CVRP.

<p align="center">
  <img width="500" alt="EPB hybrid gradient descent"
       src="https://github.com/user-attachments/assets/0449ac37-f997-455e-95eb-05b87bacd0fa" />
  <br />
  <em>Figure 2: Hybrid Gradient Descent</em>
</p>


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

🧬 Let’s Evolve!

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


## Citation

If you find our paper or code useful, please consider citing our work:

```bibtex
@inproceedings{duan2026interpreting,
  title     = {Interpreting Neural Combinatorial Optimization via Evolving Programmatic Bottlenecks},
  author    = {Duan, Haocheng and Guo, Yuxin and Bi, Jieyi and Xie, Anqi and Li, Sirui and Ma, Yining and Wu, Cathy},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
