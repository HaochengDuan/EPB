# `epb/counterfactual/`: counterfactual faithfulness tests

Each test edits decision states so that the concept read by a carrier program changes (treatment arm vs a control
arm) and measures the change in the program score, the routing weight, the carrier's contribution, the student and
the teacher.

| file | |
|---|---|
| `__main__.py` | the CLI, `python -m epb.counterfactual <test>` |
| `common.py` | shared code: student / teacher loading, state collection, experiment fingerprints, bootstrap CI and sign-flip p-value |
| `hull.py`, `hull.json` | POMO TSP-100: convex-hull peripherality |
| `isolation.py`, `isolation.json` | LEHD TSP-100: isolation (distance to the nearest other unvisited node) |
| `corridor.py`, `corridor.json` | POMO CVRP-50: homeward corridor |
| `demand_density.py`, `demand_density.json` | POMO CVRP-50: local unvisited demand |

The `.json` files hold each test's parameters (program slots, thresholds, instance ranges); `--config` replaces them.

## Running

From the repository root, with `rl4co` importable and a CUDA GPU. `--teacher-ckpt` is a POMO checkpoint
([yd-kwon/POMO](https://github.com/yd-kwon/POMO) format; TSP-100 for hull, CVRP-50 for corridor and demand density)
or, for isolation, a LEHD-TSP checkpoint ([CIAM-Group/NCO_code](https://github.com/CIAM-Group/NCO_code) format) with
`--nco-root <NCO_code clone>`. The student is the matching checkpoint of `epb/students` (`--student` to change it).

```
python -m epb.counterfactual hull --teacher-ckpt <POMO_TSP100_CKPT> --out OUT
python -m epb.counterfactual isolation --teacher-ckpt <LEHD_TSP_CKPT> --nco-root <NCO_code> --out OUT \
    --heldout-cache OUT/heldout.pt --decompose --save-triplets
python -m epb.counterfactual corridor --teacher-ckpt <POMO_CVRP50_CKPT> --out OUT
python -m epb.counterfactual demand_density --teacher-ckpt <POMO_CVRP50_CKPT> --out OUT
```

The results are in `OUT/five_values.json` (demand density: `OUT/main_table.{json,md}`). Hull and demand density can
run in shards (`--inst-lo` / `--inst-hi`, then `--merge`), corridor per batch (`--batch A|B`, then `--analyze`);
`python -m epb.counterfactual <test> --help` lists the options. Float32 readouts depend on the batch layout, so keep
the defaults to reproduce them bitwise.
