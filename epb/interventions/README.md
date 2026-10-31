# `epb/interventions/`

Hand edits of a student's bank for instances on a thin diagonal band; the router and every weight are unchanged.
This module holds the edited students and runs the controls: POMO given the same rule(s), the rules alone (CVRP)
and the divergence of the edited student from the rules.

| file | |
|---|---|
| `pomo_tsp100_band/` | `students/pomo_tsp100` with its bank edited (17 programs: 3 appended, a rule written into all) |
| `pomo_cvrp50_band/` | `students/pomo_cvrp50` with its bank edited (the 20 programs with three rules injected) |
| `band_tsp100.py`, `band_cvrp50.py` | the rules and the controls of each case |
| `common.py`, `__main__.py` | the runs and their result files; the CLI |
| `rules/` | the rule given to the teacher (TSP); the band rule of the rules-alone policy (CVRP) |

## Running

```
python -m epb.eval --setting pomo_tsp --method epb --ckpt epb/interventions/pomo_tsp100_band --instances band:diag_band_s3 \
    --batch 64 --ms-chunk 4 --starts-per-chunk 50
python -m epb.interventions band_tsp100 --teacher-ckpt <POMO_TSP100_CKPT>
python -m epb.interventions band_cvrp50 --teacher-ckpt <POMO_CVRP50_CKPT>
python -m epb.eval --table results
```

The original student (`--ckpt epb/students/pomo_tsp100`, `pomo_cvrp50`) and the teacher are evaluated the same way
on every band set (`band_tsp100`: `diag_band_s0`, `_s3`, `_s5`, `uniform_s3`; `band_cvrp50`: `diag_band_s1`..`_s4`,
`uniform_s0`..`_s2`; see `instances.py`). The TSP runs use the batch layout above.
