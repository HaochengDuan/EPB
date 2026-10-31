"""Settings of the tree-surrogate baseline: problem, family, size, and the search protocol of the fidelity table.
Evaluation instances come from the caller (python -m epb.eval); held_out_instances is not used there."""

SETTINGS = {
    "pomo_tsp50": dict(family="pomo", problem="tsp", N=50, search="multi_start", search_cfg=dict(starts="all_N")),
    "pomo_tsp100": dict(family="pomo", problem="tsp", N=100, search="multi_start", search_cfg=dict(starts="all_N")),
    "pomo_cvrp50": dict(family="pomo", problem="cvrp", N=50, pomo_size=50, search="multi_start",
                        search_cfg=dict(starts="pomo_size=50")),
    "pomo_cvrp100": dict(family="pomo", problem="cvrp", N=100, pomo_size=100, search="multi_start",
                         search_cfg=dict(starts="pomo_size=100")),
    "lehd_tsp100": dict(family="lehd", problem="tsp", N=100, search="rrc",
                        search_cfg=dict(budget=200, sub_min=4, rrc_seed=1334)),
    "lehd_cvrp100": dict(family="lehd", problem="cvrp", N=100, search="rrc",
                         search_cfg=dict(budget=200, sub_min=4, rrc_seed=1334)),
}


def held_out_instances(key, device, which="main"):
    raise RuntimeError("instances are supplied by the caller (python -m epb.eval)")
