from __future__ import annotations
import torch
from .evaluate import construct_cvrp, construct_tsp

def construct_tsp_start(locs, kind, start):
    return construct_tsp(locs, kind, start_node=int(start))

def construct_cvrp_start(locs, demand, kind, first_customer):
    return construct_cvrp(locs, demand, kind, first_customer=int(first_customer))

def _collect(locs, demand, kind, starts, chunk_size):
    starts = list(starts)
    if not starts or len(set(starts)) != len(starts):
        raise ValueError("starts must contain distinct, explicitly chosen node IDs")
    if chunk_size < 1 or len(locs) == 0:
        raise ValueError("positive chunk size and nonempty instances required")
    columns = []
    for start in starts:
        chunks = []
        for left in range(0, len(locs), chunk_size):
            x = locs[left:left+chunk_size]
            if demand is None:
                costs, _ = construct_tsp_start(x, kind, start)
            else:
                costs, _ = construct_cvrp_start(x, demand[left:left+chunk_size], kind, start)
            chunks.append(costs)
        columns.append(torch.cat(chunks))
    per_start = torch.stack(columns, -1)
    best_cost, best_column = per_start.min(-1)
    start_ids = torch.tensor(starts, device=per_start.device, dtype=torch.long)
    return {"costs_per_start": per_start, "best_cost": best_cost,
            "best_start": start_ids[best_column], "starts": starts}

@torch.no_grad()
def multistart_tsp(locs, kind, starts, chunk_size=128):
    return _collect(locs, None, kind, starts, chunk_size)

@torch.no_grad()
def multistart_cvrp(locs, demand, kind, starts, chunk_size=128):
    return _collect(locs, demand, kind, starts, chunk_size)

