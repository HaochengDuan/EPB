from __future__ import annotations
import torch

def _check_probs(probs, feasible):
    if probs.shape != feasible.shape or not torch.isfinite(probs).all():
        raise AssertionError("Invalid probability shape or nonfinite value")
    if (probs < 0).any() or (probs[~feasible] != 0).any():
        raise AssertionError("Negative or infeasible probability")
    expected = torch.ones(probs.shape[0], dtype=probs.dtype, device=probs.device)
    if not torch.allclose(probs.sum(-1), expected, atol=1e-5):
        raise AssertionError("Action probabilities do not sum to one")

def tsp_cost(locs, tour):
    expected = torch.arange(locs.shape[1], device=locs.device).expand_as(tour)
    if not torch.equal(tour.sort(-1).values, expected):
        raise AssertionError("TSP tour is not a permutation")
    xy = locs.gather(1, tour[..., None].expand(-1, -1, 2))
    return (xy - xy.roll(-1, dims=1)).norm(dim=-1).sum(-1)

@torch.no_grad()
def construct_tsp(locs, kind, distribution_fn=None, start_node=0):
    if distribution_fn is None:
        from . import tsp
        distribution_fn = tsp.pomo_action_distribution if kind == "pomo" else tsp.lehd_action_distribution
    b, n, _ = locs.shape
    rows = torch.arange(b, device=locs.device)
    if not 0 <= start_node < n:
        raise ValueError("start_node must index a TSP node")
    first = torch.full((b,), start_node, device=locs.device, dtype=torch.long)
    current = first.clone()
    visited = torch.zeros(b, n, device=locs.device, dtype=torch.bool)
    visited[:, start_node] = True
    tour = torch.empty(b, n, device=locs.device, dtype=torch.long)
    tour[:, 0] = start_node
    for step in range(1, n):
        feasible = ~visited
        if kind == "pomo":
            progress = torch.full((b,), (step - 1) / max(n - 1, 1), device=locs.device)
            probs = distribution_fn(locs, current, feasible, visited.float(), progress, first)
            _check_probs(probs, feasible)
            selected = probs.argmax(-1)
        elif kind == "lehd":
            ids = feasible.nonzero(as_tuple=False)[:, 1].reshape(b, n - step)
            candidates = locs.gather(1, ids[..., None].expand(-1, -1, 2))
            probs = distribution_fn(candidates, locs[rows, first], locs[rows, current])
            _check_probs(probs, torch.ones_like(ids, dtype=torch.bool))
            selected = ids.gather(1, probs.argmax(-1)[:, None]).squeeze(1)
        else:
            raise ValueError(kind)
        visited[rows, selected] = True
        current = selected
        tour[:, step] = current
    return tsp_cost(locs, tour), tour

def validate_cvrp(locs, demand, route, lengths):
    b, n1, _ = locs.shape
    rows = torch.arange(b, device=locs.device)
    visits = torch.zeros(b, n1, dtype=torch.long, device=locs.device)
    remaining = torch.ones(b, device=locs.device)
    total = torch.zeros(b, device=locs.device)
    if (route[:, 0] != 0).any() or (route[rows, lengths - 1] != 0).any():
        raise AssertionError("CVRP routes must start and end at the depot")
    for t in range(1, int(lengths.max())):
        active = t < lengths
        sub = rows[active]
        if not len(sub):
            continue
        prev, nxt = route[sub, t - 1], route[sub, t]
        total[sub] += (locs[sub, prev] - locs[sub, nxt]).norm(dim=-1)
        depot = nxt == 0
        remaining[sub] = torch.where(depot, 1.0, remaining[sub] - demand[sub, nxt])
        if (remaining[sub] < -1e-5).any():
            raise AssertionError("CVRP capacity violated")
        served = sub[~depot]
        visits[served, nxt[~depot]] += 1
    if not (visits[:, 1:] == 1).all():
        raise AssertionError("Each CVRP customer must be served exactly once")
    return total

@torch.no_grad()
def construct_cvrp(locs, demand, kind, distribution_fn=None, first_customer=None):
    if distribution_fn is None:
        from . import cvrp
        distribution_fn = cvrp.pomo_action_distribution if kind == "pomo" else cvrp.lehd_action_distribution
    b, n1, _ = locs.shape
    n = n1 - 1
    if n < 1 or demand.shape != (b, n1) or (demand[:, 1:] <= 0).any() or (demand > 1).any():
        raise ValueError("CVRP expects feasible positive normalized customer demands")
    rows = torch.arange(b, device=locs.device)
    current = torch.zeros(b, dtype=torch.long, device=locs.device)
    selected_first = first_customer if first_customer is not None else (1 if kind == "pomo" else None)
    if selected_first is not None and not 1 <= selected_first <= n:
        raise ValueError("first_customer must index a CVRP customer")
    first = torch.full((b,), selected_first or 1, dtype=torch.long, device=locs.device)
    visited = torch.zeros(b, n1, dtype=torch.bool, device=locs.device)
    load = torch.ones(b, device=locs.device)
    route = torch.zeros(b, 2 * n + 3, dtype=torch.long, device=locs.device)
    lengths = torch.ones(b, dtype=torch.long, device=locs.device)
    if selected_first is not None:
        current[:] = selected_first
        visited[:, selected_first] = True
        load -= demand[:, selected_first]
        route[:, 1] = selected_first
        lengths[:] = 2
    if kind not in ("pomo", "lehd"):
        raise ValueError(kind)
    for _ in range(2 * n + 1):
        alive = (~visited[:, 1:].all(-1)).nonzero(as_tuple=False).flatten()
        if not len(alive):
            break
        if kind == "pomo":
            feasible = (~visited[alive]) & (demand[alive] <= load[alive, None] + 1e-6)
            feasible[:, 0] = current[alive] != 0
            probs = distribution_fn(locs[alive], demand[alive], current[alive], feasible,
                                    visited[alive].float(), load[alive], first[alive])
            _check_probs(probs, feasible)
            groups = [(alive, probs.argmax(-1))]
        else:
            counts = (~visited[alive, 1:]).sum(-1)
            groups = []
            for k_tensor in counts.unique():
                k = int(k_tensor)
                sub = alive[counts == k]
                candidate_mask = ~visited[sub, 1:]
                ids = candidate_mask.nonzero(as_tuple=False)[:, 1].reshape(len(sub), k) + 1
                unsel = locs[sub[:, None], ids]
                dem = demand[sub[:, None], ids]
                feasible = dem <= load[sub, None] + 1e-6
                probs = distribution_fn(unsel, dem, locs[sub, current[sub]], locs[sub, 0],
                                        load[sub], feasible)
                local_valid = torch.cat((feasible, (current[sub] != 0)[:, None]), dim=1)
                _check_probs(probs, local_valid)
                selected = probs.argmax(-1)
                selected_ids = ids.gather(1, selected.clamp_max(k - 1)[:, None]).squeeze(1)
                actions = torch.where(selected == k, 0, selected_ids)
                groups.append((sub, actions))
        for sub, actions in groups:
            depot = actions == 0
            route[sub, lengths[sub]] = actions
            lengths[sub] += 1
            served = sub[~depot]
            visited[served, actions[~depot]] = True
            load[sub] = torch.where(depot, 1.0, load[sub] - demand[sub, actions])
            current[sub] = actions
    if not visited[:, 1:].all():
        raise AssertionError("CVRP construction failed to finish within 2N+1 decisions")
    route[rows, lengths] = 0
    lengths += 1
    costs = validate_cvrp(locs, demand, route, lengths)
    return costs, (route, lengths)

