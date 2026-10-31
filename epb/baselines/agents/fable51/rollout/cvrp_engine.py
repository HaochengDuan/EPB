from __future__ import annotations
import sys, os
from typing import Optional
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policies.cvrp_policy import CVRPParams, cvrp_probs

INF = float("inf")
EPS_POMO = 1e-5

@torch.no_grad()
def _step_probs(locs_full, dem_full, visited, cur, load, p, D_full, eps):
    R = locs_full.size(0)
    ar = torch.arange(R, device=locs_full.device)
    open_mask = ~visited[:, 1:]
    feas_mask = open_mask & (dem_full[:, 1:] <= load.unsqueeze(-1) + eps)
    at_depot = cur == 0
    pr = cvrp_probs(locs_full[:, 1:], dem_full[:, 1:], open_mask, feas_mask, locs_full[ar, cur],
                    locs_full[:, 0], load, at_depot, p, D_full)
    pr = torch.cat([pr[:, -1:], pr[:, :-1]], dim=-1)
    return pr, feas_mask, at_depot

@torch.no_grad()
def rollout_cvrp_pomo(depot_xy, node_xy, node_demand, first_customers, p: CVRPParams, eps=EPS_POMO,
                      max_steps=None, record_seq=False):
    R, N, _ = node_xy.shape
    dev = node_xy.device
    ar = torch.arange(R, device=dev)
    locs_full = torch.cat([depot_xy, node_xy], 1)
    dem_full = torch.cat([torch.zeros(R, 1, device=dev), node_demand], 1)
    D_full = torch.cdist(node_xy, node_xy)
    visited = torch.zeros(R, N + 1, dtype=torch.bool, device=dev)
    visited[:, 0] = True
    cur = torch.zeros(R, dtype=torch.long, device=dev)
    load = torch.ones(R, device=dev)
    cost = torch.zeros(R, device=dev)
    finished = torch.zeros(R, dtype=torch.bool, device=dev)
    seq = [cur.clone()] if record_seq else None

    def move(nxt):
        nonlocal cur, load, cost
        step_cost = (locs_full[ar, nxt] - locs_full[ar, cur]).norm(dim=-1)
        cost = cost + step_cost
        is_dep = nxt == 0
        load = torch.where(is_dep, torch.ones_like(load), load - dem_full[ar, nxt])
        visited[ar, nxt] = True
        visited[:, 0] = True
        cur = nxt
        if record_seq:
            seq.append(cur.clone())

    move(first_customers)
    max_steps = max_steps or (3 * N + 5)
    for _ in range(max_steps):
        all_served = visited[:, 1:].all(-1)
        finished = finished | (all_served & (cur == 0))
        if finished.all():
            break
        pr, feas_mask, at_depot = _step_probs(locs_full, dem_full, visited, cur, load, p, D_full, eps)
        nxt = pr.argmax(-1)
        nxt = torch.where(finished, torch.zeros_like(nxt), nxt)
        move(nxt)
    if not finished.all():
        back = (cur != 0) & ~finished
        cost = cost + torch.where(back, (locs_full[:, 0] - locs_full[ar, cur]).norm(dim=-1), torch.zeros_like(cost))
    if record_seq:
        return cost, torch.stack(seq, 1)
    return cost

@torch.no_grad()
def cvrp_pomo_eval(depot_xy, node_xy, node_demand, p: CVRPParams, pomo_size: int, chunk_rollouts=4096,
                   record_greedy_seq=False):
    B, N, _ = node_xy.shape
    dev = node_xy.device
    P = pomo_size
    per_inst = max(1, chunk_rollouts // P)
    out = torch.zeros(B, P, device=dev)
    seqs = []
    for s in range(0, B, per_inst):
        e = min(s + per_inst, B); b = e - s
        dep = depot_xy[s:e].unsqueeze(1).expand(b, P, 1, 2).reshape(b * P, 1, 2)
        nod = node_xy[s:e].unsqueeze(1).expand(b, P, N, 2).reshape(b * P, N, 2)
        dem = node_demand[s:e].unsqueeze(1).expand(b, P, N).reshape(b * P, N)
        first = torch.arange(1, P + 1, device=dev).repeat(b)
        if record_greedy_seq:
            c, sq = rollout_cvrp_pomo(dep, nod, dem, first, p, record_seq=True)
            seqs.append(sq.view(b, P, -1)[:, 0].cpu())
        else:
            c = rollout_cvrp_pomo(dep, nod, dem, first, p)
        out[s:e] = c.view(b, P)
    if record_greedy_seq:
        T = max(x.size(1) for x in seqs)
        seqs = torch.cat([torch.nn.functional.pad(x, (0, T - x.size(1))) for x in seqs])
        return out, seqs
    return out

@torch.no_grad()
def cvrp_greedy_from_depot(locs_full, dem_full, p: CVRPParams, restrict_to: Optional[torch.Tensor] = None,
                           eps=1e-6, D_full=None):
    B, N1, _ = locs_full.shape
    N = N1 - 1
    dev = locs_full.device
    ar = torch.arange(B, device=dev)
    if D_full is None:
        D_full = torch.cdist(locs_full[:, 1:], locs_full[:, 1:])
    visited = torch.ones(B, N + 1, dtype=torch.bool, device=dev)
    if restrict_to is None:
        visited[:, 1:] = False
        L = N
    else:
        visited.scatter_(1, restrict_to, False)
        visited[:, 0] = True
        L = restrict_to.size(1)
    cur = torch.zeros(B, dtype=torch.long, device=dev)
    load = torch.ones(B, device=dev)
    cost = torch.zeros(B, device=dev)
    perm = torch.zeros(B, L, dtype=torch.long, device=dev)
    n_done = torch.zeros(B, dtype=torch.long, device=dev)
    for _ in range(4 * N + 10):
        remaining = (~visited[:, 1:]).any(-1)
        if not remaining.any():
            break
        pr, feas_mask, at_depot = _step_probs(locs_full, dem_full, visited, cur, load, p, D_full, eps)
        nxt = pr.argmax(-1)
        nxt = torch.where(remaining, nxt, torch.zeros_like(nxt))
        step_cost = (locs_full[ar, nxt] - locs_full[ar, cur]).norm(dim=-1)
        cost = cost + step_cost
        is_dep = nxt == 0
        load = torch.where(is_dep, torch.ones_like(load), load - dem_full[ar, nxt])
        served = remaining & ~is_dep
        perm[ar[served], n_done[served]] = nxt[served]
        n_done = n_done + served.long()
        visited[ar, nxt] = True
        visited[:, 0] = True
        cur = nxt
    back = cur != 0
    cost = cost + torch.where(back, (locs_full[:, 0] - locs_full[ar, cur]).norm(dim=-1), torch.zeros_like(cost))
    assert bool((n_done == L).all()), "not all customers served"
    return perm, cost

@torch.no_grad()
def cvrp_constrained_replay(locs_full, dem_full, perm, p: CVRPParams, eps=1e-6, D_full=None):
    B, N1, _ = locs_full.shape
    N = N1 - 1
    dev = locs_full.device
    ar = torch.arange(B, device=dev)
    if D_full is None:
        D_full = torch.cdist(locs_full[:, 1:], locs_full[:, 1:])
    visited = torch.zeros(B, N + 1, dtype=torch.bool, device=dev)
    visited[:, 0] = True
    cur = torch.zeros(B, dtype=torch.long, device=dev)
    load = torch.ones(B, device=dev)
    cost = torch.zeros(B, device=dev)
    for step in range(perm.size(1)):
        tgt = perm[:, step]
        pr, feas_mask, at_depot = _step_probs(locs_full, dem_full, visited, cur, load, p, D_full, eps)
        p_dep = pr[:, 0]
        p_tgt = pr.gather(1, tgt.unsqueeze(-1)).squeeze(-1)
        need = dem_full[ar, tgt] > load + eps
        via = ((p_dep > p_tgt) | need) & ~at_depot
        dep_loc = locs_full[:, 0]
        cur_loc = locs_full[ar, cur]
        tgt_loc = locs_full[ar, tgt]
        direct = (tgt_loc - cur_loc).norm(dim=-1)
        via_cost = (dep_loc - cur_loc).norm(dim=-1) + (tgt_loc - dep_loc).norm(dim=-1)
        cost = cost + torch.where(via, via_cost, direct)
        load = torch.where(via, torch.ones_like(load), load) - dem_full[ar, tgt]
        visited[ar, tgt] = True
        cur = tgt
    cost = cost + (locs_full[:, 0] - locs_full[ar, cur]).norm(dim=-1)
    return cost

@torch.no_grad()
def rrc_generic(repair_fn, replay_fn, init_perm, init_costs, budget, sub_min=4, max_L=None, seed=0,
                log_every=10, log=print, tag="wb"):
    B, N = init_perm.shape
    perm = init_perm.clone()
    cur_costs = init_costs.clone()
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_accept_total = 0
    for it in range(budget):
        first_idx = int(torch.randint(0, N, (1,), generator=g).item())
        length = int(torch.randint(sub_min, N + 1, (1,), generator=g).item())
        if max_L is not None:
            length = min(length, max_L)
        length = min(length, N - first_idx)
        if length < sub_min:
            continue
        sub_perm = perm[:, first_idx:first_idx + length]
        new_sub = repair_fn(sub_perm)
        new_perm = perm.clone()
        new_perm[:, first_idx:first_idx + length] = new_sub[:, :length]
        new_costs = replay_fn(new_perm)
        accept = new_costs < cur_costs - 1e-9
        if accept.any():
            perm = torch.where(accept.unsqueeze(-1), new_perm, perm)
            cur_costs = torch.where(accept, new_costs, cur_costs)
            n_accept_total += int(accept.sum().item())
        if log is not None and ((it + 1) % log_every == 0 or it == budget - 1):
            log(f"    [RRC {tag}] it {it+1:>4}/{budget}  L={length:<4} acc={int(accept.sum().item()):>3}/{B}  "
                f"mean_cost={cur_costs.mean().item():.4f}  total_acc={n_accept_total}")
    return perm, cur_costs
