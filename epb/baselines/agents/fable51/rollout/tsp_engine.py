from __future__ import annotations
import sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policies.tsp_policy import TSPParams, tsp_probs, pairwise_dist

INF = float("inf")

def tour_length(locs, tour):
    g = torch.gather(locs, 1, tour.unsqueeze(-1).expand(-1, -1, 2))
    s = torch.roll(g, shifts=-1, dims=1)
    return torch.norm(g - s, dim=-1).sum(-1)

def path_length(locs, path):
    g = torch.gather(locs, 1, path.unsqueeze(-1).expand(-1, -1, 2))
    return torch.norm(g[:, 1:] - g[:, :-1], dim=-1).sum(-1)

@torch.no_grad()
def rollout_tsp(locs, starts, p: TSPParams, last_nodes=None, D=None, return_probs=False):
    R, N, _ = locs.shape
    dev = locs.device
    ar = torch.arange(R, device=dev)
    visited = torch.zeros(R, N, dtype=torch.bool, device=dev)
    visited[ar, starts] = True
    cur = starts.clone()
    end_loc = locs[ar, starts] if last_nodes is None else locs[ar, last_nodes]
    if D is None:
        D = pairwise_dist(locs)
    tour = [starts.clone()]
    probs_log = [] if return_probs else None
    vis_sum = locs[ar, starts].clone()
    for step in range(1, N):
        mask = ~visited
        if last_nodes is not None and step < N - 1:
            mask = mask.clone()
            mask[ar, last_nodes] = False
        vc = vis_sum / step
        probs = tsp_probs(locs, mask, locs[ar, cur], end_loc, p, D, vc)
        if return_probs:
            probs_log.append((mask.clone(), probs))
        nxt = probs.argmax(-1)
        visited[ar, nxt] = True
        cur = nxt
        vis_sum = vis_sum + locs[ar, cur]
        tour.append(cur.clone())
    tour = torch.stack(tour, 1)
    return (tour, probs_log) if return_probs else tour

@torch.no_grad()
def tsp_greedy(locs, p: TSPParams, start: int = 0, chunk: int = 2048):
    B, N, _ = locs.shape
    tours = []
    for s in range(0, B, chunk):
        l = locs[s:s + chunk]
        st = torch.full((l.size(0),), start, dtype=torch.long, device=locs.device)
        tours.append(rollout_tsp(l, st, p))
    tours = torch.cat(tours, 0)
    return tours, tour_length(locs, tours)

@torch.no_grad()
def tsp_multistart(locs, p: TSPParams, chunk_rollouts: int = 2048):
    B, N, _ = locs.shape
    dev = locs.device
    D_all = pairwise_dist(locs)
    per_inst = max(1, chunk_rollouts // N)
    best_cost = torch.full((B,), INF, device=dev)
    best_tour = torch.zeros(B, N, dtype=torch.long, device=dev)
    for s in range(0, B, per_inst):
        e = min(s + per_inst, B)
        b = e - s
        locs_rep = locs[s:e].unsqueeze(1).expand(b, N, N, 2).reshape(b * N, N, 2)
        D_rep = D_all[s:e].unsqueeze(1).expand(b, N, N, N).reshape(b * N, N, N)
        starts = torch.arange(N, device=dev).repeat(b)
        tours = rollout_tsp(locs_rep, starts, p, D=D_rep)
        costs = tour_length(locs_rep, tours).view(b, N)
        mn, idx = costs.min(dim=1)
        best_cost[s:e] = mn
        best_tour[s:e] = tours.view(b, N, N)[torch.arange(b, device=dev), idx]
    return best_tour, best_cost

def sample_rrc_segment(N, generator, sub_min=4):
    first_idx = int(torch.randint(0, N, (1,), generator=generator).item())
    length = int(torch.randint(sub_min, N + 1, (1,), generator=generator).item())
    do_flip = bool((torch.rand(1, generator=generator).item()) < 0.5)
    return first_idx, length, do_flip

def extract_subgraph(locs, tours, first_idx, length):
    B, N = tours.shape
    double_tour = torch.cat([tours, tours], dim=1)
    tour_idx_seg = double_tour[:, first_idx:first_idx + length]
    pre_seg = double_tour[:, :first_idx]
    post_seg = double_tour[:, first_idx + length:]
    sub_locs = locs.gather(1, tour_idx_seg.unsqueeze(-1).expand(B, length, 2))
    return sub_locs, tour_idx_seg, pre_seg, post_seg

def splice_back(pre_seg, new_global_seg, post_seg, N):
    full = torch.cat([pre_seg, new_global_seg, post_seg], dim=1)
    first_idx = pre_seg.size(1)
    return full[:, first_idx:first_idx + N]

@torch.no_grad()
def tsp_rrc(locs, init_tours, p: TSPParams, budget, seed, sub_min=4, max_L=None, log_every=10,
            log=print):
    dev = locs.device
    B, N, _ = locs.shape
    tours = init_tours.clone()
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_accepted = 0
    for it in range(budget):
        first_idx, length, do_flip = sample_rrc_segment(N, g, sub_min=sub_min)
        if max_L is not None:
            length = min(length, max_L)
        if length < sub_min:
            continue
        tours_iter = torch.flip(tours, dims=[1]) if do_flip else tours
        sub_locs, tour_idx_seg, pre, post = extract_subgraph(locs, tours_iter, first_idx, length)
        L = sub_locs.size(1)
        orig_len = path_length(locs, tour_idx_seg)
        first_local = torch.zeros(B, dtype=torch.long, device=dev)
        last_local = torch.full((B,), L - 1, dtype=torch.long, device=dev)
        new_path_local = rollout_tsp(sub_locs, first_local, p, last_nodes=last_local)
        new_path_global = tour_idx_seg.gather(1, new_path_local)
        new_len = path_length(locs, new_path_global)
        accept = new_len < orig_len
        if accept.any():
            spliced = splice_back(pre, new_path_global, post, N)
            tours_iter = torch.where(accept.unsqueeze(1), spliced, tours_iter)
            tours = tours_iter
            n_accepted += int(accept.sum().item())
        if log is not None and ((it + 1) % log_every == 0 or it == budget - 1):
            log(f"    [whitebox RRC] iter {it+1:>3}/{budget}  L={length:<4} flip={int(do_flip)}  "
                f"acc={int(accept.sum().item()):>4}/{B}  mean_cost={tour_length(locs, tours).mean().item():.4f}  "
                f"total_acc={n_accepted}")
    return tours
