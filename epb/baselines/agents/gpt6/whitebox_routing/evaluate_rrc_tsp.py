from __future__ import annotations
import json
import time
import torch
from .evaluate import tsp_cost
from .tsp import lehd_action_distribution

def path_length(locs, path):
    xy = locs.gather(1, path[..., None].expand(-1, -1, 2))
    return (xy[:, 1:] - xy[:, :-1]).norm(dim=-1).sum(-1)

@torch.no_grad()
def reconstruct_path(locs, first_nodes, last_nodes):
    b, n, _ = locs.shape
    rows = torch.arange(b, device=locs.device)
    visited = torch.zeros(b, n, device=locs.device, dtype=torch.bool)
    visited[rows, first_nodes] = True
    current = first_nodes.clone()
    path = torch.empty(b, n, dtype=torch.long, device=locs.device)
    path[:, 0] = first_nodes
    for step in range(1, n):
        feasible = ~visited
        if step < n - 1:
            feasible[rows, last_nodes] = False
        ids = feasible.nonzero(as_tuple=False)[:, 1].reshape(b, -1)
        candidates = locs.gather(1, ids[..., None].expand(-1, -1, 2))
        probability = lehd_action_distribution(
            candidates, locs[rows, first_nodes], locs[rows, current],
        )
        if not torch.isfinite(probability).all() or not torch.allclose(
            probability.sum(-1), torch.ones(b, device=locs.device), atol=1e-5,
        ):
            raise AssertionError("Invalid reconstruction probabilities")
        selected = ids.gather(1, probability.argmax(-1)[:, None]).squeeze(1)
        path[:, step] = selected
        visited[rows, selected] = True
        current = selected
    if not torch.equal(path[:, -1], last_nodes) or not visited.all():
        raise AssertionError("Reconstruction did not preserve its fixed endpoints")
    return path

@torch.no_grad()
def run_rrc(locs, initial_tours, budget, seed, *, max_L=None,
            chunk_size=64, checkpoints=(), log_every=50):
    b, n, _ = locs.shape
    if n < 4 or budget < 0 or chunk_size < 1:
        raise ValueError("RRC requires N >= 4, budget >= 0, and positive chunk_size")
    tours = initial_tours.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    previous_cost = tsp_cost(locs, tours)
    snapshots = {}
    accepted_count = 0
    schedule = []
    started = time.perf_counter()
    for iteration in range(budget):
        first_idx = int(torch.randint(0, n, (1,), generator=generator).item())
        length = int(torch.randint(4, n + 1, (1,), generator=generator).item())
        flip = bool(torch.rand(1, generator=generator).item() < .5)
        if max_L is not None:
            length = min(length, max_L)
        schedule.append([first_idx, length, flip])
        if length < 4:
            continue
        oriented = tours.flip(1) if flip else tours
        doubled = torch.cat((oriented, oriented), dim=1)
        segment_ids = doubled[:, first_idx:first_idx + length]
        sub_locs = locs.gather(1, segment_ids[..., None].expand(-1, -1, 2))
        original_path_cost = path_length(locs, segment_ids)
        rebuilt = []
        for offset in range(0, b, chunk_size):
            sub = sub_locs[offset:offset + chunk_size]
            first = torch.zeros(len(sub), dtype=torch.long, device=locs.device)
            last = torch.full_like(first, length - 1)
            rebuilt.append(reconstruct_path(sub, first, last))
        local_path = torch.cat(rebuilt, dim=0)
        global_path = segment_ids.gather(1, local_path)
        rebuilt_path_cost = path_length(locs, global_path)
        accepted = rebuilt_path_cost < original_path_cost
        if accepted.any():
            joined = torch.cat((doubled[:, :first_idx], global_path,
                                doubled[:, first_idx + length:]), dim=1)
            spliced = joined[:, first_idx:first_idx + n]
            tours = torch.where(accepted[:, None], spliced, oriented)
            accepted_count += int(accepted.sum().item())
        cost = tsp_cost(locs, tours)
        if (cost > previous_cost + 1e-5).any():
            raise AssertionError("Strict open-path acceptance increased a full tour")
        previous_cost = cost
        completed = iteration + 1
        if completed in checkpoints:
            snapshots[completed] = {"costs": cost.cpu().clone(),
                                    "tours": tours.cpu().clone(),
                                    "accepted_reconstructions": accepted_count,
                                    "elapsed_seconds": time.perf_counter() - started}
        if log_every and (completed % log_every == 0 or completed == budget):
            print(json.dumps({"iteration": completed, "budget": budget,
                              "mean_cost": cost.mean().item(),
                              "accepted_reconstructions": accepted_count}), flush=True)
    return tours, snapshots, schedule

