from __future__ import annotations
import torch
from whitebox_routing.cvrp import lehd_action_distribution

class EmptyBank:

    def set_N(self, nodes):
        self.nodes = nodes

    def compute_probs(self, batch, device):
        rows, candidates = batch["unselected_idx"].shape
        dtype = batch["unselected_locs"].dtype
        return (
            torch.zeros(rows, 1, self.nodes + 1, dtype=dtype, device=device),
            torch.zeros(rows, 1, candidates, dtype=dtype, device=device),
        )

class WhiteboxAdapter:

    def __call__(self, batch, full_placeholder, residual_placeholder):
        del residual_placeholder
        residual = lehd_action_distribution(
            batch["unselected_locs"], batch["unselected_demand"],
            batch["current_loc"], batch["depot_loc"], batch["load"],
            batch["feasibility_mask"],
        )
        full = torch.zeros_like(full_placeholder[:, 0])
        full.scatter_(1, batch["unselected_idx"], residual[:, :-1])
        full[:, 0] = residual[:, -1]
        return full, torch.ones(full.shape[0], 1, dtype=full.dtype, device=full.device)

def validate_physical_sequences(locs, demands, sequences):
    xy, demand = locs.cpu(), demands.cpu()
    expected = list(range(1, xy.shape[1]))
    costs, route_counts = [], []
    for row, sequence in enumerate(sequences):
        sequence = [int(node) for node in sequence]
        if sequence[0] != 0 or sequence[-1] != 0:
            raise AssertionError("Physical routes must start and end at the depot")
        if sorted(node for node in sequence if node != 0) != expected:
            raise AssertionError("Each customer must be served exactly once")
        used, route_count = 0.0, 0
        previous = 0
        for node in sequence:
            if node == 0:
                used = 0.0
            else:
                route_count += int(previous == 0)
                used += demand[row, node].item()
                if used > 1 + 1e-5:
                    raise AssertionError("Capacity violation in physical sequence")
            previous = node
        coords = xy[row, torch.tensor(sequence)]
        costs.append((coords[1:] - coords[:-1]).norm(dim=-1).sum())
        route_counts.append(route_count)
    return torch.stack(costs), route_counts

