from __future__ import annotations

import torch
from torch import Tensor

DEGREE_WEIGHT = 0.5
PERIPHERY_WEIGHT = 0.5
PROBABILITY_TEMPERATURE = 0.25

def _score_terms(
    locs: Tensor,
    feasible: Tensor,
    first_loc: Tensor,
    current_loc: Tensor,
    degree_weight: float = DEGREE_WEIGHT,
    periphery_weight: float = PERIPHERY_WEIGHT,
) -> dict[str, Tensor]:
    if locs.ndim != 3 or locs.shape[-1] != 2:
        raise ValueError("locs must have shape [B, N, 2]")
    if not locs.is_floating_point():
        raise TypeError("locs must be floating point")
    batch, nodes, _ = locs.shape
    if feasible.shape != (batch, nodes):
        raise ValueError("feasible must have shape [B, N]")
    if first_loc.shape != (batch, 2) or current_loc.shape != (batch, 2):
        raise ValueError("first_loc and current_loc must have shape [B, 2]")
    xy = locs if locs.dtype in (torch.float32, torch.float64) else locs.float()
    first = first_loc.to(xy.dtype)
    current = current_loc.to(xy.dtype)
    feasible = feasible.bool()
    remaining = feasible.sum(-1)
    distance = torch.linalg.vector_norm(xy - current[:, None], dim=-1)
    return_distance = torch.linalg.vector_norm(xy - first[:, None], dim=-1)

    pairwise = torch.cdist(xy, xy, compute_mode="donot_use_mm_for_euclid_dist")
    pairwise.masked_fill_(~feasible[:, None, :], torch.inf)
    pairwise.diagonal(dim1=1, dim2=2).fill_(torch.inf)
    connections = torch.cat((pairwise, return_distance[..., None]), dim=-1)
    nearest = connections.topk(min(2, nodes + 1), dim=-1, largest=False).values
    second = torch.where(torch.isfinite(nearest[..., -1]), nearest[..., -1], 0.0)
    degree_credit = torch.where(
        (remaining > 2)[:, None], degree_weight * second, 0.0
    )

    centroid = (xy * feasible[..., None]).sum(1) / remaining.clamp_min(1)[:, None]
    radius = torch.linalg.vector_norm(xy - centroid[:, None], dim=-1)
    periphery_credit = torch.where(
        (remaining > 2)[:, None], periphery_weight * radius, 0.0
    )

    closure_credit = return_distance / (remaining - 1).clamp_min(1)[:, None]
    closure_credit = torch.where(
        (remaining == 1)[:, None], -return_distance, closure_credit
    )
    closure_credit = torch.where((remaining == 0)[:, None], 0.0, closure_credit)
    distance = torch.where((remaining == 0)[:, None], 0.0, distance)
    decision_cost = distance - degree_credit - periphery_credit - closure_credit

    support_count = remaining + 2
    support_center = (
        (xy * feasible[..., None]).sum(1) + first + current
    ) / support_count[:, None]
    squared_spread = (
        ((xy - support_center[:, None]).square().sum(-1) * feasible).sum(-1)
        + (first - support_center).square().sum(-1)
        + (current - support_center).square().sum(-1)
    ) / support_count
    temperature = (
        PROBABILITY_TEMPERATURE * squared_spread.sqrt() / support_count.sqrt()
    )
    temperature = torch.where(temperature > 0, temperature, 1.0)
    logits = -decision_cost / temperature[:, None]
    return {
        "feasible": feasible,
        "remaining_count": remaining,
        "distance": distance,
        "degree_credit": degree_credit,
        "periphery_credit": periphery_credit,
        "closure_credit": closure_credit,
        "decision_cost": decision_cost,
        "temperature": temperature,
        "logits": logits,
    }

def _probabilities(terms: dict[str, Tensor]) -> Tensor:
    feasible = terms["feasible"]
    logits = terms["logits"].masked_fill(~feasible, -torch.inf)
    logits = torch.where(feasible.any(-1, keepdim=True), logits, 0.0)
    return logits.softmax(-1).masked_fill(~feasible, 0.0)

def _pomo_terms(locs, current_node, mask, visited, step_norm, first_node):
    batch = locs.shape[0]
    if visited.shape != locs.shape[:2]:
        raise ValueError("visited must have shape [B, N]")
    if current_node.numel() != batch or first_node.numel() != batch:
        raise ValueError("current_node and first_node must contain B indices")
    if step_norm.numel() != batch:
        raise ValueError("step_norm must contain B values")
    rows = torch.arange(batch, device=locs.device)
    return _score_terms(
        locs,
        mask.bool() & ~(visited > 0.5),
        locs[rows, first_node.reshape(batch).long()],
        locs[rows, current_node.reshape(batch).long()],
    )

def _lehd_terms(unselected_locs, first_loc, current_loc):
    feasible = torch.ones(
        unselected_locs.shape[:2], dtype=torch.bool, device=unselected_locs.device
    )
    return _score_terms(unselected_locs, feasible, first_loc, current_loc)

@torch.no_grad()
def pomo_action_distribution(locs, current_node, mask, visited, step_norm, first_node):
    return _probabilities(
        _pomo_terms(locs, current_node, mask, visited, step_norm, first_node)
    )

@torch.no_grad()
def pomo_bank_logits(locs, current_node, mask, visited, step_norm, first_node):
    terms = _pomo_terms(locs, current_node, mask, visited, step_norm, first_node)
    return terms["logits"].masked_fill(~terms["feasible"], -1e9)

@torch.no_grad()
def pomo_explain(locs, current_node, mask, visited, step_norm, first_node):
    terms = _pomo_terms(locs, current_node, mask, visited, step_norm, first_node)
    terms["probabilities"] = _probabilities(terms)
    return terms

@torch.no_grad()
def lehd_action_distribution(unselected_locs, first_loc, current_loc):
    return _probabilities(_lehd_terms(unselected_locs, first_loc, current_loc))

@torch.no_grad()
def lehd_bank_logits(unselected_locs, first_loc, current_loc):
    return _lehd_terms(unselected_locs, first_loc, current_loc)["logits"]

@torch.no_grad()
def lehd_explain(unselected_locs, first_loc, current_loc):
    terms = _lehd_terms(unselected_locs, first_loc, current_loc)
    terms["probabilities"] = _probabilities(terms)
    return terms
