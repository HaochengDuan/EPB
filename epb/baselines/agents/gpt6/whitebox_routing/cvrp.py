from __future__ import annotations

import torch
from torch import Tensor

RADIAL_WEIGHT = 0.8
DEGREE_WEIGHT = 0.25
DEPOT_CAPACITY_WEIGHT = 0.3
PROBABILITY_TEMPERATURE = 0.25
CAPACITY_TOLERANCE = 1e-5

def _terms(locs, demand, remaining, allowed, current_loc, depot_loc, load,
           at_depot, depot_allowed):
    if locs.ndim != 3 or locs.shape[-1] != 2 or not locs.is_floating_point():
        raise ValueError("locs must be floating [B,K,2]")
    b, k, _ = locs.shape
    for name, value in (("demand", demand), ("remaining", remaining), ("allowed", allowed)):
        if value.shape != (b, k):
            raise ValueError(f"{name} must have shape [B,K]")
    if current_loc.shape != (b, 2) or depot_loc.shape != (b, 2):
        raise ValueError("current_loc and depot_loc must have shape [B,2]")
    if load.numel() != b:
        raise ValueError("load must contain B normalized remaining capacities")
    xy = locs if locs.dtype in (torch.float32, torch.float64) else locs.float()
    current_loc, depot_loc = current_loc.to(xy.dtype), depot_loc.to(xy.dtype)
    load = load.reshape(b).to(xy.dtype)
    demand = demand.to(xy.dtype)
    remaining = remaining.bool()
    feasible_customer = remaining & allowed.bool() & (demand <= load[:, None] + CAPACITY_TOLERANCE)
    feasible_depot = depot_allowed.bool() & ~at_depot
    distance = (xy-current_loc[:, None]).norm(dim=-1)
    radius = (xy-depot_loc[:, None]).norm(dim=-1)
    current_radius = (current_loc-depot_loc).norm(dim=-1)

    pairwise = torch.cdist(xy, xy, compute_mode="donot_use_mm_for_euclid_dist")
    pairwise.masked_fill_(~remaining[:, None, :], torch.inf)
    pairwise.diagonal(dim1=1, dim2=2).fill_(torch.inf)
    connections = torch.cat((pairwise, radius[..., None]), dim=-1)
    nearest = connections.topk(min(2, k+1), largest=False, dim=-1).values
    second = torch.where(torch.isfinite(nearest[..., -1]), nearest[..., -1], 0.)
    second = torch.where((remaining.sum(-1)>1)[:, None], second, 0.)
    radial_credit = RADIAL_WEIGHT*load[:, None]*radius
    degree_credit = DEGREE_WEIGHT*second
    customer_cost = distance-radial_credit-degree_credit
    customer_cost = torch.where(at_depot[:, None], -radius, customer_cost)
    depot_cost = current_radius*(1+DEPOT_CAPACITY_WEIGHT*load)
    decision_cost = torch.cat((customer_cost, depot_cost[:, None]), dim=-1)
    feasible = torch.cat((feasible_customer, feasible_depot[:, None]), dim=-1)
    count = remaining.sum(-1).clamp_min(1)
    spatial_scale = ((radius.square()*remaining).sum(-1)/count).sqrt()
    temperature = PROBABILITY_TEMPERATURE*spatial_scale/count.sqrt()
    temperature = torch.where(temperature>0, temperature, 1.)
    return {
        "feasible": feasible,
        "remaining_count": remaining.sum(-1),
        "distance": distance,
        "depot_distance": radius,
        "radial_credit": radial_credit,
        "degree_credit": degree_credit,
        "at_depot": at_depot,
        "customer_cost": customer_cost,
        "depot_cost": depot_cost,
        "decision_cost": decision_cost,
        "temperature": temperature,
        "logits": -decision_cost/temperature[:, None],
    }

def _probabilities(terms):
    feasible = terms["feasible"]
    logits = terms["logits"].masked_fill(~feasible, -torch.inf)
    logits = torch.where(feasible.any(-1, keepdim=True), logits, 0.)
    return logits.softmax(-1).masked_fill(~feasible, 0.)

def _pomo_terms(locs, demand, current_node, mask, visited, load, first_node):
    if locs.ndim != 3 or locs.shape[1] < 1 or locs.shape[-1] != 2:
        raise ValueError("locs must include depot 0 and have shape [B,N+1,2]")
    b, n, _ = locs.shape
    if demand.shape != (b,n) or mask.shape != (b,n) or visited.shape != (b,n):
        raise ValueError("demand, mask, visited must have shape [B,N+1]")
    if current_node.numel() != b or first_node.numel() != b:
        raise ValueError("current_node and first_node must contain B indices")
    rows = torch.arange(b, device=locs.device)
    current = current_node.reshape(b).long()
    return _terms(locs[:, 1:], demand[:, 1:], ~(visited[:, 1:]>0.5),
                  mask[:, 1:], locs[rows, current], locs[:, 0], load,
                  current==0, mask[:, 0])

def _lehd_terms(unselected_locs, unselected_demand, current_loc, depot_loc,
                load, feasibility_mask):
    remaining = torch.ones(unselected_locs.shape[:2], dtype=torch.bool,
                           device=unselected_locs.device)
    full_load = load.reshape(-1) >= 1-CAPACITY_TOLERANCE
    at_depot = (current_loc==depot_loc).all(-1) & full_load
    return _terms(unselected_locs, unselected_demand, remaining,
                  feasibility_mask, current_loc, depot_loc, load, at_depot,
                  torch.ones_like(at_depot))

def _depot_first(value):
    return torch.cat((value[:, -1:], value[:, :-1]), dim=-1)

@torch.no_grad()
def pomo_action_distribution(locs, demand, current_node, mask, visited, load, first_node):
    return _depot_first(_probabilities(_pomo_terms(
        locs, demand, current_node, mask, visited, load, first_node)))

@torch.no_grad()
def pomo_bank_logits(locs, demand, current_node, mask, visited, load, first_node):
    terms = _pomo_terms(locs, demand, current_node, mask, visited, load, first_node)
    return _depot_first(terms["logits"].masked_fill(~terms["feasible"], -1e9))

@torch.no_grad()
def pomo_explain(locs, demand, current_node, mask, visited, load, first_node):
    terms = _pomo_terms(locs, demand, current_node, mask, visited, load, first_node)
    terms["probabilities"] = _depot_first(_probabilities(terms))
    return terms

@torch.no_grad()
def lehd_action_distribution(unselected_locs, unselected_demand, current_loc,
                             depot_loc, load, feasibility_mask):
    return _probabilities(_lehd_terms(unselected_locs, unselected_demand,
                                     current_loc, depot_loc, load, feasibility_mask))

@torch.no_grad()
def lehd_bank_logits(unselected_locs, unselected_demand, current_loc,
                     depot_loc, load, feasibility_mask):
    terms = _lehd_terms(unselected_locs, unselected_demand, current_loc,
                        depot_loc, load, feasibility_mask)
    logits = terms["logits"].masked_fill(~terms["feasible"], -torch.inf)
    logits[:, -1] = torch.where(terms["feasible"][:, -1], logits[:, -1], -1e9)
    return logits

@torch.no_grad()
def lehd_explain(unselected_locs, unselected_demand, current_loc,
                 depot_loc, load, feasibility_mask):
    terms = _lehd_terms(unselected_locs, unselected_demand, current_loc,
                        depot_loc, load, feasibility_mask)
    terms["probabilities"] = _probabilities(terms)
    return terms
