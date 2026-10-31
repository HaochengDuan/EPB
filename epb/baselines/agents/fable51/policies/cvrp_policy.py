from __future__ import annotations
from dataclasses import dataclass, asdict
import torch

INF = float("inf")

@dataclass
class CVRPParams:
    a_onward: float = 0.0
    a_regret: float = 0.0
    a_dir: float = 0.0
    a_dense: float = 0.0
    k_dense: int = 3
    b_far: float = 0.0
    b_far_depot: float = 0.0
    b_ret: float = 0.0
    l_ret: float = 0.3
    b_dem: float = 0.0
    b_ang: float = 0.0
    b_arc: float = 0.0
    c_0: float = 0.0
    c_load: float = 0.0
    tau: float = 0.1

    def to_dict(self):
        return asdict(self)

def cvrp_scores(cust_locs, cust_dem, open_mask, feas_mask, cur_loc, depot_loc, load, at_depot,
                p: CVRPParams, D=None):
    B, K, _ = cust_locs.shape
    dev = cust_locs.device
    openf = open_mask.float()
    m = openf.sum(-1)
    d = (cust_locs - cur_loc.unsqueeze(1)).norm(dim=-1)
    d0 = (cust_locs - depot_loc.unsqueeze(1)).norm(dim=-1)
    dcd = (cur_loc - depot_loc).norm(dim=-1)
    if D is None:
        D = torch.cdist(cust_locs, cust_locs)
    eye = torch.eye(K, device=dev, dtype=torch.bool).unsqueeze(0)
    Dm = D.masked_fill(~open_mask.unsqueeze(1) | eye, INF)
    nn_c = Dm.min(dim=-1).values
    nn = torch.minimum(nn_c, d0)
    sigma = (torch.minimum(nn_c, d0) * openf).sum(-1) / m.clamp(min=1)
    sigma = sigma.clamp(min=1e-6)
    Dbar = (d0 * openf).sum(-1) / m.clamp(min=1)

    s = d.clone()
    if p.a_onward != 0.0:
        s = s + p.a_onward * nn
    if p.a_regret != 0.0:
        s = s - p.a_regret * torch.relu(nn - d)
    if p.a_dir != 0.0:
        cen = (cust_locs * openf.unsqueeze(-1)).sum(1) / m.clamp(min=1).unsqueeze(-1)
        v = cen - cur_loc
        u = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        proj = ((cust_locs - cur_loc.unsqueeze(1)) * u.unsqueeze(1)).sum(-1)
        s = s + p.a_dir * proj
    if p.a_dense != 0.0:
        k = min(p.k_dense, K)
        topk = Dm.topk(k, dim=-1, largest=False).values
        fin = torch.isfinite(topk)
        dens = topk.masked_fill(~fin, 0.0).sum(-1) / fin.sum(-1).clamp(min=1)
        dens = torch.where(fin.any(-1), dens, d0)
        s = s + p.a_dense * dens
    if p.b_far != 0.0 or p.b_far_depot != 0.0:
        bf = torch.where(at_depot, torch.full_like(load, p.b_far_depot), torch.full_like(load, p.b_far))
        s = s - bf.unsqueeze(-1) * d0
    if p.b_ret != 0.0:
        rem = (load.unsqueeze(-1) - cust_dem)
        phi = (1.0 - rem / p.l_ret).clamp(0.0, 1.0)
        s = s + p.b_ret * d0 * phi
    if p.b_dem != 0.0:
        s = s - p.b_dem * cust_dem * Dbar.unsqueeze(-1)
    if p.b_ang != 0.0 or p.b_arc != 0.0:
        vj = cust_locs - depot_loc.unsqueeze(1)
        vc = cur_loc - depot_loc
        th_j = torch.atan2(vj[..., 1], vj[..., 0])
        th_c = torch.atan2(vc[:, 1], vc[:, 0]).unsqueeze(-1)
        dth = torch.remainder(th_j - th_c + torch.pi, 2 * torch.pi) - torch.pi
        ang = dth.abs() * (~at_depot).float().unsqueeze(-1)
        if p.b_ang != 0.0:
            s = s + p.b_ang * Dbar.unsqueeze(-1) * ang
        if p.b_arc != 0.0:
            s = s + p.b_arc * torch.minimum(d0, dcd.unsqueeze(-1)) * ang
    s_depot = dcd + p.c_0 * sigma + p.c_load * load * Dbar
    return s, s_depot, sigma

def cvrp_probs(cust_locs, cust_dem, open_mask, feas_mask, cur_loc, depot_loc, load, at_depot,
               p: CVRPParams, D=None):
    s, s_depot, sigma = cvrp_scores(cust_locs, cust_dem, open_mask, feas_mask, cur_loc, depot_loc,
                                    load, at_depot, p, D)
    logits = torch.cat([s, s_depot.unsqueeze(-1)], dim=-1)
    logits = -logits / (p.tau * sigma.unsqueeze(-1))
    mask = torch.cat([feas_mask, (~at_depot).unsqueeze(-1)], dim=-1)
    none = ~mask.any(-1)
    if none.any():
        mask = mask.clone(); mask[none, -1] = True
    logits = logits.masked_fill(~mask, -INF)
    return torch.softmax(logits, dim=-1)

def cvrp_probs_pomo(locs, demand, visited, current_node, load, mask, p: CVRPParams, D=None):
    B = locs.size(0)
    cust_locs, cust_dem = locs[:, 1:], demand[:, 1:]
    open_mask = visited[:, 1:] == 0
    feas_mask = mask[:, 1:].bool() & open_mask
    at_depot = current_node == 0
    cur_loc = locs[torch.arange(B, device=locs.device), current_node]
    pr = cvrp_probs(cust_locs, cust_dem, open_mask, feas_mask, cur_loc, locs[:, 0], load, at_depot, p, D)
    return torch.cat([pr[:, -1:], pr[:, :-1]], dim=-1)

def cvrp_probs_lehd(unselected_locs, unselected_demand, current_loc, depot_loc, load, feasibility_mask,
                    p: CVRPParams):
    B, K, _ = unselected_locs.shape
    open_mask = torch.ones(B, K, dtype=torch.bool, device=unselected_locs.device)
    at_depot = (current_loc - depot_loc).norm(dim=-1) < 1e-9
    return cvrp_probs(unselected_locs, unselected_demand, open_mask, feasibility_mask.bool(),
                      current_loc, depot_loc, load, at_depot, p)

def probs_to_bank_logits(probs, mask, bank_temperature: float = 1.0):
    logp = torch.log(probs.clamp(min=1e-30))
    return (bank_temperature * logp).masked_fill(~mask.bool(), -INF)
