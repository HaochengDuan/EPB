from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import torch

INF = float("inf")

@dataclass
class TSPParams:
    a_onward: float = 0.0
    a_regret: float = 0.0
    a_close: float = 0.0
    lam_close: float = 5.0
    a_dir: float = 0.0
    a_dense: float = 0.0
    k_dense: int = 3
    a_mom: float = 0.0
    a_skip: float = 0.0
    r_skip: float = 0.3
    a_ang: float = 0.0
    tau: float = 0.1

    def to_dict(self):
        return asdict(self)

def pairwise_dist(x: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, x)

def tsp_scores(cand_locs: torch.Tensor, cand_mask: torch.Tensor, cur_loc: torch.Tensor,
               end_loc: torch.Tensor, p: TSPParams, D: torch.Tensor | None = None,
               visited_centroid: torch.Tensor | None = None):
    B, K, _ = cand_locs.shape
    dev = cand_locs.device
    maskf = cand_mask.float()
    m = maskf.sum(-1)

    d = (cand_locs - cur_loc.unsqueeze(1)).norm(dim=-1)
    e = (cand_locs - end_loc.unsqueeze(1)).norm(dim=-1)

    if D is None:
        D = pairwise_dist(cand_locs)
    eye = torch.eye(K, device=dev, dtype=torch.bool).unsqueeze(0)
    Dm = D.masked_fill(~cand_mask.unsqueeze(1) | eye, INF)
    nn = Dm.min(dim=-1).values
    nn = torch.where(torch.isinf(nn), d, nn)

    sigma = (nn * maskf).sum(-1) / m.clamp(min=1)
    sigma = torch.where(m >= 2, sigma, (d * maskf).sum(-1) / m.clamp(min=1)).clamp(min=1e-6)

    s = d.clone()
    if p.a_onward != 0.0:
        s = s + p.a_onward * nn
    if p.a_regret != 0.0:
        s = s - p.a_regret * torch.relu(nn - d)
    if p.a_close != 0.0:
        w = torch.exp(-(m - 2).clamp(min=0) / p.lam_close).unsqueeze(-1)
        s = s - p.a_close * w * e
    if p.a_dir != 0.0:
        cen = (cand_locs * maskf.unsqueeze(-1)).sum(1) / m.clamp(min=1).unsqueeze(-1)
        v = cen - cur_loc
        u = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        proj = ((cand_locs - cur_loc.unsqueeze(1)) * u.unsqueeze(1)).sum(-1)
        s = s + p.a_dir * proj
    if p.a_dense != 0.0:
        k = min(p.k_dense, K)
        topk = Dm.topk(k, dim=-1, largest=False).values
        fin = torch.isfinite(topk)
        dens = (topk.masked_fill(~fin, 0.0).sum(-1) / fin.sum(-1).clamp(min=1))
        dens = torch.where(fin.any(-1), dens, d)
        s = s + p.a_dense * dens
    if p.a_skip != 0.0:
        via = (d.unsqueeze(-1) + Dm).min(dim=1).values
        via = torch.where(torch.isinf(via), d + d, via)
        detour = via - d
        s = s + p.a_skip * torch.relu(p.r_skip * d - detour)
    if p.a_ang != 0.0:
        cen = (cand_locs * maskf.unsqueeze(-1)).sum(1) / m.clamp(min=1).unsqueeze(-1)
        vj = cand_locs - cen.unsqueeze(1)
        vc = cur_loc - cen
        th_j = torch.atan2(vj[..., 1], vj[..., 0])
        th_c = torch.atan2(vc[:, 1], vc[:, 0]).unsqueeze(-1)
        dth = torch.remainder(th_j - th_c + torch.pi, 2 * torch.pi) - torch.pi
        s = s + p.a_ang * sigma.unsqueeze(-1) * dth.abs()
    if p.a_mom != 0.0 and visited_centroid is not None:
        v = cur_loc - visited_centroid
        u = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        mom = ((cand_locs - cur_loc.unsqueeze(1)) * u.unsqueeze(1)).sum(-1)
        s = s + p.a_mom * mom
    return s, sigma

def tsp_probs(cand_locs, cand_mask, cur_loc, end_loc, p: TSPParams, D=None, visited_centroid=None):
    s, sigma = tsp_scores(cand_locs, cand_mask, cur_loc, end_loc, p, D, visited_centroid)
    logits = -s / (p.tau * sigma.unsqueeze(-1))
    logits = logits.masked_fill(~cand_mask, -INF)
    return torch.softmax(logits, dim=-1)

def tsp_probs_pomo(locs, current_node, mask, visited, step_norm, first_node, p: TSPParams, D=None):
    B = locs.size(0)
    ar = torch.arange(B, device=locs.device)
    vis = visited.float()
    vc = (locs * vis.unsqueeze(-1)).sum(1) / vis.sum(-1).clamp(min=1).unsqueeze(-1)
    return tsp_probs(locs, mask.bool(), locs[ar, current_node], locs[ar, first_node], p, D, vc)

def tsp_probs_lehd(unselected_locs, first_loc, current_loc, p: TSPParams):
    B, K, _ = unselected_locs.shape
    mask = torch.ones(B, K, dtype=torch.bool, device=unselected_locs.device)
    return tsp_probs(unselected_locs, mask, current_loc, first_loc, p)

def probs_to_bank_logits(probs: torch.Tensor, mask: torch.Tensor, bank_temperature: float = 1.0):
    logp = torch.log(probs.clamp(min=1e-30))
    return (bank_temperature * logp).masked_fill(~mask.bool(), -INF)
