from __future__ import annotations

import numpy as np
import torch

_ANG = torch.arange(8, dtype=torch.float32) * (2 * np.pi / 8)
DIRS8 = torch.stack([torch.cos(_ANG), torch.sin(_ANG)], dim=-1)

EPS = 1e-9
BIG = 1e6

TSP_FEATURES = [
    "d_cur",
    "d_first",
    "d_centroid_unvis",
    "periph_norm",
    "outward_cos",
    "dir_cons_cos",
    "cand_nn_dist",
    "cand_nn_gap",
    "d_cur_minus_dmin",
    "d_cur_rank_norm",
    "state_gap12",
    "hull_static",
    "extreme_frac",
    "n_rem_norm",
    "n_remaining",
    "step_norm",
]

_CVRP_EXTRA = [
    "demand",
    "load",
    "demand_over_load",
    "d_depot",
    "cw_savings",
    "is_depot",
    "load_slack",
    "d_cur_plus_depot",
]

CVRP_FEATURES = TSP_FEATURES + _CVRP_EXTRA
LEHD_CVRP_FEATURES = CVRP_FEATURES + [
    "via_depot",
    "move_dist",
]

STAGE_FEATURES = {"n_rem_norm", "n_remaining", "step_norm", "load"}

def static_hull(locs: torch.Tensor) -> torch.Tensor:
    from scipy.spatial import ConvexHull, QhullError

    xy = locs.detach().cpu().numpy()
    B, C, _ = xy.shape
    out = np.zeros((B, C), dtype=np.float32)
    for b in range(B):
        try:
            out[b, ConvexHull(xy[b]).vertices] = 1.0
        except (QhullError, ValueError):
            pass
    return torch.from_numpy(out).to(locs.device)

def pairwise_dist(locs: torch.Tensor) -> torch.Tensor:
    return torch.cdist(locs, locs)

def tsp_features(
    locs: torch.Tensor,
    D: torch.Tensor,
    hull: torch.Tensor,
    cur: torch.Tensor,
    first: torch.Tensor,
    prev: torch.Tensor,
    unvis: torch.Tensor,
    step: int,
) -> torch.Tensor:
    B, N, _ = locs.shape
    dev = locs.device
    fl = unvis.float()
    n_rem = fl.sum(1, keepdim=True).clamp(min=1.0)

    bar = torch.arange(B, device=dev)
    x_cur = locs[bar, cur]
    x_prev = locs[bar, prev]
    d_cur = D[bar, cur]
    d_first = D[bar, first]

    cen = (locs * fl.unsqueeze(-1)).sum(1) / n_rem
    d_cen = (locs - cen.unsqueeze(1)).norm(dim=-1)

    periph = (D * fl.unsqueeze(1)).sum(-1) / n_rem
    periph_mean = (periph * fl).sum(1, keepdim=True) / n_rem
    periph_norm = periph / periph_mean.clamp(min=EPS)

    v_cand = locs - x_cur.unsqueeze(1)
    v_cand_n = v_cand / v_cand.norm(dim=-1, keepdim=True).clamp(min=EPS)
    v_out = (x_cur - cen)
    v_out_n = v_out / v_out.norm(dim=-1, keepdim=True).clamp(min=EPS)
    outward_cos = (v_cand_n * v_out_n.unsqueeze(1)).sum(-1)
    v_head = (x_cur - x_prev)
    head_norm = v_head.norm(dim=-1, keepdim=True)
    v_head_n = v_head / head_norm.clamp(min=EPS)
    dir_cons_cos = (v_cand_n * v_head_n.unsqueeze(1)).sum(-1)
    dir_cons_cos = torch.where(head_norm < EPS, torch.zeros_like(dir_cons_cos),
                               dir_cons_cos)

    D_un = D.masked_fill(~unvis.unsqueeze(1), BIG)
    D_un = D_un + torch.eye(N, device=dev).unsqueeze(0) * BIG
    two = torch.topk(D_un, k=min(2, N), dim=-1, largest=False).values
    cand_nn_dist = two[..., 0]
    cand_nn_gap = (two[..., 1] - two[..., 0]) if two.size(-1) > 1 \
        else torch.zeros_like(cand_nn_dist)
    cand_nn_dist = cand_nn_dist.clamp(max=BIG / 2)
    cand_nn_gap = cand_nn_gap.clamp(-BIG / 2, BIG / 2)

    d_cur_f = d_cur.masked_fill(~unvis, BIG)
    st_two = torch.topk(d_cur_f, k=min(2, N), dim=-1, largest=False).values
    d_min = st_two[:, :1]
    state_gap12 = (st_two[:, 1:2] - st_two[:, :1]) if st_two.size(1) > 1 \
        else torch.zeros_like(d_min)
    d_cur_minus_dmin = (d_cur - d_min).clamp(-BIG / 2, BIG / 2)
    rank = torch.argsort(torch.argsort(d_cur_f, dim=-1), dim=-1).float()
    d_cur_rank_norm = rank / n_rem

    dirs = DIRS8.to(dev)
    proj = locs @ dirs.t()
    proj_m = proj.masked_fill(~unvis.unsqueeze(-1), -BIG)
    proj_max = proj_m.max(dim=1, keepdim=True).values
    extreme_frac = (proj >= proj_max - 1e-6).float().mean(-1)

    n_rem_b = n_rem.expand(B, N)
    feats = torch.stack([
        d_cur, d_first, d_cen, periph_norm,
        outward_cos, dir_cons_cos,
        cand_nn_dist, cand_nn_gap,
        d_cur_minus_dmin, d_cur_rank_norm, state_gap12.expand(B, N),
        hull, extreme_frac,
        n_rem_b / N, n_rem_b,
        torch.full((B, N), step / max(N - 1, 1), device=dev),
    ], dim=-1)
    return feats

def cvrp_features(
    locs: torch.Tensor,
    demand: torch.Tensor,
    D: torch.Tensor,
    hull: torch.Tensor,
    cur: torch.Tensor,
    first: torch.Tensor,
    prev: torch.Tensor,
    unvis_cust: torch.Tensor,
    feas: torch.Tensor,
    load: torch.Tensor,
    step: int,
    step_max: int,
) -> torch.Tensor:
    B, C, _ = locs.shape
    dev = locs.device
    fl = unvis_cust.float()
    n_rem = fl.sum(1, keepdim=True).clamp(min=1.0)

    bar = torch.arange(B, device=dev)
    x_cur = locs[bar, cur]
    x_prev = locs[bar, prev]
    d_cur = D[bar, cur]
    d_first = D[bar, first]
    d_depot = D[:, 0, :]

    cen = (locs * fl.unsqueeze(-1)).sum(1) / n_rem
    d_cen = (locs - cen.unsqueeze(1)).norm(dim=-1)

    periph = (D * fl.unsqueeze(1)).sum(-1) / n_rem
    periph_mean = (periph * fl).sum(1, keepdim=True) / n_rem
    periph_norm = periph / periph_mean.clamp(min=EPS)

    v_cand = locs - x_cur.unsqueeze(1)
    v_cand_n = v_cand / v_cand.norm(dim=-1, keepdim=True).clamp(min=EPS)
    v_out = x_cur - cen
    v_out_n = v_out / v_out.norm(dim=-1, keepdim=True).clamp(min=EPS)
    outward_cos = (v_cand_n * v_out_n.unsqueeze(1)).sum(-1)
    v_head = x_cur - x_prev
    head_norm = v_head.norm(dim=-1, keepdim=True)
    v_head_n = v_head / head_norm.clamp(min=EPS)
    dir_cons_cos = (v_cand_n * v_head_n.unsqueeze(1)).sum(-1)
    dir_cons_cos = torch.where(head_norm < EPS, torch.zeros_like(dir_cons_cos),
                               dir_cons_cos)

    D_un = D.masked_fill(~unvis_cust.unsqueeze(1), BIG)
    D_un = D_un + torch.eye(C, device=dev).unsqueeze(0) * BIG
    two = torch.topk(D_un, k=min(2, C), dim=-1, largest=False).values
    cand_nn_dist = two[..., 0].clamp(max=BIG / 2)
    cand_nn_gap = (two[..., 1] - two[..., 0]).clamp(-BIG / 2, BIG / 2)

    d_cur_f = d_cur.masked_fill(~feas, BIG)
    st_two = torch.topk(d_cur_f, k=min(2, C), dim=-1, largest=False).values
    d_min = st_two[:, :1]
    state_gap12 = (st_two[:, 1:2] - st_two[:, :1])
    d_cur_minus_dmin = (d_cur - d_min).clamp(-BIG / 2, BIG / 2)
    n_feas = feas.float().sum(1, keepdim=True).clamp(min=1.0)
    rank = torch.argsort(torch.argsort(d_cur_f, dim=-1), dim=-1).float()
    d_cur_rank_norm = rank / n_feas

    dirs = DIRS8.to(dev)
    proj = locs @ dirs.t()
    proj_m = proj.masked_fill(~unvis_cust.unsqueeze(-1), -BIG)
    proj_max = proj_m.max(dim=1, keepdim=True).values
    extreme_frac = (proj >= proj_max - 1e-6).float().mean(-1)

    load_b = load.unsqueeze(1).expand(B, C)
    is_depot = torch.zeros(B, C, device=dev)
    is_depot[:, 0] = 1.0
    d_cur_depot = D[bar, cur, 0].unsqueeze(1)
    cw = d_cur_depot + d_depot - d_cur

    n_rem_b = n_rem.expand(B, C)
    feats = torch.stack([
        d_cur, d_first, d_cen, periph_norm,
        outward_cos, dir_cons_cos,
        cand_nn_dist, cand_nn_gap,
        d_cur_minus_dmin, d_cur_rank_norm, state_gap12.expand(B, C),
        hull, extreme_frac,
        n_rem_b / max(C - 1, 1), n_rem_b,
        torch.full((B, C), step / max(step_max, 1), device=dev),
        demand, load_b, demand / load_b.clamp(min=EPS),
        d_depot, cw, is_depot,
        load_b - demand, d_cur + d_depot,
    ], dim=-1)
    return feats

def lehd_cvrp_features(
    locs: torch.Tensor,
    demand: torch.Tensor,
    D: torch.Tensor,
    hull: torch.Tensor,
    cur: torch.Tensor,
    first: torch.Tensor,
    prev: torch.Tensor,
    unvis_cust: torch.Tensor,
    feas_cust: torch.Tensor,
    load: torch.Tensor,
    step: int,
    step_max: int,
) -> torch.Tensor:
    B, C1, _ = locs.shape
    N = C1 - 1
    dev = locs.device

    base = cvrp_features(locs, demand, D, hull, cur, first, prev,
                         unvis_cust, feas_cust, load, step, step_max)
    base_c = base[:, 1:, :]
    F0 = base_c.size(-1)

    bar = torch.arange(B, device=dev)
    d_cur_cust = D[bar, cur][:, 1:]
    d_cur_depot = D[bar, cur, 0].unsqueeze(1)
    d_depot_cust = D[:, 0, 1:]

    direct = torch.cat([base_c,
                        torch.zeros(B, N, 1, device=dev),
                        d_cur_cust.unsqueeze(-1)], dim=-1)
    via_move = (d_cur_depot + d_depot_cust).unsqueeze(-1)
    via = torch.cat([base_c,
                     torch.ones(B, N, 1, device=dev),
                     via_move], dim=-1)
    i_load = CVRP_FEATURES.index("load")
    i_dol = CVRP_FEATURES.index("demand_over_load")
    i_slack = CVRP_FEATURES.index("load_slack")
    cap = torch.ones(B, N, device=dev)
    dem_c = demand[:, 1:]
    via = via.clone()
    via[..., i_load] = cap
    via[..., i_dol] = dem_c / cap.clamp(min=EPS)
    via[..., i_slack] = cap - dem_c
    assert F0 + 2 == len(LEHD_CVRP_FEATURES)
    return torch.cat([direct, via], dim=1)

_ANG32 = torch.arange(32, dtype=torch.float32) * (2 * np.pi / 32)
DIRS32 = torch.stack([torch.cos(_ANG32), torch.sin(_ANG32)], dim=-1)

LEHD_TSP_FEATURES = [
    "d_cur", "d_first", "d_centroid_unvis", "periph_norm", "outward_cos",
    "cand_nn_dist", "cand_nn_gap", "d_cur_minus_dmin", "d_cur_rank_norm",
    "state_gap12", "hull_proxy32", "n_remaining",
]

LEHD_CVRP_R_FEATURES = [
    "d_cur", "d_depot", "d_centroid_unvis", "periph_norm", "outward_cos",
    "cand_nn_dist", "cand_nn_gap", "d_cur_minus_dmin", "d_cur_rank_norm",
    "state_gap12", "hull_proxy32", "n_remaining",
    "demand", "load", "demand_over_load", "cw_savings", "load_slack",
    "via_depot", "move_dist",
]

def hull_proxy(locs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dirs = DIRS32.to(locs.device)
    proj = locs @ dirs.t()
    proj_m = proj.masked_fill(~mask.unsqueeze(-1), -BIG)
    mx = proj_m.max(dim=1, keepdim=True).values
    return (proj >= mx - 1e-6).float().mean(-1)

def _remaining_geometry(locs, D, ref, unvis, feas):
    B, C, _ = locs.shape
    dev = locs.device
    fl = unvis.float()
    n_rem = fl.sum(1, keepdim=True).clamp(min=1.0)
    bar = torch.arange(B, device=dev)

    d_cur = D[bar, ref]
    cen = (locs * fl.unsqueeze(-1)).sum(1) / n_rem
    d_cen = (locs - cen.unsqueeze(1)).norm(dim=-1)

    periph = (D * fl.unsqueeze(1)).sum(-1) / n_rem
    periph_mean = (periph * fl).sum(1, keepdim=True) / n_rem
    periph_norm = periph / periph_mean.clamp(min=EPS)

    x_cur = locs[bar, ref]
    v_cand = locs - x_cur.unsqueeze(1)
    v_cand_n = v_cand / v_cand.norm(dim=-1, keepdim=True).clamp(min=EPS)
    v_out = x_cur - cen
    v_out_n = v_out / v_out.norm(dim=-1, keepdim=True).clamp(min=EPS)
    outward_cos = (v_cand_n * v_out_n.unsqueeze(1)).sum(-1)

    D_un = D.masked_fill(~unvis.unsqueeze(1), BIG)
    D_un = D_un + torch.eye(C, device=dev).unsqueeze(0) * BIG
    two = torch.topk(D_un, k=min(2, C), dim=-1, largest=False).values
    cand_nn_dist = two[..., 0].clamp(max=BIG / 2)
    cand_nn_gap = (two[..., 1] - two[..., 0]).clamp(-BIG / 2, BIG / 2)

    d_cur_f = d_cur.masked_fill(~feas, BIG)
    st_two = torch.topk(d_cur_f, k=min(2, C), dim=-1, largest=False).values
    d_min = st_two[:, :1]
    state_gap12 = (st_two[:, 1:2] - st_two[:, :1])
    d_cur_minus_dmin = (d_cur - d_min).clamp(-BIG / 2, BIG / 2)
    n_feas = feas.float().sum(1, keepdim=True).clamp(min=1.0)
    rank = torch.argsort(torch.argsort(d_cur_f, dim=-1), dim=-1).float()
    d_cur_rank_norm = rank / n_feas

    return (d_cur, d_cen, periph_norm, outward_cos, cand_nn_dist, cand_nn_gap,
            d_cur_minus_dmin, d_cur_rank_norm, state_gap12.expand(B, C),
            hull_proxy(locs, unvis), n_rem.expand(B, C))

def lehd_tsp_features(locs, D, cur, first, unvis, **_ignored):
    B, N, _ = locs.shape
    bar = torch.arange(B, device=locs.device)
    (d_cur, d_cen, periph_norm, outward_cos, nn_d, nn_g,
     d_minus, d_rank, gap12, hullp, n_rem) = _remaining_geometry(
        locs, D, cur, unvis, unvis)
    d_first = D[bar, first]
    feats = torch.stack([
        d_cur, d_first, d_cen, periph_norm, outward_cos,
        nn_d, nn_g, d_minus, d_rank, gap12, hullp, n_rem,
    ], dim=-1)
    assert feats.size(-1) == len(LEHD_TSP_FEATURES)
    return feats

def lehd_cvrp_features_r(locs, demand, D, cur, unvis_cust, load, **_ignored):
    B, C1, _ = locs.shape
    N = C1 - 1
    dev = locs.device
    bar = torch.arange(B, device=dev)

    (d_cur, d_cen, periph_norm, outward_cos, nn_d, nn_g,
     d_minus, d_rank, gap12, hullp, n_rem) = _remaining_geometry(
        locs, D, cur, unvis_cust, unvis_cust)
    d_depot = D[:, 0, :]
    d_cur_depot = D[bar, cur, 0].unsqueeze(1)
    cw = d_cur_depot + d_depot - d_cur
    load_b = load.unsqueeze(1).expand(B, C1)

    base = torch.stack([
        d_cur, d_depot, d_cen, periph_norm, outward_cos,
        nn_d, nn_g, d_minus, d_rank, gap12, hullp, n_rem,
        demand, load_b, demand / load_b.clamp(min=EPS), cw, load_b - demand,
    ], dim=-1)[:, 1:, :]
    F0 = base.size(-1)

    d_cur_cust = d_cur[:, 1:]
    d_depot_cust = d_depot[:, 1:]
    dem_c = demand[:, 1:]
    cap = torch.ones(B, N, device=dev)
    i_load = LEHD_CVRP_R_FEATURES.index("load")
    i_dol = LEHD_CVRP_R_FEATURES.index("demand_over_load")
    i_slack = LEHD_CVRP_R_FEATURES.index("load_slack")

    direct = torch.cat([base, torch.zeros(B, N, 1, device=dev),
                        d_cur_cust.unsqueeze(-1)], dim=-1)
    via = torch.cat([base, torch.ones(B, N, 1, device=dev),
                     (d_cur_depot + d_depot_cust).unsqueeze(-1)], dim=-1).clone()
    via[..., i_load] = cap
    via[..., i_dol] = dem_c / cap.clamp(min=EPS)
    via[..., i_slack] = cap - dem_c
    assert F0 + 2 == len(LEHD_CVRP_R_FEATURES)
    return torch.cat([direct, via], dim=1)
