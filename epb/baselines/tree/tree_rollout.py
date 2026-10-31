from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import protocols as P
import features as FT
from gpu_tree import masked_argmax

BIG = 1e6

@torch.no_grad()
def tsp_rollout(
    locs, first, *, teacher=None, tree=None, last_node=None,
    collect=None, need_features=None, D=None, hull=None, probe=None,
    feature_mode="pomo",
):
    B, N, _ = locs.shape
    dev = locs.device
    need_features = tree is not None if need_features is None else need_features

    if D is None:
        D = FT.pairwise_dist(locs)
    if hull is None:
        want_hull = need_features and feature_mode == "pomo"
        hull = FT.static_hull(locs) if want_hull else torch.zeros(B, N, device=dev)

    unvis = torch.ones(B, N, dtype=torch.bool, device=dev)
    bar = torch.arange(B, device=dev)
    unvis[bar, first] = False
    cur = first.clone()
    prev = first.clone()
    tour = [cur.clone()]
    if teacher is not None:
        teacher.begin(locs, first)

    has_last = last_node is not None
    if has_last:
        unvis[bar, last_node] = False

    for step in range(1, N):
        if has_last and step == N - 1:
            feas = torch.zeros(B, N, dtype=torch.bool, device=dev)
            feas[bar, last_node] = True
        else:
            feas = unvis
        if not bool(feas.any(dim=1).all()):
            raise RuntimeError(f"TSP rollout: dead end at step {step}")

        feats = None
        if need_features:
            feats = (FT.lehd_tsp_features(locs, D, cur, first, feas)
                     if feature_mode == "lehd"
                     else FT.tsp_features(locs, D, hull, cur, first, prev,
                                          feas, step))

        if teacher is not None:
            logits = teacher.step(cur, feas)
            logits = logits.masked_fill(~feas, -float("inf"))
            chosen = logits.argmax(-1)
        else:
            chosen = masked_argmax(tree.score(feats), feas)

        if collect is not None and feats is not None:
            collect.append((feats, feas.clone(), chosen.clone(), step))
        if probe is not None:
            probe(dict(locs=locs, cur=cur, first=first, prev=prev,
                       feas=feas, step=step, N=N), chosen)

        if teacher is not None and hasattr(teacher, "commit"):
            teacher.commit(chosen)
        unvis[bar, chosen] = False
        prev = cur
        cur = chosen
        tour.append(cur.clone())

    return torch.stack(tour, dim=1)

def tour_length(locs, tour):
    g = torch.gather(locs, 1, tour.unsqueeze(-1).expand(-1, -1, 2))
    return (g - torch.roll(g, -1, dims=1)).norm(dim=-1).sum(-1)

def path_length(locs, path):
    g = torch.gather(locs, 1, path.unsqueeze(-1).expand(-1, -1, 2))
    return (g[:, 1:] - g[:, :-1]).norm(dim=-1).sum(-1)

@torch.no_grad()
def pomo_cvrp_rollout(
    depot_xy, node_xy, node_demand, P_starts, *,
    teacher=None, tree=None, collect=None, need_features=None, probe=None,
):
    dev = node_xy.device
    B, N, _ = node_xy.shape
    C = N + 1
    Pn = P_starts
    need_features = tree is not None if need_features is None else need_features

    all_locs = torch.cat([depot_xy, node_xy], dim=1)
    all_dem = torch.cat([torch.zeros(B, 1, device=dev), node_demand], 1)

    if teacher is not None:
        teacher.begin(depot_xy, node_xy, node_demand)

    load = torch.ones(B, Pn, device=dev)
    visited_ninf = torch.zeros(B, Pn, C, device=dev)
    finished = torch.zeros(B, Pn, dtype=torch.bool, device=dev)

    first_customer = torch.arange(1, Pn + 1, device=dev)[None, :].expand(B, Pn)
    cur = first_customer.clone()
    prev = torch.zeros_like(cur)
    bidx = torch.arange(B, device=dev)[:, None].expand(B, Pn)
    pidx = torch.arange(Pn, device=dev)[None, :].expand(B, Pn)

    dem_BP = all_dem[:, None, :].expand(B, Pn, C)
    load -= dem_BP.gather(2, cur[:, :, None]).squeeze(2)

    visited_ninf[bidx, pidx, cur] = float("-inf")
    at_depot = cur == 0
    visited_ninf[:, :, 0][~at_depot] = 0

    seq = [torch.zeros(B, Pn, dtype=torch.long, device=dev), cur.clone()]

    if need_features:
        D_inst = FT.pairwise_dist(all_locs)
        hull_inst = FT.static_hull(all_locs)
        BP = B * Pn
        locs_f = all_locs[:, None].expand(B, Pn, C, 2).reshape(BP, C, 2)
        dem_f = all_dem[:, None].expand(B, Pn, C).reshape(BP, C)
        D_f = D_inst[:, None].expand(B, Pn, C, C).reshape(BP, C, C)
        hull_f = hull_inst[:, None].expand(B, Pn, C).reshape(BP, C)

    step = 2
    max_steps = N * 3 + 5
    step_max = float(N + max(1, int(N * 0.35)))
    while not bool(finished.all()) and step < max_steps:
        ninf = visited_ninf.clone()
        too_big = load[:, :, None] + 1e-5 < dem_BP
        ninf[too_big] = float("-inf")
        new_done = (visited_ninf == float("-inf")).all(dim=2)
        finished = finished | new_done
        ninf[:, :, 0][finished] = 0
        feas = ninf == 0

        feats = None
        if need_features:
            unvis_cust = visited_ninf > -1.0
            unvis_cust = unvis_cust.clone()
            unvis_cust[:, :, 0] = False
            feats = FT.cvrp_features(
                locs_f, dem_f, D_f, hull_f,
                cur.reshape(-1), first_customer.reshape(-1), prev.reshape(-1),
                unvis_cust.reshape(-1, C), feas.reshape(-1, C),
                load.reshape(-1), step, step_max,
            ).view(B, Pn, C, -1)

        if teacher is not None:
            sel, _, _ = teacher.model(cur, load, ninf)
        else:
            sc = tree.score(feats.reshape(B * Pn, C, -1)).view(B, Pn, C)
            sel = sc.masked_fill(~feas, -float("inf")).argmax(-1)
        sel = torch.where(finished, torch.zeros_like(sel), sel)

        if collect is not None and feats is not None:
            collect.append((feats.reshape(B * Pn, C, -1), feas.reshape(-1, C),
                            sel.reshape(-1), step, (~finished).reshape(-1)))

        if probe is not None:
            probe(dict(all_locs=all_locs, all_dem=all_dem,
                       visited_ninf=visited_ninf, cur=cur, load=load,
                       first=first_customer, feas=feas, step=step), sel)

        seq.append(sel)
        prev = cur
        cur = sel
        at_depot = sel == 0
        load = load - dem_BP.gather(2, sel[:, :, None]).squeeze(2)
        load = torch.where(at_depot, torch.ones_like(load), load)
        visited_ninf[bidx, pidx, sel] = float("-inf")
        visited_ninf[:, :, 0][~at_depot] = 0
        step += 1

    seq.append(torch.zeros(B, Pn, dtype=torch.long, device=dev))
    seq = torch.stack(seq, dim=-1)
    cost = cvrp_seq_cost(all_locs, seq)
    return seq, cost

def cvrp_seq_cost(all_locs, seq):
    B, Pn, T = seq.shape
    locs_BP = all_locs[:, None, :, :].expand(B, Pn, all_locs.size(1), 2)
    coords = locs_BP.gather(2, seq[:, :, :, None].expand(B, Pn, T, 2))
    return (coords[:, :, 1:] - coords[:, :, :-1]).norm(dim=-1).sum(-1)

@torch.no_grad()
def lehd_cvrp_rollout(
    locs_full, dem_full, *, teacher=None, tree=None, restrict_to=None,
    collect=None, capacity=1.0, need_features=None, need_seqs=True,
):
    dev = locs_full.device
    need_features = tree is not None if need_features is None else need_features
    B = locs_full.size(0)

    if restrict_to is None:
        sub_locs, sub_dem = locs_full, dem_full
        M = locs_full.size(1) - 1
        id_map = None
    else:
        L = restrict_to.size(1)
        bidx = torch.arange(B, device=dev).unsqueeze(1)
        sub_locs = torch.cat([locs_full[:, 0:1], locs_full[bidx, restrict_to]], 1)
        sub_dem = torch.cat([torch.zeros(B, 1, device=dev),
                             dem_full[bidx, restrict_to]], 1)
        M = L
        id_map = restrict_to

    C1 = M + 1
    if teacher is not None:
        cap_col = torch.full((B, C1), capacity, device=dev)
        problems = torch.cat([sub_locs, sub_dem.unsqueeze(-1),
                              cap_col.unsqueeze(-1)], dim=-1)
        teacher.model.capacity = capacity
        encoded = teacher.model.encoder(problems, capacity)
        selected = torch.zeros(B, 0, dtype=torch.long, device=dev)

    if need_features:
        D = FT.pairwise_dist(sub_locs)

    load = torch.full((B,), capacity, device=dev)
    unvis = torch.ones(B, C1, dtype=torch.bool, device=dev)
    unvis[:, 0] = False
    cur = torch.zeros(B, dtype=torch.long, device=dev)
    prev = torch.zeros(B, dtype=torch.long, device=dev)
    first = torch.zeros(B, dtype=torch.long, device=dev)
    bar = torch.arange(B, device=dev)

    order = torch.zeros(B, M, dtype=torch.long, device=dev)
    order_local = torch.zeros(B, M, dtype=torch.long, device=dev)
    via_all = torch.zeros(B, M, dtype=torch.bool, device=dev)
    seqs = [[0] for _ in range(B)] if need_seqs else None

    for step in range(M):
        dem_c = sub_dem[:, 1:]
        unvis_c = unvis[:, 1:]
        fits = dem_c <= load.unsqueeze(1) + 1e-6
        feas_direct = unvis_c & fits
        feas_via = unvis_c
        feas = torch.cat([feas_direct, feas_via], dim=1)

        feats = None
        if need_features:
            feats = FT.lehd_cvrp_features_r(sub_locs, sub_dem, D, cur,
                                            unvis, load)

        if teacher is not None:
            problems[:, :, 3] = load.unsqueeze(-1)
            remaining = problems[:, 1, 3]
            probs = teacher.model.decoder(encoded, selected, capacity, remaining).to(dev)
            argmax = probs.argmax(dim=-1)
            sel_cust = argmax % M + 1
            is_via = argmax >= M
            sel_dem = dem_c.gather(1, (sel_cust - 1).unsqueeze(-1)).squeeze(-1)
            is_via = is_via | ((~is_via) & (sel_dem > load + 1e-6))
        else:
            sc = tree.score(feats)
            act = masked_argmax(sc, feas)
            sel_cust = act % M + 1
            is_via = act >= M
            sel_dem = dem_c.gather(1, (sel_cust - 1).unsqueeze(-1)).squeeze(-1)

        if collect is not None and feats is not None:
            act_idx = torch.where(is_via, (sel_cust - 1) + M, sel_cust - 1)
            collect.append((feats, feas, act_idx, step))

        load = torch.where(is_via, torch.full_like(load, capacity), load) - sel_dem
        if id_map is None:
            gid = sel_cust
        else:
            gid = id_map.gather(1, (sel_cust - 1).unsqueeze(-1)).squeeze(-1)
        if need_seqs:
            sc_l = sel_cust.tolist(); iv = is_via.tolist()
            for i in range(B):
                if iv[i]:
                    seqs[i].append(0)
                seqs[i].append(int(sc_l[i]))
        order[:, step] = gid
        order_local[:, step] = sel_cust
        via_all[:, step] = is_via
        unvis[bar, sel_cust] = False
        prev = torch.where(is_via, torch.zeros_like(cur), cur)
        cur = sel_cust
        if teacher is not None:
            selected = torch.cat([selected, sel_cust.unsqueeze(-1)], dim=1)

    if need_seqs:
        for i in range(B):
            if seqs[i][-1] != 0:
                seqs[i].append(0)
    return order, seqs, sub_locs, dict(order_local=order_local, via=via_all)

@torch.no_grad()
def lehd_cvrp_replay(locs_full, dem_full, perm, *, teacher=None, tree=None,
                     capacity=1.0, D=None, hull=None, need_seqs=True):
    dev = locs_full.device
    B, C1, _ = locs_full.shape
    N = C1 - 1
    need_features = tree is not None

    if teacher is not None:
        cap_col = torch.full((B, C1), capacity, device=dev)
        problems = torch.cat([locs_full, dem_full.unsqueeze(-1),
                              cap_col.unsqueeze(-1)], dim=-1)
        teacher.model.capacity = capacity
        encoded = teacher.model.encoder(problems, capacity)
        selected = torch.zeros(B, 0, dtype=torch.long, device=dev)

    if need_features and D is None:
        D = FT.pairwise_dist(locs_full)

    load = torch.full((B,), capacity, device=dev)
    unvis = torch.ones(B, C1, dtype=torch.bool, device=dev)
    unvis[:, 0] = False
    cur = torch.zeros(B, dtype=torch.long, device=dev)
    prev = torch.zeros(B, dtype=torch.long, device=dev)
    first = torch.zeros(B, dtype=torch.long, device=dev)
    bar = torch.arange(B, device=dev)
    seqs = [[0] for _ in range(B)] if need_seqs else None
    via_all = torch.zeros(B, N, dtype=torch.bool, device=dev)

    for step in range(N):
        tgt = perm[:, step]
        sel_dem = dem_full.gather(1, tgt.unsqueeze(-1)).squeeze(-1)
        need_refill = sel_dem > load + 1e-6
        if teacher is not None:
            problems[:, :, 3] = load.unsqueeze(-1)
            remaining = problems[:, 1, 3]
            probs = teacher.model.decoder(encoded, selected, capacity, remaining).to(dev)
            direct = probs.gather(1, (tgt - 1).unsqueeze(-1)).squeeze(-1)
            via = probs.gather(1, (N + tgt - 1).unsqueeze(-1)).squeeze(-1)
            is_via = (via > direct) | need_refill
            selected = torch.cat([selected, tgt.unsqueeze(-1)], dim=1)
        else:
            feats = FT.lehd_cvrp_features_r(locs_full, dem_full, D, cur,
                                            unvis, load)
            sc = tree.score(feats)
            direct = sc.gather(1, (tgt - 1).unsqueeze(-1)).squeeze(-1)
            via = sc.gather(1, (N + tgt - 1).unsqueeze(-1)).squeeze(-1)
            is_via = (via > direct) | need_refill
        load = torch.where(is_via, torch.full_like(load, capacity), load) - sel_dem
        via_all[:, step] = is_via
        if need_seqs:
            iv = is_via.tolist(); tg = tgt.tolist()
            for i in range(B):
                if iv[i]:
                    seqs[i].append(0)
                seqs[i].append(int(tg[i]))
        unvis[bar, tgt] = False
        prev = torch.where(is_via, torch.zeros_like(cur), cur)
        cur = tgt

    if need_seqs:
        for i in range(B):
            if seqs[i][-1] != 0:
                seqs[i].append(0)
    costs = cost_from_order_via(locs_full, perm, via_all)
    return costs, seqs

@torch.no_grad()
def cost_from_order_via(locs, order, via):
    B, M = order.shape
    dev = locs.device
    bar = torch.arange(B, device=dev)
    depot = locs[:, 0]
    prev = depot
    total = torch.zeros(B, device=dev)
    d_dep = (locs - depot.unsqueeze(1)).norm(dim=-1)
    for t in range(M):
        c = order[:, t]
        x = locs[bar, c]
        direct = (x - prev).norm(dim=-1)
        viacost = (prev - depot).norm(dim=-1) + d_dep[bar, c]
        total = total + torch.where(via[:, t], viacost, direct)
        prev = x
    return total + (prev - depot).norm(dim=-1)

def cost_from_seqs(locs_full, seqs):
    dev = locs_full.device
    out = torch.zeros(len(seqs), device=dev)
    for i, s in enumerate(seqs):
        idx = torch.tensor(s, device=dev, dtype=torch.long)
        pts = locs_full[i].index_select(0, idx)
        out[i] = (pts[1:] - pts[:-1]).norm(dim=-1).sum()
    return out

def feature_mode_for(key):
    return "lehd" if P.SETTINGS[key]["family"] == "lehd" else "pomo"

