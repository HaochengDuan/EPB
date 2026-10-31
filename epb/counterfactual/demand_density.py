import collections
import csv
import glob
import json
import math
import os
import time

import numpy as np
import torch

from ..settings.pomo_cvrp import masked_softmax
from . import common as C

NAME = "demand_density"
SETTING = "pomo_cvrp"
STUDENT = "pomo_cvrp50"
CONFIG = os.path.join(C.HERE, "demand_density.json")
PROTOCOL = 1
ANALYSIS_KEYS = ("analysis_seed", "n_boot", "n_perm")
DESIGN = "deplete_tgt"
FEAS_PREDICATES = [lambda d, L: d <= L + 1e-5, lambda d, L: d <= L, lambda d, L: d <= L + 1e-6,
                   lambda d, L: (d - L).abs() < 1e-3, lambda d, L: (L - d).clamp_min(0) < 1e-6]


def add_args(ap):
    ap.add_argument("--batch", help="one batch of the config (all its shards, then its merge); with --inst-lo / "
                                    "--inst-hi only that shard")
    ap.add_argument("--merge", action="store_true", help="only merge the shards of --batch")
    ap.add_argument("--allow-partial", action="store_true",
                    help="merge shards that cover only a contiguous prefix of the batch (smoke tests); by default the "
                         "shards must cover the whole batch")
    ap.add_argument("--analyze", action="store_true", help="only the tables (main_table, ladder_table)")


def h02(locs, demand, current_node, mask, visited, load, first_node, override=None, return_terms=False):
    B, N1, _ = locs.shape
    device = locs.device
    eps = 1e-6

    cur_loc = locs[torch.arange(B, device=device), current_node]
    dist_to_cur = torch.norm(locs - cur_loc.unsqueeze(1), dim=-1).clamp_min(eps)
    dist2 = dist_to_cur ** 2

    load_exp = load.unsqueeze(-1).clamp_min(eps)
    unvisited = 1.0 - visited

    feasible_customers_nodepot = mask.float().clone()
    feasible_customers_nodepot[:, 0] = 0.0
    n_feasible_customers = feasible_customers_nodepot.sum(dim=-1)

    unvisited_demand = demand * unvisited
    total_unvisited_demand = unvisited_demand.sum(dim=-1)
    feasible_demand = (demand * feasible_customers_nodepot).sum(dim=-1)
    feasible_fraction = feasible_demand / (total_unvisited_demand + eps)

    proximity_score = 1.0 / dist2
    direct_score = demand / dist2

    locs_i = locs.unsqueeze(2)
    locs_j = locs.unsqueeze(1)
    pairwise_dist = torch.norm(locs_i - locs_j, dim=-1)
    radius = 0.12
    within_radius = (pairwise_dist <= radius).float()
    local_demand_density = (within_radius * unvisited_demand.unsqueeze(1)).sum(dim=-1)
    if override is not None:
        local_demand_density = override
    gravity_score = local_demand_density / dist2

    capacity_fit = (demand / load_exp).clamp(0.0, 1.0)

    fit_ratio = (demand / load_exp).clamp(0.0, 1.0)
    perfect_fit_bonus = 150.0 * (fit_ratio ** 2)

    load_factor = load.unsqueeze(-1)
    proximity_weight = 5.0 + 10.0 * load_factor
    logits = (proximity_weight * proximity_score
              + 2.0 * direct_score
              + 1.0 * gravity_score
              + 0.5 * capacity_fit
              + perfect_fit_bonus)

    is_last_feasible = (n_feasible_customers == 1).float().unsqueeze(-1)
    logits = logits + 400.0 * is_last_feasible * feasible_customers_nodepot

    is_few_feasible = ((n_feasible_customers >= 2) & (n_feasible_customers <= 3)).float().unsqueeze(-1)
    logits = logits + 100.0 * is_few_feasible * feasible_customers_nodepot

    n_fc = n_feasible_customers.clamp_min(0.0)

    depot_penalty_customers = 400.0 * n_fc

    no_feasible = (n_feasible_customers < 1).float()
    low_load = (load <= 0.15).float()
    low_feasible_fraction = (feasible_fraction < 0.3).float()
    depot_boost = 200.0 * low_load * low_feasible_fraction * no_feasible

    depot_adjustment = depot_boost - depot_penalty_customers
    logits[:, 0] = logits[:, 0] + depot_adjustment

    if return_terms:
        return logits, {"within_radius": within_radius, "ldd": local_demand_density,
                        "dist2": dist2, "gravity": gravity_score,
                        "n_feasible": n_feasible_customers}
    return logits


def h14(locs, demand, current_node, mask, visited, load, first_node, override=None, return_terms=False):
    B, N1, _ = locs.shape
    device = locs.device

    cur_loc = locs[torch.arange(B, device=device), current_node]
    depot_loc = locs[:, 0:1, :]

    dist_to_cur = torch.norm(locs - cur_loc.unsqueeze(1), dim=-1)

    unvisited_mask = (1.0 - visited[:, 1:])
    unvisited_count = unvisited_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

    total_remaining_demand = (demand[:, 1:] * unvisited_mask).sum(dim=-1, keepdim=True)

    unvisited_locs = locs[:, 1:, :]
    centroid = (unvisited_locs * unvisited_mask.unsqueeze(-1)).sum(dim=1) / unvisited_count

    dist_cur_to_centroid = torch.norm(cur_loc - centroid, dim=-1)

    dist_depot_to_centroid = torch.norm(depot_loc.squeeze(1) - centroid, dim=-1)

    capacity_slack = (load.unsqueeze(-1) - demand).clamp_min(0.0)
    fits_well = capacity_slack / load.unsqueeze(-1).clamp_min(1e-3)
    logits = -dist_to_cur + 0.5 * fits_well

    dist_to_depot = torch.norm(locs - depot_loc, dim=-1)
    savings = dist_to_depot - dist_to_cur

    unvisited_demand = demand[:, 1:] * unvisited_mask
    load_after = load.unsqueeze(-1) - demand[:, 1:]
    load_after = load_after.clamp_min(0.0)
    load_after_exp = load_after.unsqueeze(-1)
    other_demand_exp = unvisited_demand.unsqueeze(1)
    diff = torch.abs(load_after_exp - other_demand_exp)
    valid_pair = unvisited_mask.unsqueeze(-1) * unvisited_mask.unsqueeze(1)
    diff = torch.where(valid_pair.bool(), diff, torch.full_like(diff, 1e6))
    min_diff, _ = torch.min(diff, dim=-1)
    clean_fit = torch.exp(-min_diff * 5.0) * unvisited_mask
    avg_demand = total_remaining_demand / unvisited_count.clamp_min(1.0)
    load_after_normalized = load_after / load.unsqueeze(-1).clamp_min(1e-3)
    avg_demand_normalized = avg_demand / load.unsqueeze(-1).clamp_min(1e-3)
    clean_fit_normalized = torch.exp(-torch.abs(load_after_normalized - avg_demand_normalized) * 3.0)
    clean_fit_bonus = 1.5 * clean_fit * clean_fit_normalized * unvisited_mask
    logits[:, 1:] = logits[:, 1:] + clean_fit_bonus

    savings_bonus = 0.6 * savings[:, 1:] * fits_well[:, 1:] + 0.8 * savings[:, 1:] * clean_fit
    logits[:, 1:] = logits[:, 1:] + savings_bonus

    vec_to_node = locs[:, 1:, :] - depot_loc.squeeze(1).unsqueeze(1)
    angle_to_node = torch.atan2(vec_to_node[:, :, 1], vec_to_node[:, :, 0])

    cur_vec = cur_loc - depot_loc.squeeze(1)
    cur_angle = torch.atan2(cur_vec[:, 1], cur_vec[:, 0])
    first_vec = locs[torch.arange(B, device=device), first_node] - depot_loc.squeeze(1)
    first_angle = torch.atan2(first_vec[:, 1], first_vec[:, 0])
    route_angle = torch.where(current_node == 0, first_angle, cur_angle)

    angle_diff = torch.abs(angle_to_node - route_angle.unsqueeze(-1))
    angle_diff = torch.min(angle_diff, 2 * math.pi - angle_diff)
    angular_score = torch.cos(angle_diff) * unvisited_mask

    angle_diff_matrix = torch.abs(angle_to_node.unsqueeze(-1) - angle_to_node.unsqueeze(1))
    angle_diff_matrix = torch.min(angle_diff_matrix, 2 * math.pi - angle_diff_matrix)
    window = math.pi / 6.0
    cluster_mask = (angle_diff_matrix < window).float() * unvisited_mask.unsqueeze(-1) * unvisited_mask.unsqueeze(1)
    cluster_size = cluster_mask.sum(dim=-1)
    max_cluster_size = unvisited_count.squeeze(-1).clamp_min(1.0)
    cluster_score = cluster_size / max_cluster_size.unsqueeze(-1)

    cluster_demand = (cluster_mask * unvisited_demand.unsqueeze(1)).sum(dim=-1)
    cluster_demand_score = cluster_demand / total_remaining_demand.clamp_min(1e-3)
    if override is not None:
        cluster_demand_score = override

    at_depot = (current_node == 0).float().unsqueeze(-1)
    angular_weight = torch.where(at_depot.bool(), 1.0, 1.0)
    cluster_weight = torch.where(at_depot.bool(), 2.5, 0.8)
    cluster_demand_weight = torch.where(at_depot.bool(), 2.0, 0.5)

    angular_bonus = (angular_weight * angular_score + cluster_weight * cluster_score * unvisited_mask
                     + cluster_demand_weight * cluster_demand_score * unvisited_mask)
    logits[:, 1:] = logits[:, 1:] + angular_bonus

    low_load = (load < 0.3).float().unsqueeze(-1)
    far_penalty = -2.0 * (dist_to_depot[:, 1:] > 0.5).float() * low_load
    logits[:, 1:] = logits[:, 1:] + far_penalty

    terminal_mask = (unvisited_count <= 2).float()
    dist_depot_customers = dist_to_depot[:, 1:] * unvisited_mask
    max_dist_depot = dist_depot_customers.max(dim=-1, keepdim=True).values
    terminal_bonus = 2.5 * (dist_depot_customers / max_dist_depot.clamp_min(1e-3)) * terminal_mask
    logits[:, 1:] = logits[:, 1:] + terminal_bonus

    small_remaining = (unvisited_count <= 3).float()
    ample_capacity = (load.unsqueeze(-1) > total_remaining_demand).float()
    arm_completion_mask = small_remaining * ample_capacity
    dist_depot_norm = dist_depot_customers / max_dist_depot.clamp_min(1e-3)
    arm_completion_bonus = 3.0 * dist_depot_norm * arm_completion_mask
    logits[:, 1:] = logits[:, 1:] + arm_completion_bonus

    very_low_load = (load < 0.15).float().unsqueeze(-1)
    small_demand = (total_remaining_demand < 0.3).float()
    low_load_small_demand = very_low_load * small_demand
    savings_boost = 1.5 * savings[:, 1:] * low_load_small_demand
    logits[:, 1:] = logits[:, 1:] + savings_boost
    angular_boost = 1.0 * angular_score * low_load_small_demand
    logits[:, 1:] = logits[:, 1:] + angular_boost

    load_used = 1.0 - load
    demand_exceeds_capacity = (total_remaining_demand.squeeze(-1) > load).float()
    capacity_pressure = load_used + demand_exceeds_capacity

    depot_advantage = dist_cur_to_centroid - dist_depot_to_centroid

    max_customer_logit = logits[:, 1:].max(dim=-1).values
    any_customer_feasible = mask[:, 1:].any(dim=-1).float()

    depot_boost = torch.where(
        (depot_advantage > 0.1) & (capacity_pressure > 0.7) & (max_customer_logit < 0.5) &
        ((any_customer_feasible == 0) | (total_remaining_demand.squeeze(-1) > load * 1.5)),
        5.0 + 3.0 * torch.tanh(depot_advantage * 6.0) * capacity_pressure,
        torch.zeros_like(depot_advantage)
    )
    logits[:, 0] = logits[:, 0] + depot_boost

    low_pressure_penalty = 0.5 * (capacity_pressure < 0.3).float()
    logits[:, 0] = logits[:, 0] - low_pressure_penalty

    logits = logits * 2.0

    if return_terms:
        pad = lambda x: torch.cat([torch.zeros(B, 1, device=device, dtype=x.dtype), x], 1)   # noqa: E731
        cm = torch.zeros(B, N1, N1, device=device)
        cm[:, 1:, 1:] = (angle_diff_matrix < window).float()
        return logits, {"cone": cm, "cluster_demand": pad(cluster_demand),
                        "cds": pad(cluster_demand_score),
                        "D_rem": total_remaining_demand.squeeze(-1),
                        "clean_fit": pad(clean_fit), "small_remaining": small_remaining.squeeze(-1),
                        "low_load_small_demand": low_load_small_demand.squeeze(-1),
                        "ample_capacity": ample_capacity.squeeze(-1)}
    return logits


class Instrumented:

    def __init__(self, pos, fn, member_key):
        self.pos, self.fn, self.member_key = int(pos), fn, member_key

    def run(self, b, mask, override=None):
        return self.fn(b["locs"], b["demand"], b["current_node"], mask, b["visited"], b["load"], b["first_node"],
                       override=override, return_terms=True)

    def membership(self, t):
        return t[self.member_key].bool()


SCALER = 40
KMIN, KMAX_LEVEL = 1, 9


def lattice_levels(dem):
    lat = torch.tensor([k / float(SCALER) for k in range(KMAX_LEVEL + 1)], dtype=torch.float32)
    lv = torch.full((dem.numel(),), -1, dtype=torch.long)
    for k in range(KMAX_LEVEL + 1):
        lv[dem == lat[k]] = k
    return lv, lat


def load_level(L):
    q = 40.0 * float(L)
    lq = int(round(q))
    return lq, abs(q - lq)


def rooms(lv, unv_ids, lL):
    up, dn = {}, {}
    for j in unv_ids:
        lj = int(lv[j])
        if lj == lL or lj < 0:
            up[j] = dn[j] = 0
            continue
        if lj < lL:
            up[j] = max(0, min(KMAX_LEVEL, lL - 1) - lj)
            dn[j] = max(0, lj - KMIN)
        else:
            up[j] = max(0, KMAX_LEVEL - lj)
            dn[j] = max(0, lj - max(KMIN, lL + 1))
    return up, dn


def plan_edit(st, i, r, k, mem, leak_mats, lv, lL):
    dem, vis = st["demand"], st["visited"]
    cur = st["current_node"]
    n = dem.numel()
    unv = [j for j in range(1, n) if vis[j] < 0.5 and j not in (i, r, cur)]
    up, dn = rooms(lv, unv, lL)
    mi, mr = mem[i], mem[r]
    in_i = {j: bool(mi[j]) for j in unv}
    in_r = {j: bool(mr[j]) for j in unv}
    outside = [j for j in unv if not in_i[j] and not in_r[j]]

    def pattern(j):
        return tuple((bool(Mx[i, j]), bool(Mx[r, j])) for Mx in leak_mats)

    t_cands = [j for j in unv if in_i[j] and not in_r[j]]
    t_room, p_room, s_room = dn, dn, up
    t_cands = [j for j in t_cands if t_room[j] >= 1]
    if not t_cands:
        return None, "no_T_specific_node"
    if sum(t_room[j] for j in t_cands) < k:
        return None, "T_specific_capacity_short"

    used, t_spec, p_spec, left = set(), [], [], k
    for a in sorted(t_cands, key=lambda j: (-t_room[j], j)):
        if left == 0:
            break
        partners = [b for b in outside if b not in used and int(lv[b]) == int(lv[a])
                    and pattern(b) == pattern(a) and p_room[b] >= 1]
        if not partners:
            continue
        b = sorted(partners, key=lambda j: (-p_room[j], j))[0]
        u = min(t_room[a], p_room[b], left)
        t_spec.append((a, u)); p_spec.append((b, u)); used.add(b); left -= u
    if left > 0:
        return None, "no_matched_P0_partner"

    pool = [j for j in outside if j not in used and s_room[j] >= 1]
    if sum(s_room[j] for j in pool) < k:
        return None, "shared_capacity_short"
    shared, left = [], k
    for j in sorted(pool, key=lambda j: (-s_room[j], j)):
        if left == 0:
            break
        u = min(s_room[j], left)
        shared.append((j, u)); left -= u

    _, lat = lattice_levels(dem[:1])
    dT, dP = dem.clone(), dem.clone()
    for a, u in t_spec:
        dT[a] = lat[int(lv[a]) - u]
    for b, u in p_spec:
        dP[b] = lat[int(lv[b]) - u]
    for j, u in shared:
        dT[j] = lat[int(lv[j]) + u]
        dP[j] = lat[int(lv[j]) + u]
    return {"t_spec": t_spec, "p0_spec": p_spec, "shared": shared, "demand_T": dT,
            "demand_P0": dP, "n_t_cands": len(t_cands), "n_outside": len(outside),
            "patterns": [pattern(a) for a, _ in t_spec]}, None


def flip_dose(margin, per_quantum):
    if not (per_quantum > 0) or not math.isfinite(margin):
        return None
    return int(math.floor(margin / per_quantum + 1e-9)) + 1


@torch.no_grad()
def eval_states(teacher, bank, student, carrier, leak, states, dev, bs, demand_key="demand", want_teacher=True):
    acc = collections.defaultdict(list)
    for lo in range(0, len(states), bs):
        sub = states[lo:lo + bs]
        b, mask = C.cvrp_stack_states(sub, dev, demand_key)
        V = bank.compute_probs(b)[0]
        piS, w = student(b, V)
        raw = C.cvrp_raw_scores(bank, b, mask)
        zc, tc = carrier.run(b, mask)
        zl, tl = leak.run(b, mask)
        assert torch.equal(zc.float(), raw[:, carrier.pos]), "carrier copy != bank program"
        assert torch.equal(zl.float(), raw[:, leak.pos]), "cone-program copy != bank program"
        acc["V"].append(V.cpu()); acc["w"].append(w.cpu()); acc["piS"].append(piS.cpu())
        acc["mask"].append(mask.cpu()); acc["raw"].append(raw.cpu())
        for k_ in ("ldd", "gravity", "dist2"):
            acc["c_" + k_].append(tc[k_].cpu())
        for k_ in ("cluster_demand", "cds", "clean_fit", "D_rem"):
            acc["l_" + k_].append(tl[k_].cpu())
        if want_teacher:
            acc["piT"].append(C.cvrp_teacher_probs(teacher, b, mask).cpu())
    return {k: torch.cat(v, 0) for k, v in acc.items()}


def ev_slice(EV, j):
    return {k: v[j] for k, v in EV.items()}


@torch.no_grad()
def structures(carrier, leak, states, dev, bs=1024):
    S, L = [], []
    for lo in range(0, len(states), bs):
        b, mask = C.cvrp_stack_states(states[lo:lo + bs], dev)
        _, tc = carrier.run(b, mask)
        _, tl = leak.run(b, mask)
        S.append(carrier.membership(tc).cpu()); L.append(leak.membership(tl).cpu())
    return torch.cat(S), torch.cat(L)


@torch.no_grad()
def prefilter(states, cfg, carrier, dev, chunk=2048):
    keep = []
    kcap = cfg["k_max"]
    for lo in range(0, len(states), chunk):
        sub = states[lo:lo + chunk]
        b, mask = C.cvrp_stack_states(sub, dev)
        z, t = carrier.run(b, mask)
        S = carrier.membership(t)
        n, N1 = mask.shape
        ar = torch.arange(n, device=dev)
        cust = mask.clone(); cust[:, 0] = False
        locs = b["locs"].double()
        cur = b["current_node"]
        dcur = (locs - locs[ar, cur][:, None]).norm(dim=-1).masked_fill(~cust, float("inf"))
        r = dcur.argmin(1)
        lv = torch.round(b["demand"].double() * 40).long()
        lL = torch.round(b["load"].double() * 40).long()[:, None]
        on_lat = ((b["load"].double() * 40 - lL[:, 0]).abs() < 1e-3)
        unv = b["visited"] < 0.5
        unv[:, 0] = False
        idx = torch.arange(N1, device=dev)[None, :]
        elig = unv & (idx != r[:, None]) & (idx != cur[:, None])
        feas_lv = lv < lL
        dn = torch.where(feas_lv, torch.clamp(lv - 1, min=0),
                         torch.clamp(lv - torch.clamp(lL + 1, min=1), min=0))
        dn = torch.where(lv == lL, torch.zeros_like(dn), dn)
        marg = (z - z[ar, r][:, None]).double()
        cand = cust & (idx != r[:, None]) & (marg > 0)
        Sr = S[ar, r]
        eye = torch.eye(N1, dtype=torch.bool, device=dev)[None]
        D = t["dist2"].double()
        has = (S & ~Sr[:, None, :] & elig[:, None, :] & (dn >= 1)[:, None, :] & ~eye).any(-1)
        pq = (1 / 40) / D
        ok_any = cand & has & (marg < kcap * pq * (1 + 1e-9))
        ok = (cur != 0) & (cust.sum(1) >= 2) & on_lat & ok_any.any(1)
        keep.extend([lo + int(j) for j in ok.nonzero().flatten()])
    return keep


def factual(states, EV, cfg, carrier):
    m = carrier.pos
    mask = EV["mask"]
    piS, piT = EV["piS"].double(), EV["piT"].double()
    w, V = EV["w"].double(), EV["V"].double()
    n = len(states)
    ar = torch.arange(n)
    locs = torch.stack([s["locs"] for s in states]).double()
    cur = torch.tensor([s["current_node"] for s in states])
    cust = mask.clone(); cust[:, 0] = False
    tS = piS.masked_fill(~mask, -1e9).argmax(1)
    tT = piT.masked_fill(~mask, -1e9).argmax(1)
    dcur = (locs - locs[ar, cur][:, None]).norm(dim=-1).masked_fill(~cust, float("inf"))
    rnn = dcur.argmin(1)
    Cm = w * (V[ar, :, tS] - V[ar, :, rnn])
    MS = piS[ar, tS] - piS[ar, rnn]
    alive = torch.ones(n, dtype=torch.bool)
    counts = {}

    def step(name, ok):
        nonlocal alive
        counts[name] = int((alive & ~ok).sum())
        alive = alive & ok

    step("current_is_depot", cur != 0)
    step("fewer_than_two_feasible_customers", cust.sum(1) >= 2)
    step("teacher_student_disagree", tS == tT)
    step("selected_action_is_depot", tS != 0)
    step("target_is_nearest_neighbour", tS != rnn)
    step("carrier_does_not_support_target", Cm[:, m] > 0)
    step("student_margin_not_positive", MS > 0)
    A = Cm[:, m] / MS.clamp_min(1e-300)
    cons = ((A >= cfg["construct_A_min"]) | (w[:, m] >= cfg["construct_w_min"])) & \
           (MS >= cfg["construct_MS_min"])
    counts["n_factual"] = int(alive.sum())
    step("outside_construction_population", cons)
    rows = []
    for k in alive.nonzero().flatten().tolist():
        i, r = int(tS[k]), int(rnn[k])
        Ck = Cm[k]
        rows.append({"_k": k, "instance_id": states[k]["instance_id"], "pomo": states[k]["pomo"],
                     "step_i": states[k]["step_i"], "target": i, "r_nn": r,
                     "load": float(states[k]["load"]), "M_S": float(MS[k]),
                     "M_T": float(piT[k, i] - piT[k, r]), "A_c": float(A[k]),
                     "w_c": float(w[k, m]), "C_c": float(Ck[m]),
                     "n_feasible_customers": int(cust[k].sum()),
                     "n_unvisited": int((states[k]["visited"][1:] < 0.5).sum()),
                     "d_target": float(dcur[k, i]), "d_rnn": float(dcur[k, r]),
                     "C_rank": int((Ck > Ck[m]).sum()) + 1, "w_rank": int((w[k] > w[k, m]).sum()) + 1,
                     "A_max_other": float(max(float(Ck[q]) for q in range(Ck.numel()) if q != m) / MS[k]),
                     "carrier_margin": float(EV["raw"][k, m, i].double() - EV["raw"][k, m, r].double()),
                     "D_i": float(EV["c_dist2"][k, i]), "D_r": float(EV["c_dist2"][k, r]),
                     "ldd_i": float(EV["c_ldd"][k, i]), "ldd_r": float(EV["c_ldd"][k, r]),
                     "V_c_i": float(V[k, m, i]), "V_c_r": float(V[k, m, r])})
    return rows, counts


def build_triplets(st, f, cfg, Sk, Lk):
    i, r = f["target"], f["r_nn"]
    lv, _ = lattice_levels(st["demand"])
    lL, off = load_level(st["load"])
    out, why = [], []
    if off > 1e-3:
        return [], ["load_off_lattice"]
    if (lv[1:] < 1).any():
        return [], ["demand_off_lattice"]
    pq = (1.0 / 40.0) / f["D_i"]
    kst = flip_dose(f["carrier_margin"], pq)
    if kst is None or kst > cfg["k_max"]:
        return [], [f"{DESIGN}|dose_gt_kmax"]
    for mo in cfg["overshoot_grid"]:
        k = kst + mo - 1
        plan, w_ = plan_edit(st, i, r, k, Sk, [Lk], lv, lL)
        if plan is None:
            why.append(f"{DESIGN}|m{mo}|{w_}")
            continue
        t_nodes = [a for a, _ in plan["t_spec"]]
        p_nodes = [b for b, _ in plan["p0_spec"]]
        s_nodes = [j for j, _ in plan["shared"]]
        out.append({
            "st": st, "fact": f, "design": DESIGN, "m_over": mo, "k_star": kst, "k": k,
            "target": i, "r_nn": r, "current_node": st["current_node"],
            "load": float(st["load"]), "visited": st["visited"], "demand_base": st["demand"],
            "demand_T": plan["demand_T"], "demand_P0": plan["demand_P0"], "plan": plan,
            "edited_T": t_nodes + s_nodes, "edited_P0": p_nodes + s_nodes,
            "pred_drop": k * pq, "per_quantum": pq, "load_level": lL,
            "t_spec": json.dumps(plan["t_spec"]), "p0_spec": json.dumps(plan["p0_spec"]),
            "shared": json.dumps(plan["shared"]), "n_t_spec": len(t_nodes),
            "n_shared": len(s_nodes), "patterns": json.dumps(plan["patterns"]),
            "t_spec_feasible": int(sum(int(lv[a]) < lL for a in t_nodes)),
            "t_spec_levels": json.dumps([int(lv[a]) for a in t_nodes]),
        })
        why.append(f"{DESIGN}|m{mo}|ok")
    return out, why


def check_invariants(tr, ev, cfg, carrier):
    e, fail = {}, []
    i, r, cur = tr["target"], tr["r_nn"], tr["current_node"]
    m, cone = carrier.pos, cfg["cone_program"]
    base, vis = tr["demand_base"], tr["visited"]
    unv = (vis < 0.5).clone(); unv[0] = False
    L = tr["st"]["load_t"].float()
    s0 = float((base.double() * unv.double()).sum())
    edited = {"T": tr["edited_T"], "P0": tr["edited_P0"]}
    for arm in ("T", "P0"):
        d = tr["demand_" + arm]
        keep = torch.ones_like(base, dtype=torch.bool)
        for x in edited[arm]:
            keep[x] = False
        if not torch.equal(d[keep], base[keep]):
            fail.append(f"demand_changed_outside_edited_{arm}")
        if not (d[i] == base[i] and d[r] == base[r] and d[cur] == base[cur]):
            fail.append(f"target_reference_or_current_demand_changed_{arm}")
        lvx, _ = lattice_levels(d)
        for x in edited[arm]:
            if not (1 <= int(lvx[x]) <= 9) or vis[x] >= 0.5:
                fail.append(f"edited_node_off_lattice_or_visited_{arm}")
        s1 = float((d.double() * unv.double()).sum())
        e[f"remsum_rel_{arm}"] = abs(s1 - s0) / max(s0, 1e-12)
        if e[f"remsum_rel_{arm}"] > cfg["remsum_rel_tol"]:
            fail.append(f"remaining_demand_moved_{arm}")
        for q, P in enumerate(FEAS_PREDICATES):
            if not torch.equal(P(d, L) & unv, P(base, L) & unv):
                fail.append(f"feasibility_predicate{q}_changed_{arm}")
        if not torch.equal(ev[arm]["mask"], ev["base"]["mask"]):
            fail.append(f"mask_flip_{arm}")
        if not torch.equal(C.cvrp_feasibility_mask(d, vis, cur, float(tr["load"])), ev["base"]["mask"]):
            fail.append(f"mask_recompute_flip_{arm}")
    dT, dP = tr["demand_T"], tr["demand_P0"]
    fm = ev["base"]["mask"].clone(); fm[0] = False
    if not torch.equal(torch.sort(dT[unv]).values, torch.sort(dP[unv]).values):
        fail.append("unvisited_multiset_differs_T_P0")
    if not torch.equal(torch.sort(dT[fm]).values, torch.sort(dP[fm]).values):
        fail.append("feasible_multiset_differs_T_P0")
    e["displacement_T"] = float((dT.double() - base.double()).abs().sum())
    e["displacement_P0"] = float((dP.double() - base.double()).abs().sum())
    if abs(e["displacement_T"] - e["displacement_P0"]) > 1e-6:
        fail.append("displacement_not_matched")

    for node, tag in ((i, "i"), (r, "r")):
        if not bool(ev["P0"]["c_ldd"][node] == ev["base"]["c_ldd"][node]):
            fail.append(f"placebo_moved_carrier_aggregate_{tag}")
        e[f"ldd_{tag}_T_minus_base"] = float(ev["T"]["c_ldd"][node]) - float(ev["base"]["c_ldd"][node])
    moved, still = i, r
    if not bool(ev["T"]["c_ldd"][still] == ev["base"]["c_ldd"][still]):
        fail.append("treatment_moved_the_other_aggregate")
    e["ldd_moved_expected"] = -tr["k"] / 40.0
    e["ldd_moved_err"] = abs(float(ev["T"]["c_ldd"][moved].double() - ev["base"]["c_ldd"][moved].double())
                             - e["ldd_moved_expected"])
    if e["ldd_moved_err"] > 1e-5:
        fail.append("treatment_aggregate_change_wrong")

    def tm(arm):
        return float(ev[arm]["c_gravity"][i].double() - ev[arm]["c_gravity"][r].double())

    def zm(arm, q):
        return float(ev[arm]["raw"][q][i].double() - ev[arm]["raw"][q][r].double())
    pred = tr["pred_drop"]
    e["E_term_pred"] = pred
    e["E_term_obs"] = tm("P0") - tm("T")
    e["E_full_obs"] = zm("P0", m) - zm("T", m)
    zmag = max(abs(float(ev["base"]["raw"][m][i])), abs(float(ev["base"]["raw"][m][r])), abs(pred), 1.0)
    gmag = max(abs(float(ev["base"]["c_gravity"][i])), abs(float(ev["base"]["c_gravity"][r])), abs(pred), 1.0)
    e["analytic_err_term"] = abs(e["E_term_obs"] - pred)
    e["analytic_err_full"] = abs(e["E_full_obs"] - pred)
    e["analytic_tol_term"] = cfg["invariance_tol"] + cfg["rel_tol"] * gmag
    e["analytic_tol_full"] = cfg["invariance_tol"] + cfg["rel_tol"] * zmag
    if e["analytic_err_term"] > e["analytic_tol_term"]:
        fail.append("analytic_prediction_failed_term")
    if e["analytic_err_full"] > e["analytic_tol_full"]:
        fail.append("analytic_prediction_failed_full")
    e["carrier_margin_base"] = zm("base", m)
    e["carrier_margin_P0_minus_base"] = zm("P0", m) - zm("base", m)
    e["carrier_margin_T"] = zm("T", m)
    if e["carrier_margin_P0_minus_base"] != 0.0:
        fail.append("placebo_moved_carrier_margin")
    if not (zm("base", m) > 0 and zm("T", m) < 0):
        fail.append("flip_not_realised")
    e["leak14_cd_max_abs"] = max(abs(float(ev["P0"]["l_cluster_demand"][x]) - float(ev["T"]["l_cluster_demand"][x]))
                                 for x in (i, r))
    if e["leak14_cd_max_abs"] > cfg["invariance_tol"]:
        fail.append("leak14_cone_demand_differs_T_P0")
    for x in (i, r):
        if not bool(ev["T"]["l_clean_fit"][x] == ev["P0"]["l_clean_fit"][x]):
            fail.append("leak14_clean_fit_differs_T_P0")
    e["leak14_margin_P0_minus_T"] = zm("P0", cone) - zm("T", cone)
    for q in cfg["elementwise"]:
        if not (zm("T", q) == zm("base", q) and zm("P0", q) == zm("base", q)):
            fail.append(f"elementwise_program_{q:02d}_moved")
    e["pinned_max_abs"] = max(abs(zm("P0", q) - zm("T", q)) for q in cfg["pinned"])
    if e["pinned_max_abs"] > cfg["pinned_tol"]:
        fail.append("pinned_program_moved")
    e["leak12_margin_P0_minus_T"] = zm("P0", cfg["normalised_program"]) - zm("T", cfg["normalised_program"])
    for arm in ("base", "T", "P0"):
        rc = float((ev[arm]["piS"] - (ev[arm]["w"][:, None] * ev[arm]["V"]).sum(0)).abs().max())
        e[f"recon_{arm}"] = rc
        if rc > cfg["recon_tol"]:
            fail.append(f"mixture_reconstruction_{arm}")
    return e, fail


def metrics(tr, ev, cfg, carrier):
    i, r = tr["target"], tr["r_nn"]
    m = carrier.pos
    eps = cfg["eps_logodds"]
    M = ev["base"]["V"].shape[0]

    def E(v, q):
        return float(ev[v]["raw"][q][i].double() - ev[v]["raw"][q][r].double())

    def Rw(v, q):
        lw = torch.log(ev[v]["w"].double() + eps)
        return float(lw[q] - (lw.sum() - lw[q]) / (M - 1))

    def Cq(v, q):
        return float(ev[v]["w"][q].double() * (ev[v]["V"][q][i].double() - ev[v]["V"][q][r].double()))

    def MS(v):
        return float(ev[v]["piS"][i].double() - ev[v]["piS"][r].double())

    def MT(v):
        return float(ev[v]["piT"][i].double() - ev[v]["piT"][r].double())

    def LO(v, key, e_=eps):
        return math.log(float(ev[v][key][i]) + e_) - math.log(float(ev[v][key][r]) + e_)

    def top1(v, key):
        return int(ev[v][key].masked_fill(~ev[v]["mask"], -1e9).argmax())

    def Et(v):
        return float(ev[v]["c_gravity"][i].double() - ev[v]["c_gravity"][r].double())

    o = {"g_E": Et("P0") - Et("T"), "g_Efull": E("P0", m) - E("T", m),
         "g_E_nats": (Et("P0") - Et("T")) / cfg["tau_h"],
         "g_R": Rw("P0", m) - Rw("T", m), "g_C": Cq("P0", m) - Cq("T", m),
         "g_MS": MS("P0") - MS("T"), "g_MT": MT("P0") - MT("T"),
         "g_LS": LO("P0", "piS") - LO("T", "piS"), "g_LT": LO("P0", "piT") - LO("T", "piT"),
         "g_LT_eps1e-6": LO("P0", "piT", 1e-6) - LO("T", "piT", 1e-6),
         "base_MS": MS("base"), "base_MT": MT("base"), "base_C": Cq("base", m),
         "base_w": float(ev["base"]["w"][m]), "base_R": Rw("base", m), "base_E": E("base", m),
         "base_Eterm": Et("base"), "base_LS": LO("base", "piS"), "base_LT": LO("base", "piT"),
         "base_Vc_i": float(ev["base"]["V"][m][i]), "base_Vc_r": float(ev["base"]["V"][m][r]),
         "T_Vc_i": float(ev["T"]["V"][m][i]), "T_Vc_r": float(ev["T"]["V"][m][r]),
         "base_piT_i": float(ev["base"]["piT"][i]), "base_piT_r": float(ev["base"]["piT"][r]),
         "base_piS_i": float(ev["base"]["piS"][i]), "base_piS_r": float(ev["base"]["piS"][r])}
    for X, f_ in (("MS", MS), ("MT", MT)):
        o[f"P0_minus_base_{X}"] = f_("P0") - f_("base")
        o[f"T_minus_base_{X}"] = f_("T") - f_("base")
    for X, key in (("LS", "piS"), ("LT", "piT")):
        o[f"P0_minus_base_{X}"] = LO("P0", key) - LO("base", key)
        o[f"T_minus_base_{X}"] = LO("T", key) - LO("base", key)
    o["P0_minus_base_C"] = Cq("P0", m) - Cq("base", m)
    o["T_minus_base_C"] = Cq("T", m) - Cq("base", m)
    for v in ("base", "P0", "T"):
        o[f"top1_S_{v}"] = top1(v, "piS")
        o[f"top1_T_{v}"] = top1(v, "piT")
        o[f"top1_carrier_{v}"] = int(ev[v]["V"][m].masked_fill(~ev[v]["mask"], -1e9).argmax())
    o["flip_S_T_arm"] = int(o["top1_S_T"] != o["top1_S_base"])
    o["flip_S_P0_arm"] = int(o["top1_S_P0"] != o["top1_S_base"])
    o["flip_T_T_arm"] = int(o["top1_T_T"] != o["top1_T_base"])
    o["flip_T_P0_arm"] = int(o["top1_T_P0"] != o["top1_T_base"])
    o["flip_S_T_to_rnn"] = int(o["top1_S_T"] == r)
    o["flip_T_T_to_rnn"] = int(o["top1_T_T"] == r)
    o["carrier_top1_T_is_r"] = int(o["top1_carrier_T"] == r)
    ed = sorted(set(tr["edited_T"]) | set(tr["edited_P0"]))
    o["flip_S_T_to_edited"] = int(o["top1_S_T"] in ed)
    o["flip_T_T_to_edited"] = int(o["top1_T_T"] in ed)
    for v in ("base", "T", "P0"):
        o[f"teacher_mass_edited_{v}"] = float(sum(float(ev[v]["piT"][a]) for a in ed))
        o[f"student_mass_edited_{v}"] = float(sum(float(ev[v]["piS"][a]) for a in ed))
    for q in range(M):
        o[f"gE_m{q:02d}"] = E("P0", q) - E("T", q)
        o[f"gR_m{q:02d}"] = Rw("P0", q) - Rw("T", q)
        o[f"gC_m{q:02d}"] = Cq("P0", q) - Cq("T", q)
        o[f"baseC_m{q:02d}"] = Cq("base", q)
        o[f"basew_m{q:02d}"] = float(ev["base"]["w"][q])
    return o


@torch.no_grad()
def clamp_block(bank, student, carrier, trips, EV, dev, bs, leak_programs):
    m = carrier.pos
    tau = bank.tau_h
    modes = ["component", "whole", "leak", "leak_plus_carrier", "all"]
    out = {f"g_MS_clamp_{k}_{v}": [] for k in modes for v in ("fixed", "e2e")}
    out.update({f"g_LS_clamp_{k}_e2e": [] for k in modes})
    out.update({f"g_R_clamp_{k}_e2e": [] for k in modes})
    out["clamp_component_selfcheck"] = []
    M = EV["base"]["V"].shape[1]
    for lo in range(0, len(trips), bs):
        sub = trips[lo:lo + bs]
        sl = slice(lo, lo + len(sub))
        Vb = EV["base"]["V"][sl].to(dev)
        ldd_b = EV["base"]["c_ldd"][sl].to(dev)
        res = {}
        for arm in ("P0", "T"):
            b, mask = C.cvrp_stack_states([dict(t["st"], demand_E=t["demand_" + arm]) for t in sub], dev,
                                          demand_key="demand_E")
            Va = EV[arm]["V"][sl].to(dev)
            wa = EV[arm]["w"][sl].to(dev)
            zc, _ = carrier.run(b, mask, override=ldd_b)
            Vc = Va.clone()
            Vc[:, m] = masked_softmax((zc.float() / tau).masked_fill(~mask, float("-inf")), mask)
            if arm == "T":
                zs, ts = carrier.run(b, mask)
                Vs = masked_softmax((zs.float() / tau).masked_fill(~mask, float("-inf")), mask)
                selfc = (Vs - Va[:, m]).abs().amax(-1).cpu()
            variants = {"component": Vc}
            Vw = Va.clone(); Vw[:, m] = Vb[:, m]; variants["whole"] = Vw
            Vl = Va.clone()
            for q in leak_programs:
                Vl[:, q] = Vb[:, q]
            variants["leak"] = Vl
            Vlc = Vl.clone(); Vlc[:, m] = Vb[:, m]; variants["leak_plus_carrier"] = Vlc
            variants["all"] = Vb.clone()
            for kname, V in variants.items():
                pi_fix = (wa.unsqueeze(-1) * V).sum(1)
                pi_e2e, w_e2e = student(b, V)
                res[(arm, kname, "fixed")] = pi_fix.double().cpu()
                res[(arm, kname, "e2e")] = pi_e2e.double().cpu()
                res[(arm, kname, "w")] = w_e2e.double().cpu()
        for j, t in enumerate(sub):
            i, r = t["target"], t["r_nn"]
            out["clamp_component_selfcheck"].append(float(selfc[j]))
            for kname in modes:
                for v in ("fixed", "e2e"):
                    gP = float(res[("P0", kname, v)][j, i] - res[("P0", kname, v)][j, r])
                    gT = float(res[("T", kname, v)][j, i] - res[("T", kname, v)][j, r])
                    out[f"g_MS_clamp_{kname}_{v}"].append(gP - gT)
                lo_ = lambda p: math.log(float(p[j, i]) + 1e-9) - math.log(float(p[j, r]) + 1e-9)   # noqa: E731
                out[f"g_LS_clamp_{kname}_e2e"].append(lo_(res[("P0", kname, "e2e")]) -
                                                      lo_(res[("T", kname, "e2e")]))

                def rw(wv):
                    lw = torch.log(wv[j] + 1e-9)
                    return float(lw[m] - (lw.sum() - lw[m]) / (M - 1))
                out[f"g_R_clamp_{kname}_e2e"].append(rw(res[("P0", kname, "w")]) - rw(res[("T", kname, "w")]))
    return out


FACT_KEYS_SKIP = ("_k",)


@torch.no_grad()
def run_shard(cfg, teacher, bank, student, dev, inst_lo, inst_hi):
    t0 = time.time()
    carrier = Instrumented(cfg["carrier"], h02, "within_radius")
    leak = Instrumented(cfg["cone_program"], h14, "cone")
    funnel = collections.Counter()
    facts, trips = [], []
    bsz = cfg["rollout_batch_instances"]
    for lo in range(inst_lo, inst_hi, bsz):
        batches = C.cvrp_instances(cfg["seed"], lo, lo + bsz, bsz, cfg["problem_size"])

        def _pf(s):
            if not s:
                return []
            return [s[j] for j in prefilter(s, cfg, carrier, dev)]
        states, nseen = C.cvrp_collect_states(teacher, cfg, dev, batches, cfg["instance_id_offset"] + lo,
                                              on_batch=_pf)
        funnel["states"] += nseen
        funnel["after_prefilter"] += len(states)
        if not states:
            continue
        EV = eval_states(teacher, bank, student, carrier, leak, states, dev, cfg["eval_batch"])
        frows, counts = factual(states, EV, cfg, carrier)
        for k_, v_ in counts.items():
            funnel["fact:" + k_] += v_
        if not frows:
            continue
        Sm, Lm = structures(carrier, leak, [states[f["_k"]] for f in frows], dev)
        for j, f in enumerate(frows):
            st = states[f["_k"]]
            got, why = build_triplets(st, f, cfg, Sm[j], Lm[j])
            for w_ in why:
                funnel["con:" + w_] += 1
            f["construction"] = ";".join(why)
            facts.append({k_: v_ for k_, v_ in f.items() if k_ not in FACT_KEYS_SKIP})
            for tr in got:
                tr["base_raw"] = EV["raw"][f["_k"]].clone()
                trips.append(tr)
        del states, EV
    funnel["triplets_built"] = len(trips)
    print(f"[run {inst_lo}-{inst_hi}] states {funnel['states']} prefilter {funnel['after_prefilter']} "
          f"facts {len(facts)} triplets built {len(trips)} ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    if trips:
        EVA = {"base": eval_states(teacher, bank, student, carrier, leak, [t["st"] for t in trips], dev,
                                   cfg["eval_batch"])}
        for arm in ("T", "P0"):
            EVA[arm] = eval_states(teacher, bank, student, carrier, leak,
                                   [dict(t["st"], demand_E=t["demand_" + arm]) for t in trips], dev,
                                   cfg["eval_batch"], demand_key="demand_E")
        keep = []
        for j, tr in enumerate(trips):
            ev = {arm: ev_slice(EVA[arm], j) for arm in ("base", "T", "P0")}
            if not torch.equal(ev["base"]["raw"], tr["base_raw"]):
                funnel["INV:base_reevaluation_raw_differs"] += 1
            errs, fails = check_invariants(tr, ev, cfg, carrier)
            if fails:
                for f_ in sorted(set(fails)):
                    funnel[f"INV:{f_}"] += 1
                continue
            mt = metrics(tr, ev, cfg, carrier)
            f = tr["fact"]
            row = {**{k_: v_ for k_, v_ in f.items() if k_ not in FACT_KEYS_SKIP + ("construction",)},
                   "design": tr["design"], "m_over": tr["m_over"], "k_star": tr["k_star"], "k": tr["k"],
                   "per_quantum": tr["per_quantum"], "load_level": tr["load_level"],
                   "t_spec": tr["t_spec"], "p0_spec": tr["p0_spec"], "shared": tr["shared"],
                   "n_t_spec": tr["n_t_spec"], "n_shared": tr["n_shared"], "patterns": tr["patterns"],
                   "t_spec_feasible": tr["t_spec_feasible"], "t_spec_levels": tr["t_spec_levels"],
                   **mt, **{f"inv_{k_}": v_ for k_, v_ in errs.items()}}
            rows.append(row)
            keep.append(j)
        funnel["triplets_accepted"] = len(rows)
        print(f"[run {inst_lo}-{inst_hi}] accepted {len(rows)} ({time.time() - t0:.0f}s)", flush=True)
        if rows:
            kept = [trips[j] for j in keep]
            sub = {arm: {k_: v_[keep] for k_, v_ in EVA[arm].items()} for arm in EVA}
            cl = clamp_block(bank, student, carrier, kept, sub, dev, cfg["clamp_batch"], cfg["leak_programs"])
            for k_, v_ in cl.items():
                for row, val in zip(rows, v_):
                    row[k_] = val
            print(f"[run {inst_lo}-{inst_hi}] clamps done ({time.time() - t0:.0f}s)", flush=True)
    return {"facts": facts, "rows": rows, "funnel": dict(funnel), "elapsed_s": time.time() - t0}


def write_shard(res, out, meta):
    os.makedirs(out, exist_ok=True)
    for name, data in (("facts.csv", res["facts"]), ("triplets.csv", res["rows"])):
        if data:
            C.write_csv(os.path.join(out, name), data, sorted({k for d in data for k in d}))
    fj = {"funnel": res["funnel"], "n_facts": len(res["facts"]), "n_triplets": len(res["rows"]),
          "elapsed_s": res["elapsed_s"], **meta}
    json.dump(fj, open(os.path.join(out, "funnel.json.tmp"), "w"), indent=1)
    os.replace(os.path.join(out, "funnel.json.tmp"), os.path.join(out, "funnel.json"))


def merge_batch(out, cfg, batch, fp, allow_unverified=False, allow_partial=False):
    shards = sorted(glob.glob(os.path.join(out, "shards", "shard_*")),
                    key=lambda p: int(os.path.basename(p).split("_")[1]))
    missing = [s for s in shards if not os.path.exists(os.path.join(s, "funnel.json"))]
    if not shards or missing:
        raise SystemExit(f"STOP: no or unfinished shards: {missing or out}")
    fjs = [json.load(open(os.path.join(sd, "funnel.json"))) for sd in shards]
    names = [os.path.basename(sd) for sd in shards]
    for name, fj in zip(names, fjs):
        if name != f"shard_{fj['inst_lo']}_{fj['inst_hi']}":
            raise SystemExit(f"STOP: {name} holds instances [{fj['inst_lo']}, {fj['inst_hi']})")
    shared = C.common_fingerprint([fj.get("fingerprint") for fj in fjs], names, f"shards of {batch}",
                                  allow_unverified)
    if shared is not None:
        C.check_fingerprint(shared, fp, f"the shards of {batch}")
    for name, fj in zip(names, fjs):
        if fj.get("config") != fjs[0].get("config"):
            raise SystemExit(f"STOP: shards {name} and {names[0]} of {batch} were run with different configurations")
    if shared is None and C.config_sha256(fjs[0].get("config") or {}, ANALYSIS_KEYS) != fp["config"]:
        raise SystemExit(f"STOP: the shards of {batch} were run with a configuration other than this config's")
    start = cfg["batches"][batch]["start"]
    end = C.check_tiling([(fj["inst_lo"], fj["inst_hi"], n) for n, fj in zip(names, fjs)], start,
                         start + cfg["batch_instances"], f"shards of {batch}", complete=not allow_partial)
    for name in ("facts.csv", "triplets.csv"):
        rows, keys = [], []
        for sd in shards:
            p = os.path.join(sd, name)
            if not os.path.exists(p):
                continue
            with open(p) as f:
                rd = csv.DictReader(f)
                for k in rd.fieldnames:
                    if k not in keys:
                        keys.append(k)
                rows.extend(rd)
        C.write_csv(os.path.join(out, name), rows, sorted(keys))
        print(f"  merged {name}: {len(rows)} rows from {len(shards)} shards")
    fun, meta = collections.Counter(), {"shards": [], "elapsed_s": 0.0, "n_facts": 0, "n_triplets": 0}
    for sd, fj in zip(shards, fjs):
        fun.update(fj["funnel"])
        meta["elapsed_s"] += fj["elapsed_s"]; meta["n_facts"] += fj["n_facts"]; meta["n_triplets"] += fj["n_triplets"]
        meta["shards"].append({"dir": os.path.basename(sd), "inst_lo": fj["inst_lo"], "inst_hi": fj["inst_hi"],
                               "n_triplets": fj["n_triplets"]})
    meta["config"] = fjs[0]["config"]
    meta.update({"instances": [start, end], "complete": end == start + cfg["batch_instances"],
                 "fingerprint": shared})
    if not meta["complete"]:
        print(f"  NOTE: partial batch {batch}: instances [{start}, {end}) of [{start}, "
              f"{start + cfg['batch_instances']}) (--allow-partial)")
    json.dump({"funnel": dict(fun), **meta}, open(os.path.join(out, "funnel.json"), "w"), indent=1)
    print(f"  funnel: states {fun['states']} prefilter {fun['after_prefilter']} facts {meta['n_facts']} "
          f"triplets {meta['n_triplets']} ({meta['elapsed_s']:.0f}s)")


READOUTS = [("g_E", "E carrier term (raw units, analytic)"), ("g_E_nats", "E carrier term (nats)"),
            ("g_R", "R carrier routing log-weight"), ("g_C", "C carrier contribution"),
            ("g_LS", "L_S student log-odds"), ("g_MS", "M_S student prob. gap"),
            ("g_LT", "L_T teacher log-odds (eps 1e-9)"), ("g_LT_eps1e-6", "L_T teacher log-odds (eps 1e-6)"),
            ("g_MT", "M_T teacher prob. gap")]
DECOMP = [("T_minus_base_MS", "M_S: T - base"), ("P0_minus_base_MS", "M_S: P0 - base"),
          ("T_minus_base_MT", "M_T: T - base"), ("P0_minus_base_MT", "M_T: P0 - base"),
          ("T_minus_base_LT", "L_T: T - base"), ("P0_minus_base_LT", "L_T: P0 - base")]
CLAMPS = ["component", "whole", "leak", "leak_plus_carrier", "all"]


def fl(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", "nan", None) else float("nan")


def stat(pop, key, rng, cfg):
    v = np.array([fl(r, key) for r in pop])
    ids = np.array([int(float(r["instance_id"])) for r in pop])
    ok = np.isfinite(v)
    v, ids = v[ok], ids[ok]
    if len(v) < 5:
        return None
    lo, hi = C.bootstrap_ci(v, ids, "demand", cfg["n_boot"], rng=rng)
    return {"mean_state": float(v.mean()), "ci95_state_lo": lo, "ci95_state_hi": hi,
            "p_signflip_state_mean": C.signflip_p(v, ids, "demand", cfg["n_perm"], rng=rng),
            "mean_instance": C.instance_mean(v, ids, "demand"),
            "pct_positive": float(100 * (v > 1e-6).mean()), "n": int(len(v)),
            "n_instances": int(len(set(ids.tolist()))),
            "median": float(np.median(v)), "mean_abs": float(np.abs(v).mean())}


def block(pop, rng, cfg):
    res = {}
    for key, _ in READOUTS + DECOMP:
        res[key] = stat(pop, key, rng, cfg)
    gms = np.array([fl(r, "g_MS") for r in pop])
    med = {}
    for c in CLAMPS:
        for v in ("e2e", "fixed"):
            x = np.array([fl(r, f"g_MS_clamp_{c}_{v}") for r in pop])
            med[f"{c}_{v}"] = {"mean": float(np.nanmean(x)), "ratio": float(np.nanmean(x) / gms.mean())}
        x = np.array([fl(r, f"g_LS_clamp_{c}_e2e") for r in pop])
        med[f"{c}_LS_e2e"] = {"mean": float(np.nanmean(x)),
                              "ratio": float(np.nanmean(x) / np.nanmean([fl(r, 'g_LS') for r in pop]))}
    res["mediation"] = med
    res["mediation_component_e2e_stat"] = stat(pop, "g_MS_clamp_component_e2e", rng, cfg)
    fl_ = lambda k: float(np.nanmean([fl(r, k) for r in pop]))   # noqa: E731
    res["flips"] = {"student_top1_flip_T": fl_("flip_S_T_arm"), "student_top1_flip_P0": fl_("flip_S_P0_arm"),
                    "student_T_top1_is_r": fl_("flip_S_T_to_rnn"), "student_T_top1_edited": fl_("flip_S_T_to_edited"),
                    "carrier_T_top1_is_r": fl_("carrier_top1_T_is_r")}
    m = np.array([fl(r, "g_MT") for r in pop])
    res["flips"].update({"teacher_top1_flip_T": fl_("flip_T_T_arm"), "teacher_top1_flip_P0": fl_("flip_T_P0_arm"),
                         "teacher_T_top1_is_r": fl_("flip_T_T_to_rnn"),
                         "teacher_T_top1_edited": fl_("flip_T_T_to_edited")})
    res["teacher_state_response"] = {"mean_abs_gMT": float(np.abs(m).mean()),
                                     "frac_abs_gMT_gt_0.05": float((np.abs(m) > 0.05).mean()),
                                     "base_MT_mean": fl_("base_MT"),
                                     "frac_base_piT_i_gt_0.999": float(np.mean([fl(r, "base_piT_i") > 0.999
                                                                                for r in pop]))}
    spec = {}
    for q in range(cfg["bank_size"]):
        gE = np.array([fl(r, f"gE_m{q:02d}") for r in pop])
        spec[f"bank{q:02d}"] = {"gE_mean": float(gE.mean()), "gE_max_abs": float(np.abs(gE).max()),
                                "gE_exact_zero": float((gE == 0).mean()),
                                "gR_mean": fl_(f"gR_m{q:02d}"), "gC_mean": fl_(f"gC_m{q:02d}"),
                                "baseC_mean": fl_(f"baseC_m{q:02d}"), "basew_mean": fl_(f"basew_m{q:02d}")}
    res["spectra"] = spec
    ew = np.array([[fl(r, f"gE_m{q:02d}") for q in cfg["elementwise"]] for r in pop])
    res["elementwise_exact_zero_share"] = float((ew == 0).mean())
    res["dose"] = {"k_star_mean": fl_("k_star"), "k_mean": fl_("k"),
                   "k_star_hist": dict(collections.Counter(int(fl(r, "k_star")) for r in pop))}
    res["base"] = {k: fl_(k) for k in ("A_c", "w_c", "M_S", "base_C", "base_Vc_i", "load", "d_target", "d_rnn")}
    res["n"], res["n_inst"] = len(pop), len({r["instance_id"] for r in pop})
    return res


def md_block(name, b):
    L = [f"\n### {name}  (n = {b['n']} states / {b['n_inst']} instances)\n",
         "| readout | mean (states) | 95% CI (states) | > 1e-6 | p (state mean) |", "|---|---|---|---|---|"]
    for key, lab in READOUTS + DECOMP:
        s = b.get(key)
        if not s:
            continue
        L.append(f"| {lab} | {s['mean_state']:+.3f} | [{s['ci95_state_lo']:+.3f}, {s['ci95_state_hi']:+.3f}] | "
                 f"{s['pct_positive']:.0f}% | {s['p_signflip_state_mean']:.2g} |")
    f = b["flips"]
    L.append("\nTop-1 under T: student flips %.0f%% (to r %.0f%%, to an edited node %.1f%%); carrier's top-1 = r in "
             "%.0f%%." % (100 * f["student_top1_flip_T"], 100 * f["student_T_top1_is_r"],
                          100 * f["student_T_top1_edited"], 100 * f["carrier_T_top1_is_r"]))
    L.append("Teacher: top-1 flips %.1f%% under T, %.1f%% under P0 (to r %.1f%%, to an edited node %.1f%%); "
             "mean |g_MT| %.3f, |g_MT| > 0.05 in %.0f%%."
             % (100 * f["teacher_top1_flip_T"], 100 * f["teacher_top1_flip_P0"], 100 * f["teacher_T_top1_is_r"],
                100 * f["teacher_T_top1_edited"], b["teacher_state_response"]["mean_abs_gMT"],
                100 * b["teacher_state_response"]["frac_abs_gMT_gt_0.05"]))
    m = b["mediation"]
    L.append("Mediation (g_MS clamped / g_MS, e2e | fixed router): " + ", ".join(
        f"{c} {m[c + '_e2e']['ratio']:.3f} | {m[c + '_fixed']['ratio']:.3f}" for c in CLAMPS))
    L.append("Elementwise programs' raw-margin change exactly 0 in %.4f%% of (state, program) pairs." %
             (100 * b["elementwise_exact_zero_share"]))
    return "\n".join(L)


def table(cfg, name, spec, root):
    rows, labels = [], []
    for short, batch in spec["batches"].items():
        path = os.path.join(root, batch, "triplets.csv")
        labels.append(f"{short}={batch}/triplets.csv")
        for r in csv.DictReader(open(path)):
            r["_batch"] = short
            rows.append(r)
    rng = np.random.default_rng(cfg["analysis_seed"])
    prim = lambda r: fl(r, "A_c") >= 0.5 and fl(r, "M_S") >= 0.1   # noqa: E731
    pops = collections.OrderedDict()
    if spec["ladder"]:
        ms = sorted({int(float(r["m_over"])) for r in rows})
        for m in ms:
            pops[f"PRIMARY m={m}"] = [r for r in rows if prim(r) and int(float(r["m_over"])) == m]
        for m in ms:
            pops[f"theta>=0.25 m={m}"] = [r for r in rows if fl(r, "A_c") >= 0.25 and fl(r, "M_S") >= 0.1
                                          and int(float(r["m_over"])) == m]
    else:
        rows = [r for r in rows if int(float(r.get("m_over", 1))) == 1]
        pops["PRIMARY pooled"] = [r for r in rows if prim(r)]
        for short in sorted({r["_batch"] for r in rows}):
            pops[f"PRIMARY {short}"] = [r for r in rows if prim(r) and r["_batch"] == short]
        pops["w>=0.5 pooled"] = [r for r in rows if fl(r, "w_c") >= 0.5 and fl(r, "M_S") >= 0.1]
        pops["theta>=0.25 pooled"] = [r for r in rows if fl(r, "A_c") >= 0.25 and fl(r, "M_S") >= 0.1]
    res, md = {}, [f"# Local demand density (carrier: bank program {cfg['carrier']}); g = P0 - T\n"
                   f"Batches: {', '.join(labels)}"]
    for pname, pop in pops.items():
        if len(pop) < 5:
            res[pname] = {"n": len(pop)}
            md.append(f"\n### {pname}: n = {len(pop)} (too few)")
            continue
        res[pname] = block(pop, rng, cfg)
        md.append(md_block(pname, res[pname]))
        print(md[-1], flush=True)
    C.write_json(os.path.join(root, name + ".json"), res, indent=1)
    open(os.path.join(root, name + ".md"), "w").write("\n".join(md) + "\n")


def batch_config(cfg, batch):
    out = {k: v for k, v in cfg.items() if k not in ("batches", "tables")}
    out["overshoot_grid"] = cfg["batches"][batch]["overshoot_grid"]
    return out


def batch_fingerprint(cfg, batch, a):
    return C.experiment_fingerprint(NAME, PROTOCOL, ANALYSIS_KEYS, a, batch_config(cfg, batch))


def check_table_batches(cfg, spec, root, a):
    names = list(spec["batches"].values())
    fjs = [json.load(open(os.path.join(root, b, "funnel.json"))) for b in names]
    shared = C.common_fingerprint([fj.get("fingerprint") for fj in fjs], names, "batches of the table",
                                  a.allow_unverified)
    if shared is not None:
        C.check_fingerprint(shared, batch_fingerprint(cfg, names[0], a), "the batches of the table")
    for b, fj in zip(names, fjs):
        if fj.get("complete") is False:
            print(f"  NOTE: batch {b} is partial (instances {fj.get('instances')})")


def run(a, cfg):
    if (a.inst_lo is not None or a.inst_hi is not None or a.merge) and not a.batch:
        raise SystemExit("demand_density: --inst-lo / --inst-hi / --merge need --batch")
    if a.batch and a.batch not in cfg["batches"]:
        raise SystemExit(f"unknown batch {a.batch!r}; one of {list(cfg['batches'])}")
    batches = [a.batch] if a.batch else list(cfg["batches"])
    if not (a.analyze or a.merge):
        device = torch.device(a.device)
        torch.manual_seed(cfg["torch_seed"])
        student, bank, ck, bank_json = C.load_student(SETTING, a.student, device, bank=a.bank)
        C.bank_entries(bank_json)
        if len(bank) != cfg["bank_size"]:
            raise SystemExit(f"STOP: bank has {len(bank)} programs, expected {cfg['bank_size']}")
        if abs(bank.tau_h - cfg["tau_h"]) >= 1e-12:
            raise SystemExit(f"STOP: bank tau_h {bank.tau_h} != config {cfg['tau_h']}")
        teacher = C.load_teacher(SETTING, a.teacher_ckpt, device).model
        for batch in batches:
            bc = batch_config(cfg, batch)
            fp = batch_fingerprint(cfg, batch, a)
            print(f"[fingerprint] {batch}: {C.fingerprint_id(fp)}  {fp}")
            start, size = cfg["batches"][batch]["start"], cfg["batch_instances"]
            single = a.inst_lo is not None or a.inst_hi is not None
            if single:
                lo, hi = a.inst_lo if a.inst_lo is not None else start, a.inst_hi or start + size
                ranges = [(lo, hi)]
            else:
                ranges = [(s, s + cfg["shard_instances"]) for s in range(start, start + size, cfg["shard_instances"])]
            for lo, hi in ranges:
                sd = os.path.join(a.out, batch, "shards", f"shard_{lo}_{hi}")
                if not single and os.path.exists(os.path.join(sd, "funnel.json")):
                    done = json.load(open(os.path.join(sd, "funnel.json")))
                    C.check_fingerprint(done.get("fingerprint"), fp, sd, a.allow_unverified)
                    print(f"  {sd} done (same experiment), skipping")
                    continue
                res = run_shard(bc, teacher, bank, student, device, lo, hi)
                write_shard(res, sd, {"inst_lo": lo, "inst_hi": hi, "config": bc, "student_epoch": ck.get("epoch"),
                                      "fingerprint": fp})
                print(f"wrote {sd}: {len(res['rows'])} triplets, {len(res['facts'])} facts "
                      f"({res['elapsed_s']:.0f}s)", flush=True)
            if not single:
                merge_batch(os.path.join(a.out, batch), cfg, batch, fp, a.allow_unverified, a.allow_partial)
        if a.batch:
            return
    if a.merge:
        merge_batch(os.path.join(a.out, a.batch), cfg, a.batch, batch_fingerprint(cfg, a.batch, a),
                    a.allow_unverified, a.allow_partial)
        return
    for name, spec in cfg["tables"].items():
        if all(os.path.exists(os.path.join(a.out, b, "triplets.csv")) for b in spec["batches"].values()):
            check_table_batches(cfg, spec, a.out, a)
            table(cfg, name, spec, a.out)
        else:
            print(f"  {name}: batches missing under {a.out}, skipped")
