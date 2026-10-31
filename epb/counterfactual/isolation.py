import collections
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import common as C

NAME = "isolation"
SETTING = "lehd_tsp"
STUDENT = "lehd_tsp100"
CONFIG = os.path.join(C.HERE, "isolation.json")
EPS_LOGODDS = 1e-9
EPS_I = 1e-12


def add_args(ap):
    ap.add_argument("--heldout-cache", help="a .pt of the held-out instances + teacher tours: read if it exists, "
                                            "otherwise built and written there")
    ap.add_argument("--decompose", action="store_true",
                    help="also run the three-arm (B / T / P) decomposition: OUT/decomp.json, DECOMP.md, "
                         "triplets_BTP.pt")
    ap.add_argument("--save-triplets", action="store_true", help="also write OUT/triplets.pt (per-state readouts)")


def h03_instr(unselected_locs, first_loc, current_loc):
    B, K, _ = unselected_locs.shape
    device = unselected_locs.device
    dtype = unselected_locs.dtype
    eps = 1e-8
    if K == 1:
        z = torch.zeros(B, 1, device=device, dtype=dtype)
        return z, {"zero": z}

    deltas_cur = unselected_locs - current_loc.unsqueeze(1)
    deltas_start = unselected_locs - first_loc.unsqueeze(1)
    d_cur = torch.norm(deltas_cur, dim=2)
    d_first = torch.norm(deltas_start, dim=2)

    v_sc = current_loc - first_loc
    v_sc_norm = torch.norm(v_sc, dim=1, keepdim=True)
    v_sc_hat = v_sc / (v_sc_norm + eps)
    centroid = unselected_locs.mean(dim=1)
    v_ccent = centroid - current_loc
    v_ccent_hat = v_ccent / (torch.norm(v_ccent, dim=1, keepdim=True) + eps)
    use_centroid = (v_sc_norm < 1e-8).to(dtype)
    outward_hat = v_sc_hat * (1.0 - use_centroid) + v_ccent_hat * use_centroid
    back_hat = -outward_hat

    delta_hat = deltas_cur / (d_cur.unsqueeze(-1) + eps)

    k_val = float(K)
    clos_k = 1.0 / (1.0 + math.log1p(k_val))
    clos_k_t = torch.full((B, 1), float(clos_k), device=device, dtype=dtype)
    mean_d_cur = d_cur.mean(dim=1, keepdim=True)
    min_d_cur = d_cur.min(dim=1, keepdim=True).values
    nn_ratio = min_d_cur / (mean_d_cur + eps)
    clos_eff = clos_k_t * (0.2 + 0.8 * nn_ratio)

    diffs = unselected_locs.unsqueeze(2) - unselected_locs.unsqueeze(1)
    D = torch.norm(diffs, dim=3)
    eye = torch.eye(K, device=device, dtype=dtype).unsqueeze(0)
    big = torch.tensor(1e6, device=device, dtype=dtype)
    D_noself = D + eye * big

    M = D + d_first.unsqueeze(1) + eye * big
    min_M = M.min(dim=2).values
    L2 = d_cur + min_M

    nearest_dist = D_noself.min(dim=2).values
    nd_min = nearest_dist.min(dim=1, keepdim=True).values
    nd_max = nearest_dist.max(dim=1, keepdim=True).values
    outlier_nn = (nearest_dist - nd_min) / (nd_max - nd_min + eps)

    df_min = d_first.min(dim=1, keepdim=True).values
    df_max = d_first.max(dim=1, keepdim=True).values
    outlier_start = (d_first - df_min) / (df_max - df_min + eps)

    b = 0.3 * clos_eff
    aim_vec = (1.0 - b) * outward_hat + b * back_hat
    aim_hat = aim_vec / (torch.norm(aim_vec, dim=1, keepdim=True) + eps)
    cos_dir = (delta_hat * aim_hat.unsqueeze(1)).sum(dim=2)

    w_nn = 3.2
    s_nn = -w_nn * d_cur
    w_l2 = 3.0
    s_l2 = -w_l2 * clos_eff * L2
    w_out_nn = 2.0
    w_out_start = 0.8
    s_outlier_compiled = clos_eff * (w_out_nn * outlier_nn + w_out_start * outlier_start)
    s_outlier_nn = clos_eff * (w_out_nn * outlier_nn)
    s_outlier_start = clos_eff * (w_out_start * outlier_start)
    is_min_start = (d_first <= df_min + 1e-12).to(dtype)
    w_defer = 1.3
    s_defer = -w_defer * clos_eff * is_min_start
    w_dir = 0.35 * (1.0 - 0.6 * clos_eff)
    s_dir = w_dir * cos_dir

    parts = {"nn": s_nn, "dir": s_dir, "outlier_nn": s_outlier_nn, "outlier_start": s_outlier_start,
             "lookahead": s_l2, "defer": s_defer}
    score = s_nn + s_dir + s_outlier_compiled + s_l2 + s_defer
    return score, parts


def h05_instr(unselected_locs, first_loc, current_loc):
    B, K = unselected_locs.shape[:2]
    device = unselected_locs.device
    dtype = unselected_locs.dtype
    if K == 1:
        z = torch.zeros(B, 1, device=device, dtype=dtype)
        return z, {"zero": z}

    eps = 1e-8
    den_eps = 1e-6

    centroid = unselected_locs.mean(dim=1)
    v_cand = unselected_locs - centroid[:, None, :]
    r_cand = torch.norm(v_cand, dim=2)

    r_cur_vec = current_loc - centroid
    r_cur = torch.norm(r_cur_vec, dim=1)
    u_cur = r_cur_vec / (r_cur.clamp_min(eps)).unsqueeze(1)
    t_cur = torch.stack([-u_cur[:, 1], u_cur[:, 0]], dim=1)

    step_vec = unselected_locs - current_loc[:, None, :]
    step_len = torch.norm(step_vec, dim=2)
    step_hat = step_vec / (step_len.clamp_min(eps)).unsqueeze(2)

    step_len_mean = step_len.mean(dim=1, keepdim=True)
    norm_step = step_len / (step_len_mean.clamp_min(den_eps))
    min_step = step_len.min(dim=1, keepdim=True).values
    over_min = (step_len - min_step) / (step_len_mean.clamp_min(den_eps))

    step_spread = step_len.std(dim=1, keepdim=True, unbiased=False) / (step_len_mean.clamp_min(den_eps))
    g_compact = 1.0 / (1.0 + 5.0 * step_spread)
    gK_small = torch.tensor(1.0 / (K + 1.0), device=device, dtype=dtype)
    g_end = (g_compact.squeeze(1)) + gK_small
    g_end = g_end.clamp(0.0, 1.0)

    dist_to_start = torch.norm(unselected_locs - first_loc[:, None, :], dim=2)
    mean_to_start = dist_to_start.mean(dim=1, keepdim=False)
    mean_radial = r_cand.mean(dim=1, keepdim=False)
    denom_close = (mean_to_start + mean_radial).clamp_min(den_eps)
    x_close = (mean_radial - mean_to_start) / denom_close
    f_close = torch.sigmoid(6.0 * x_close)

    tan_signed = (step_hat * t_cur[:, None, :]).sum(dim=2)

    radial_gain = r_cand - r_cur.unsqueeze(1)
    gain_unit = radial_gain / (step_len + den_eps)
    outside_flag = (r_cur > r_cand.mean(dim=1)).to(dtype)
    w_outward_factor = (1.0 - g_end) * (1.0 - 0.5 * outside_flag)
    s_rad = (- (2.0 + 1.5 * g_end).unsqueeze(1) * torch.relu(-gain_unit)
             + (1.5 * w_outward_factor).unsqueeze(1) * torch.relu(gain_unit))

    outward_unit = torch.relu(gain_unit)
    tan_gate = 1.0 - 0.6 * outward_unit * outside_flag.unsqueeze(1)
    tan_gate = tan_gate.clamp(0.2, 1.0)
    w_tan_eff = (2.8 * (1.0 - 0.6 * g_end)).unsqueeze(1)
    s_tan = w_tan_eff * (tan_signed * tan_gate)

    w_len_eff = (8.5 + 6.0 * g_end).unsqueeze(1)
    w_overmin = (6.0 * g_end).unsqueeze(1)
    s_len = (-w_len_eff * norm_step) + (-w_overmin * over_min)

    r_mean = r_cand.mean(dim=1, keepdim=True)
    r_std = r_cand.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    r_z = (r_cand - r_mean) / r_std
    w_periph = (0.8 + 2.2 * g_end).unsqueeze(1)
    s_periph = w_periph * r_z

    diff = unselected_locs[:, :, None, :] - unselected_locs[:, None, :, :]
    pair_d = torch.norm(diff, dim=3)
    pair_d = pair_d + torch.eye(K, device=device, dtype=dtype).unsqueeze(0) * 1e6
    nn_dist = pair_d.min(dim=2).values
    nn_mean = nn_dist.mean(dim=1, keepdim=True)
    nn_std = nn_dist.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    nn_z = (nn_dist - nn_mean) / nn_std
    w_iso = (2.0 * g_end).unsqueeze(1)
    s_iso = w_iso * nn_z

    d_mean = dist_to_start.mean(dim=1, keepdim=True)
    d_std = dist_to_start.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    d_start_z = (dist_to_start - d_mean) / d_std
    w_far = (1.5 * gK_small * (1.0 - f_close)).unsqueeze(1)
    s_far = w_far * d_start_z

    gap_vec = first_loc - current_loc
    gap_hat = gap_vec / (torch.norm(gap_vec, dim=1, keepdim=True).clamp_min(eps))
    align_to_start = (step_hat * gap_hat[:, None, :]).sum(dim=2)
    w_align = (2.0 + 2.0 * g_end) * f_close
    s_align = w_align.unsqueeze(1) * align_to_start

    s_out_closer_pen = -(1.5 * g_end * f_close).unsqueeze(1) * outward_unit

    parts = {"len": s_len, "tan": s_tan, "rad": s_rad, "periph": s_periph,
             "iso": s_iso, "far": s_far, "align": s_align, "out_pen": s_out_closer_pen}
    score = s_len + s_tan + s_rad + s_periph + s_iso + s_far + s_align + s_out_closer_pen
    return score, parts


def h06_instr(unselected_locs, first_loc, current_loc):
    B, K, _ = unselected_locs.shape
    device = unselected_locs.device
    if K == 1:
        z = torch.zeros(B, 1, device=device)
        return z, {"zero": z}

    Cn = unselected_locs
    eps = 1e-6

    dist_current = torch.cdist(current_loc.unsqueeze(1), Cn).squeeze(1)
    dist_start = torch.cdist(first_loc.unsqueeze(1), Cn).squeeze(1)

    pair = torch.cdist(Cn, Cn)
    eye = torch.eye(K, device=device, dtype=torch.bool).unsqueeze(0)
    pair_masked = pair.masked_fill(eye, float("inf"))
    partner_dists = torch.cat([pair_masked, dist_start.unsqueeze(-1)], dim=-1)

    min1_vals, min1_idx = partner_dists.min(dim=-1)
    one_hot_min = F.one_hot(min1_idx, num_classes=partner_dists.size(-1)).to(torch.bool)
    masked2 = partner_dists.masked_fill(one_hot_min, float("inf"))
    min2_vals, _ = masked2.min(dim=-1)

    tau = min1_vals.mean(dim=1, keepdim=True) + eps

    dc_scaled = dist_current / tau
    ds_scaled = dist_start / tau
    iso1 = min1_vals / tau
    iso_gap = (min2_vals - min1_vals) / tau

    centroid = Cn.mean(dim=1, keepdim=True)
    radial = torch.norm(Cn - centroid, dim=-1)
    radial_scale = radial.mean(dim=1, keepdim=True) + eps
    hullness = radial / radial_scale

    heading = current_loc - first_loc
    h_norm = torch.clamp(torch.norm(heading, dim=-1, keepdim=True), min=eps)
    h_unit = heading / h_norm

    vecs = Cn - current_loc.unsqueeze(1)
    vecs_norm = torch.clamp(torch.norm(vecs, dim=-1, keepdim=True), min=eps)
    dirs = vecs / vecs_norm
    cos_fwd = (dirs * h_unit.unsqueeze(1)).sum(dim=-1)
    fwd_pos = F.relu(cos_fwd)
    back_pos = F.relu(-cos_fwd)

    short_bonus = torch.clamp((tau - dist_current) / tau, min=0.0, max=1.0)
    short_gate = 0.35 + 0.65 * fwd_pos

    gate_relaxed = 0.4 + 0.6 * torch.sigmoid(3.0 * (min1_vals - 0.8 * dist_current) / tau)

    w_dc, w_iso1, w_gap = 4.2, 2.1, 1.4
    w_hull, w_fwd, w_back, w_short, w_away = 0.7, 1.0, 1.2, 0.8, 0.25

    parts = {
        "dc":    -w_dc * dc_scaled,
        "iso1":   w_iso1 * iso1,
        "gap":    w_gap * gate_relaxed * torch.clamp(iso_gap, min=0.0, max=3.0),
        "hull":   w_hull * hullness,
        "fwd":    w_fwd * fwd_pos,
        "back":  -w_back * back_pos,
        "short":  w_short * short_gate * short_bonus,
        "away":   w_away * ds_scaled,
    }
    score = (parts["dc"] + parts["iso1"] + parts["gap"] + parts["hull"]
             + parts["fwd"] + parts["back"] + parts["short"] + parts["away"])
    return score, parts


INSTRUMENTED = {"h03": h03_instr, "h05": h05_instr, "h06": h06_instr}


class Family:

    def __init__(self, cfg):
        fam = cfg["family"]
        self.pos = [int(m) for m in fam["slots"]]
        self.fn = {int(m): INSTRUMENTED[name] for m, name in fam["instrumented"].items()}
        self.iso_part = {int(m): p for m, p in fam["iso_part"].items()}
        self.excluded = {k: (int(m), p) for k, (m, p) in fam["excluded_terms"].items()}
        if sorted(self.fn) != sorted(self.pos) or sorted(self.iso_part) != sorted(self.pos):
            raise SystemExit("STOP: config family: every slot needs an instrumented copy and an isolation part")

    def parts(self, bank, U, S_, C_):
        out = {}
        for m in self.pos:
            score, parts = self.fn[m](U, S_, C_)
            if not torch.equal(score.float(), bank.programs[m].fn(U, S_, C_).float()):
                raise SystemExit(f"STOP: the instrumented copy of slot {m} does not reproduce the bank program "
                                 f"(the copies apply to the bank this test was written for)")
            out[m] = parts
        return out


def load(cfg, a, device, upto):
    student, bank, ck, bank_json = C.load_student(SETTING, a.student, device, bank=a.bank, tau_h=cfg["tau_h"])
    C.bank_entries(bank_json)
    if float(student.routing_temperature) != float(cfg["tau_r"]):
        raise SystemExit(f"STOP: router tau_r {student.routing_temperature} != config {cfg['tau_r']}")
    teacher = C.load_teacher(SETTING, a.teacher_ckpt, device, nco_root=a.nco_root).model
    held = C.lehd_heldout(teacher, device, upto, a.heldout_cache, fp=C.heldout_fingerprint(a.teacher_ckpt, upto),
                          allow_unverified=a.allow_unverified)
    print(f"  bank    : {bank_json}  M={len(bank)}  tau_h={bank.tau_h}")
    print(f"  student : epoch={ck.get('epoch')}  tau_r={student.routing_temperature}")
    print(f"  heldout : {held['instance_locs'].shape[0]} instances")
    return bank, student, teacher, held


def step_list(cfg):
    N = cfg["problem_size"]
    steps = [N - k for k in range(cfg["n_remaining_min"], cfg["n_remaining_max"] + 1)]
    steps = [s for s in steps if s >= 2]
    g = torch.Generator().manual_seed(cfg["seed"])
    sel = torch.randperm(len(steps), generator=g)[:cfg["states_per_instance"]].tolist()
    return sorted(steps[k] for k in sel)


def geom_features(X, s, c):
    K = X.shape[0]
    ar = torch.arange(K)
    d_cur = torch.norm(X - c[None, :], dim=1)
    d_start = torch.norm(X - s[None, :], dim=1)
    D = torch.cdist(X[None], X[None])[0]
    Dm = D.clone()
    Dm.fill_diagonal_(float("inf"))
    v1, j1 = Dm.min(dim=1)
    D2 = Dm.clone()
    D2[ar, j1] = float("inf")
    v2, j2 = D2.min(dim=1)
    return {"d_cur": d_cur, "d_start": d_start, "d1": v1, "d2": v2, "j1": j1, "j2": j2}


@torch.no_grad()
def student_readout(student, bank, U, S_, C_, unsel_idx, N):
    batch = C.lehd_sub_batch(U, S_, C_, unsel_idx, N)
    V_full, V_K, _ = bank.compute_probs(batch)
    y_hat, w, scores = student(batch, V_full, V_K, return_scores=True)
    return {"V_K": V_K, "w": w, "scores": scores, "pi_S_K": y_hat.gather(1, unsel_idx)}


def contributions(w, V_K, i, r):
    return w * (V_K[:, i] - V_K[:, r])


def isolation_I(G):
    denom = float(G["d1"].median()) + EPS_I
    return G["d1"] / denom, denom


def R_F_from_logits(z, pos):
    notpos = [m for m in range(z.shape[0]) if m not in pos]
    return float(torch.logsumexp(z[pos], 0) - torch.logsumexp(z[notpos], 0))


def logodds_eps(p, i, r, eps=EPS_LOGODDS):
    return math.log(float(p[i]) + eps) - math.log(float(p[r]) + eps)


def build(cfg, X, s, c, G, i, r, rho):
    q = int(G["j1"][i])
    if q == r:
        return None, "V1_q_is_rNN"
    R = float(torch.norm(X[i] - c))
    rq = float(torch.norm(X[q] - c))
    d1 = float(G["d1"][i])
    if R < 1e-9 or rq < 1e-9:
        return None, "V2_degenerate_radius"
    d_new = (1.0 - rho) * d1
    cosb = (R * R + rq * rq - d_new * d_new) / (2.0 * R * rq)
    if not (-1.0 <= cosb <= 1.0):
        return None, "V3_unreachable_on_circle"
    beta = C.wrap_angle(C.polar_angle(X[i] - c) - C.polar_angle(X[q] - c))
    beta_new = math.acos(cosb) * (1.0 if beta >= 0 else -1.0)
    delta = beta - beta_new
    iT = c + C.rotate2(X[i] - c, -delta)
    iP = c + C.rotate2(X[i] - c, +delta)
    con = {"q": q, "iT": iT, "iP": iP, "delta": float(delta), "d1_base": d1}
    for arm in ("T", "P"):
        x = con["iT"] if arm == "T" else con["iP"]
        if not (cfg["coord_lo"] <= float(x.min()) and float(x.max()) <= cfg["coord_hi"]):
            return None, f"V4_out_of_bounds_{arm}"
        Y = X.clone()
        Y[i] = x
        Ge = geom_features(Y, s, c)
        if int(Ge["j1"][i]) != q:
            return None, f"V5_q_no_longer_nearest_{arm}"
        if int(Ge["d_cur"].argmin()) != r:
            return None, f"V6_rNN_identity_{arm}"
        if i in (int(G["j1"][r]), int(G["j2"][r])):
            return None, "V7_target_is_rNN_partner"
        if float(torch.norm(x - X[r])) <= float(G["d2"][r]) + cfg["partner_margin"]:
            return None, f"V7_perturbs_reference_partners_{arm}"
        d = torch.norm(Y - x[None, :], dim=1)
        d[i] = float("inf")
        if float(d.min()) < cfg["min_node_sep"]:
            return None, f"V8_node_collision_{arm}"
    return con, None


def apply_at(X, i, con, arm):
    Y = X.clone()
    Y[i] = con["iT"] if arm == "T" else con["iP"]
    return Y


@torch.no_grad()
def census(cfg, fam, bank, student, teacher, held, device, lo0, hi0):
    N = cfg["problem_size"]
    steps = step_list(cfg)
    recs, t0 = [], time.perf_counter()
    for lo in range(lo0, hi0, cfg["chunk_instances"]):
        ids = list(range(lo, min(lo + cfg["chunk_instances"], hi0)))
        for step in steps:
            st = C.lehd_prefix_states(held, ids, step, device)
            U, S_, C_ = st["unsel_locs"], st["first_loc"], st["current_loc"]
            ev = student_readout(student, bank, U, S_, C_, st["unsel_idx"], N)
            t_probs, t_unsel = C.lehd_teacher_probs(teacher, st["locs"], st["prefix"])
            assert torch.equal(t_unsel, st["unsel_idx"])
            d_cur = torch.norm(U - C_[:, None, :], dim=-1)
            rnn = d_cur.argmin(dim=1)
            tgt = ev["pi_S_K"].argmax(dim=1)
            keep = ((t_probs.argmax(dim=1) == tgt) & (tgt != rnn)).nonzero(as_tuple=True)[0].tolist()
            for b in keep:
                X = U[b].double().cpu()
                G = geom_features(X, S_[b].double().cpu(), C_[b].double().cpu())
                i = int(tgt[b]); r = int(G["d_cur"].argmin())
                assert r == int(rnn[b]) and i != r
                Iv, denom = isolation_I(G)
                pS = ev["pi_S_K"][b].double().cpu()
                Cm = contributions(ev["w"][b].double().cpu(), ev["V_K"][b].double().cpu(), i, r)
                M = float(pS[i] - pS[r])
                recs.append({"inst": int(ids[b]), "step": int(step), "K": int(st["K"]),
                             "i": i, "r": r, "M_full": M, "I_target": float(Iv[i]),
                             "I_denom": float(denom),
                             "attr_iso": (float(Cm[fam.pos].sum()) / M if abs(M) > 1e-12 else float("nan"))})
        print(f"  [census] instances {lo}..{ids[-1]}  kept={len(recs)}  "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
    return recs


def population(recs, cfg):
    base = [r for r in recs if r["K"] >= cfg["K_primary"] and r["I_target"] >= cfg["I_threshold"]]
    return [r for r in base if r["M_full"] > 0 and np.isfinite(r["attr_iso"])
            and r["attr_iso"] >= cfg["attr_share_min"]]


@torch.no_grad()
def run_arms(cfg, fam, bank, student, teacher, held, device, pop, hi0, arms):
    N = cfg["problem_size"]
    pos = fam.pos
    by_group = collections.defaultdict(list)
    for r in pop:
        by_group[(r["inst"] // cfg["chunk_instances"], r["step"])].append(r)
    recs, rej, t0 = [], collections.Counter(), time.time()
    for gi, ((chunk, step), grp) in enumerate(sorted(by_group.items())):
        ids = list(range(chunk * cfg["chunk_instances"],
                         min((chunk + 1) * cfg["chunk_instances"], hi0)))
        st = C.lehd_prefix_states(held, ids, step, device)
        row_of = {int(v): k for k, v in enumerate(ids)}
        geo, jobs = {}, []
        for r in sorted(grp, key=lambda r: (r["inst"], r["i"])):
            b = row_of[r["inst"]]
            X = st["unsel_locs"][b].double().cpu()
            s_ = st["first_loc"][b].double().cpu()
            c_ = st["current_loc"][b].double().cpu()
            G = geom_features(X, s_, c_)
            geo[r["inst"]] = (X, s_, c_, G, b)
            con, why = build(cfg, X, s_, c_, G, r["i"], r["r"], cfg["rho"])
            if con is None:
                rej[why] += 1
                continue
            for arm in arms:
                jobs.append((r, b, con, arm, X.clone() if arm == "B" else apply_at(X, r["i"], con, arm)))
        if not jobs:
            continue
        U = torch.stack([j[4] for j in jobs]).float().to(device)
        S_ = torch.stack([geo[j[0]["inst"]][1] for j in jobs]).float().to(device)
        C_ = torch.stack([geo[j[0]["inst"]][2] for j in jobs]).float().to(device)
        IDX = torch.stack([st["unsel_idx"][j[1]] for j in jobs])
        LOCS = torch.stack([st["locs"][j[1]] for j in jobs]).clone()
        LOCS.scatter_(1, IDX.unsqueeze(-1).expand(-1, IDX.shape[1], 2), U)
        PRE = torch.stack([st["prefix"][j[1]] for j in jobs])
        parts = fam.parts(bank, U, S_, C_)
        ev = student_readout(student, bank, U, S_, C_, IDX, N)
        tp, tu = C.lehd_teacher_probs(teacher, LOCS, PRE)
        assert torch.equal(tu, IDX)
        pack = collections.defaultdict(dict)
        for k, (r, b, con, arm, Xe) in enumerate(jobs):
            i, rr = r["i"], r["r"]
            wv = ev["w"][k].double().cpu(); zv = ev["scores"][k].double().cpu()
            Vv = ev["V_K"][k].double().cpu(); pS = ev["pi_S_K"][k].double().cpu()
            pT = tp[k].double().cpu()
            Cm = contributions(wv, Vv, i, rr)
            d = {"E": 0.0, "R": R_F_from_logits(zv, pos), "C": float(Cm[pos].sum()),
                 "L_S": logodds_eps(pS, i, rr), "L_T": logodds_eps(pT, i, rr),
                 "M_S": float(pS[i] - pS[rr]), "M_T": float(pT[i] - pT[rr])}
            for m in pos:
                si = parts[m][fam.iso_part[m]][k].double().cpu()
                d[f"iso{m}"] = float(si[i] - si[rr])
                d["E"] += d[f"iso{m}"]
            for name, (m, part) in fam.excluded.items():
                p = parts[m][part][k].double().cpu()
                d[name] = float(p[i] - p[rr])
            Ge = geom_features(Xe, geo[r["inst"]][1], geo[r["inst"]][2])
            d["d1"] = float(Ge["d1"][i]); d["d_start"] = float(Ge["d_start"][i])
            pack[(r["inst"], r["step"])][arm] = d
            pack[(r["inst"], r["step"])]["_r"] = r
        for (inst, step_), got in pack.items():
            if all(arm in got for arm in arms):
                recs.append({"inst": inst, "step": step_, "i": got["_r"]["i"], "r": got["_r"]["r"],
                             **{arm: got[arm] for arm in arms}})
        if gi % 40 == 0:
            print(f"    group {gi+1}/{len(by_group)}  triplets {len(recs)}  {time.time()-t0:.0f}s", flush=True)
    return recs, rej


FIVE = [("g_E_iso_pos", "E", "1. isolation-family score (analytic)"),
        ("g_route", "R", "2. family relative routing log-weight"),
        ("g_C_iso", "C", "3. isolation-family contribution"),
        ("g_student_logprob", "L_S", "4. student relative log-odds"),
        ("g_teacher_logprob", "L_T", "5. teacher relative log-odds")]
TWO_ARM_KEYS = {"E": "E_iso_pos", "R": "R_F", "C": "C_iso", "L_S": "lo_S", "L_T": "lo_T", "M_S": "M", "M_T": "tM"}
CHANNELS = [("E", "1. carrier score E_S (analytic, manipulation check)"),
            ("R", "2. relative routing log-weight R_S"),
            ("C", "3. contribution C_S"),
            ("L_S", "4. student log-odds L_S"),
            ("L_T", "5. teacher log-odds L_T")]
EXCLUDED_LABELS = {"H06_gap": "H06 partner-gap term (excluded)", "H03_start": "H03 start-distance term (excluded)"}
CONTRASTS = [("down", "B", "T"), ("up", "P", "B"), ("g", "P", "T")]


def chan(v, ins, cfg):
    v = np.asarray(v, float)
    lo, hi = C.bootstrap_ci(v, ins, "isolation", cfg["bootstrap_B"], seed=cfg["stats_seed"])
    return {"mean_state": float(v.mean()), "mean_instance": C.instance_mean(v, ins, "isolation"),
            "median": float(np.median(v)), "ci95_state_lo": lo, "ci95_state_hi": hi,
            "frac_positive": float((v > 1e-12).mean()),
            "p_signflip_instance_mean": C.signflip_p(v, ins, "isolation", cfg["perm_B"], seed=cfg["stats_seed"]),
            "n_states": int(v.size), "n_instances": int(len(set(np.asarray(ins).tolist())))}


def five_values(trips, cfg):
    ins = np.array([r["inst"] for r in trips])
    out = {"rho": cfg["rho"], "n_states": len(trips), "n_instances": int(len(set(ins.tolist()))), "rows": {}}
    print(f"\n  rho = {cfg['rho']}   n = {len(trips)} triplets / {out['n_instances']} instances   (g = P - T)")
    print(f"  {'channel':<42}{'mean':<14}{'inst-mean':<14}{'95% CI (states)':<24}{'dir':<6}p (instance means)")
    for key, f, label in FIVE:
        g = np.array([r["P"][f] - r["T"][f] for r in trips])
        v = out["rows"][key] = {"label": label, **chan(g, ins, cfg)}
        print(f"  {label:<42}{v['mean_state']:+.3f}{'':<8}{v['mean_instance']:+.3f}{'':<8}"
              f"[{v['ci95_state_lo']:.3f}, {v['ci95_state_hi']:.3f}]{'':<8}{100*v['frac_positive']:.0f}%"
              f"{'':<3}{v['p_signflip_instance_mean']:.2g}")
    return out


def decompose(trips, cfg, keys):
    ins = np.array([t["inst"] for t in trips])
    out = {"n_states": len(trips), "n_instances": int(len(set(ins.tolist()))), "rows": {}}
    for key, label in keys:
        row = {"label": label}
        for name, hi, lo in CONTRASTS:
            row[name] = chan(np.array([t[hi][key] - t[lo][key] for t in trips]), ins, cfg)
        row["arm_mean"] = {arm: float(np.mean([t[arm][key] for t in trips])) for arm in ("B", "T", "P")}
        out["rows"][key] = row
    return out


def _fmt(c):
    return (f"{c['mean_state']:+.3f} [{c['ci95_state_lo']:+.3f}, {c['ci95_state_hi']:+.3f}]  "
            f"dir {100*c['frac_positive']:.0f}%  inst-mean {c['mean_instance']:+.3f} "
            f"p={c['p_signflip_instance_mean']:.2g}")


def report(res, title):
    lines = [f"## {title}   n = {res['n_states']} states / {res['n_instances']} instances", "",
             "Each cell: mean over states [bootstrap 95% CI of the state mean], share > 0, mean of the instance means, "
             "sign-flip p of the instance mean.", "",
             "| channel | Delta_down = B - T | Delta_up = P - B | g = P - T |", "|---|---|---|---|"]
    for key, row in res["rows"].items():
        lines.append(f"| {row['label']} | {_fmt(row['down'])} | {_fmt(row['up'])} | {_fmt(row['g'])} |")
    lines.append("")
    lines.append("arm means (B / T / P): " + "; ".join(
        f"{k}: {r['arm_mean']['B']:+.3f} / {r['arm_mean']['T']:+.3f} / {r['arm_mean']['P']:+.3f}"
        for k, r in res["rows"].items()))
    return "\n".join(lines)


def run(a, cfg):
    lo0 = a.inst_lo if a.inst_lo is not None else cfg["conf_lo"]
    hi0 = a.inst_hi if a.inst_hi is not None else cfg["conf_hi"]
    os.makedirs(a.out, exist_ok=True)
    device = a.device
    fam = Family(cfg)
    bank, student, teacher, held = load(cfg, a, device, max(hi0, cfg["conf_hi"]))
    recs = census(cfg, fam, bank, student, teacher, held, device, lo0, hi0)
    pop = population(recs, cfg)
    print(f"  census: {len(recs)} agree&non-NN states -> population (attr_iso>={cfg['attr_share_min']}, "
          f"K>={cfg['K_primary']}, I>={cfg['I_threshold']}): {len(pop)}")

    trips, rej = run_arms(cfg, fam, bank, student, teacher, held, device, pop, hi0, ("T", "P"))
    print(f"  constructed at rho={cfg['rho']}: {len(trips)} / {len(pop)}   rejections {dict(rej.most_common(6))}")
    out = five_values(trips, cfg)
    out["census_n"] = len(recs); out["population_n"] = len(pop); out["rejections"] = dict(rej)
    out["instances"] = [lo0, hi0]
    C.write_json(os.path.join(a.out, "five_values.json"), out)
    if a.save_triplets:
        torch.save([{**{k: t[k] for k in ("inst", "step", "i", "r")},
                     **{arm: {name: t[arm][k] for k, name in TWO_ARM_KEYS.items()} for arm in ("T", "P")}}
                    for t in trips], os.path.join(a.out, "triplets.pt"))

    if a.decompose:
        trips3, rej3 = run_arms(cfg, fam, bank, student, teacher, held, device, pop, hi0, ("B", "T", "P"))
        print(f"  constructed {len(trips3)} / {len(pop)}  rejections {dict(rej3.most_common(6))}")
        keep = ("E", "R", "C", "L_S", "L_T", "M_S", "M_T", *[f"iso{m}" for m in fam.pos], *fam.excluded, "d1",
                "d_start")
        trips3 = [{**{k: t[k] for k in ("inst", "step", "i", "r")},
                   **{arm: {k: t[arm][k] for k in keep} for arm in ("B", "T", "P")}} for t in trips3]
        torch.save(trips3, os.path.join(a.out, "triplets_BTP.pt"))
        res = decompose(trips3, cfg, CHANNELS + [(k, EXCLUDED_LABELS.get(k, k)) for k in fam.excluded])
        res["rejections"] = dict(rej3)
        res["geom"] = {arm: {"d1": float(np.mean([t[arm]["d1"] for t in trips3])),
                             "d_start": float(np.mean([t[arm]["d_start"] for t in trips3]))}
                       for arm in ("B", "T", "P")}
        md = report(res, f"B/T/P, rho = {cfg['rho']}, population of the five-value run")
        md += "\n\nmean target d1 / d_start per arm: " + json.dumps(res["geom"])
        print(md)
        C.write_json(os.path.join(a.out, "decomp.json"), res)
        open(os.path.join(a.out, "DECOMP.md"), "w").write(md + "\n")
