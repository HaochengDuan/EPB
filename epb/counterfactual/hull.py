import collections
import math
import os

import torch
import torch.nn.functional as F

from ..settings.pomo_tsp import build_td, decision_state
from . import common as C

NAME = "hull"
SETTING = "pomo_tsp"
STUDENT = "pomo_tsp100"
CONFIG = os.path.join(C.HERE, "hull.json")
PROTOCOL = 1
ANALYSIS_KEYS = ("instances", "shard_size", "stop_on_primary", "target_primary", "max_valid", "n_boot", "n_perm")

H02_SEMANTICS = ("a centroid-based proxy for convex-hull peripherality combined with "
                 "outward directional continuity")
H02_CODE_FRAGMENTS = [
    "hull_bonus = dist_from_centroid / max_dist_centroid",
    "max_dist_centroid = (dist_from_centroid * unvisited_float).max(dim=-1).values",
    "dir_outward = cur_loc - centroid",
    "early   = (n_remaining > 50.0).float()",
    "1.5 * hull_bonus",
    "- 4.0 * dist_to_cur",
    "+ 0.5 * continuity",
]
FIVE = ["E_H02", "R_H02", "C_H02", "L_student", "L_teacher"]
PRED = {k: ">0" for k in FIVE}


def add_args(ap):
    ap.add_argument("--merge", action="store_true",
                    help="only merge OUT/shards (instance order, stop rule) into OUT/five_values.json")
    ap.add_argument("--shard-size", type=int, help="instances per shard of a full run (default: config)")


def load(cfg, a, device):
    student, bank, ck, bank_json = C.load_student(SETTING, a.student, device, bank=a.bank)
    hs = C.bank_entries(bank_json)
    M = cfg["bank_size"]
    if len(hs) != M or len(bank) != M:
        raise SystemExit(f"STOP: bank size {len(hs)} != {M} ({bank_json})")
    m2 = cfg["H02_index"]
    missing = [f for f in H02_CODE_FRAGMENTS if f not in hs[m2]["code"]]
    if missing:
        raise SystemExit("STOP: bank slot %d does not match the assumed H02 early-stage program.\n"
                         "  missing source fragments: %s" % (m2, missing))
    ca = ck["args"]
    if ca["num_heuristics"] != M:
        raise SystemExit(f"STOP: checkpoint num_heuristics={ca['num_heuristics']} != {M}")
    if ca["problem_size"] != cfg["problem_size"]:
        raise SystemExit(f"STOP: checkpoint problem_size={ca['problem_size']} != {cfg['problem_size']}")
    teacher = C.load_teacher(SETTING, a.teacher_ckpt, device)
    print(f"[bank ] {bank_json}  M={M}  tau_h={bank.tau_h}")
    print(f"[ckpt ] epoch={ck.get('epoch')}  tau_r={student.routing_temperature}  embed_dim={ca['embed_dim']}  "
          f"layers={ca['num_encoder_layers']}")
    print(f"[H02  ] slot {m2}  ({H02_SEMANTICS})")
    return teacher, bank, student


def h02_semantics(locs, current, mask):
    locs = locs.double()
    Uf = mask.double()
    mu = (locs * Uf.unsqueeze(-1)).sum(0) / Uf.sum().clamp(min=1.0)
    d_cen = torch.norm(locs - mu, dim=-1)
    max_d = (d_cen * Uf).max()
    hull_bonus = d_cen / max_d.clamp(min=1e-6)
    c = locs[current]
    outward = c - mu
    u_out = outward / torch.norm(outward).clamp(min=1e-6)
    dvec = locs - c
    dhat = dvec / torch.norm(dvec, dim=-1, keepdim=True).clamp(min=1e-6)
    continuity = (dhat * u_out).sum(-1)
    d_cur = torch.norm(locs - c, dim=-1)
    X2 = 1.5 * hull_bonus + 0.5 * continuity
    dm = d_cen * Uf
    anchor = int(dm.argmax())
    dm2 = dm.clone(); dm2[anchor] = -1.0
    return {"mu": mu, "u_out": u_out, "d_cen": d_cen, "max_d": max_d, "anchor": anchor,
            "anchor_second": float(dm2.max()), "hull_bonus": hull_bonus, "continuity": continuity,
            "d_cur": d_cur, "X2": X2, "z2_early": X2 - 4.0 * d_cur}


def cross_2d(u, v):
    return u[0] * v[1] - u[1] * v[0]


def signed_angle(v, u_out):
    vhat = v / torch.norm(v).clamp(min=1e-12)
    return math.atan2(float(cross_2d(u_out, vhat)), float(torch.dot(vhat, u_out)))


def mirror_about(v, u_out):
    return 2.0 * torch.dot(v, u_out) * u_out - v


def rotate(v, theta):
    ct, st = math.cos(theta), math.sin(theta)
    R = torch.tensor([[ct, -st], [st, ct]], dtype=v.dtype, device=v.device)
    return R @ v


@torch.no_grad()
def eval_state(bank, student, teacher, locs, current, visited, first, step_i, step_max, device,
               need_teacher=True):
    N = locs.shape[0]
    locs_b = locs.unsqueeze(0)
    cur_b = torch.tensor([current], device=device, dtype=torch.long)
    first_b = torch.tensor([first], device=device, dtype=torch.long)
    vis_bool = visited.unsqueeze(0)
    mask_b = ~vis_bool
    step_norm = step_i / max(step_max, 1)

    raw = torch.stack([p.fn(locs_b, cur_b, mask_b, vis_bool.float(),
                            torch.tensor([step_norm], device=device), first_b).float()[0]
                       for p in bank.programs])

    batch = decision_state(locs_b, vis_bool, cur_b, first_b, step_i + 1)
    V = bank.compute_probs(batch)[0]
    td = build_td(batch, device)
    piS, w, scores = student(td, V, return_scores=True)

    logV = F.log_softmax((raw / bank.tau_h).masked_fill(~mask_b, float("-inf")), dim=-1)
    out = {"raw": raw, "V": V[0], "w": w[0], "scores": scores[0], "piS": piS[0], "logV": logV,
           "step_norm": step_norm}
    if need_teacher:
        teacher.begin(locs_b, first_b.unsqueeze(1))
        probsT, logitsT = teacher.decode(vis_bool.view(1, 1, N), cur_b.unsqueeze(1))
        out["piT"] = probsT[0, 0]
        out["logitsT"] = logitsT[0, 0]
    return out


@torch.no_grad()
def natural_states(bank, student, teacher, locs, first, cfg, device):
    N = locs.shape[0]
    order = C.pomo_tsp_greedy_tour(teacher, locs, first)
    lo, hi = cfg["n_remaining_min"], cfg["n_remaining_max"]
    snaps = []
    visited = torch.zeros(N, dtype=torch.bool, device=device)
    visited[first] = True
    for step in range(1, N):
        step_i = step - 1
        current = order[step - 1]
        n_rem = int((~visited).sum())
        if lo <= n_rem <= hi:
            st = eval_state(bank, student, teacher, locs, current, visited, first, step_i, N - 1, device)
            piS_m = st["piS"].masked_fill(visited, float("-inf"))
            snaps.append({"current": current, "visited": visited.clone(),
                          "prefix": list(order[:step]), "step_i": step_i, "n_remaining": n_rem,
                          "a": int(piS_m.argmax()), "base": st})
        visited[order[step]] = True
    return snaps


def hull_vertices(locs, mask):
    from scipy.spatial import ConvexHull, QhullError
    idx = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
    if len(idx) < 3:
        return set(idx)
    pts = locs[mask].detach().cpu().numpy()
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return set(idx)
    return {idx[i] for i in hull.vertices.tolist()}


def build_triplet(locs, snap, first, cfg, device):
    N = locs.shape[0]
    a, current, visited = snap["a"], snap["current"], snap["visited"]
    mask = ~visited
    lo, hi = cfg["coord_lo"], cfg["coord_hi"]
    margin = cfg["anchor_unique_margin"]

    if snap["n_remaining"] < cfg["early_branch_min_n_remaining"]:
        return None, "not_early_stage"

    locs64 = locs.double()
    S = h02_semantics(locs, current, mask)
    mu, u_out = S["mu"], S["u_out"]
    c = locs64[current]
    x0 = locs64[first]
    if float(torch.norm(c - mu)) <= cfg["min_cur_centroid_norm"]:
        return None, "current_too_close_to_centroid"
    v = locs64[a] - c
    radius = float(torch.norm(v))
    if radius <= cfg["min_radius"]:
        return None, "target_radius_too_small"

    phi = signed_angle(v, u_out)
    if not (cfg["phi_min_deg"] <= abs(math.degrees(phi)) <= cfg["phi_max_deg"]):
        return None, "invalid_phi_range"

    a_P = c + mirror_about(v, u_out)
    a_T = c + rotate(v, 2.0 * phi)
    if bool((a_T < lo).any() or (a_T > hi).any()):
        return None, "treatment_target_out_of_bounds"
    if bool((a_P < lo).any() or (a_P > hi).any()):
        return None, "placebo_target_out_of_bounds"
    dA_T, dA_P = a_T - locs64[a], a_P - locs64[a]

    d0 = float(torch.norm(locs64[a] - x0))
    dd0_T = float(torch.norm(a_T - x0)) - d0
    dd0_P = float(torch.norm(a_P - x0)) - d0
    if abs(dd0_T - dd0_P) > cfg["dstart_match_tol"]:
        return None, "start_distance_mismatch"

    anchor = S["anchor"]
    if float(S["d_cen"][anchor]) - S["anchor_second"] < margin:
        return None, "max_anchor_not_unique"
    if anchor == a:
        return None, "target_is_max_anchor"
    anchor_d = float(S["d_cen"][anchor])
    if float(torch.norm(a_T - mu)) > anchor_d - margin or \
       float(torch.norm(a_P - mu)) > anchor_d - margin:
        return None, "max_anchor_changed"

    base = snap["base"]
    piS_b, piT_b = base["piS"], base["piT"]
    pool = mask.clone()
    pool[a] = False; pool[anchor] = False
    if not bool(pool.any()):
        return None, "reference_unavailable"
    r0 = int(piS_b.masked_fill(~pool, -1.0).argmax())

    bT = locs64 - dA_T
    bP = locs64 - dA_P
    ok = mask.clone()
    for j in (a, anchor, r0, current, first):
        ok[j] = False
    ok &= ((bT >= lo).all(-1) & (bT <= hi).all(-1) & (bP >= lo).all(-1) & (bP <= hi).all(-1))
    ok &= (torch.norm(bT - mu, dim=-1) <= anchor_d - margin)
    ok &= (torch.norm(bP - mu, dim=-1) <= anchor_d - margin)
    others = torch.stack([locs64[a], a_T, a_P, locs64[r0], c, x0, locs64[anchor]])
    sep = torch.stack([torch.cdist(x, others).min(dim=-1).values
                       for x in (locs64, bT, bP)]).min(0).values
    ok &= (sep >= cfg["comp_min_sep"])
    if not bool(ok.any()):
        return None, "no_valid_compensator"

    ddcur = (torch.norm(bT - c, dim=-1) - torch.norm(bP - c, dim=-1)).abs()
    ddst = (torch.norm(bT - x0, dim=-1) - torch.norm(bP - x0, dim=-1)).abs()
    big = torch.full((N,), 1e9, device=device, dtype=torch.float64)
    col_ok = mask.clone(); col_ok[a] = False
    eye = torch.eye(N, device=device, dtype=torch.float64) * 1e9

    def _nn_perturbed(bX, aX):
        D = torch.cdist(bX, locs64)
        D = D.masked_fill(~col_ok.unsqueeze(0), 1e9) + eye
        return torch.minimum(D.min(dim=-1).values, torch.norm(bX - aX, dim=-1))
    ddnn = (_nn_perturbed(bT, a_T) - _nn_perturbed(bP, a_P)).abs()
    a_imgs = torch.stack([locs64[a], a_T, a_P])
    near_t = (torch.stack([torch.cdist(x, a_imgs).min(dim=-1).values
                           for x in (locs64, bT, bP)]).min(0).values
              < cfg["comp_near_radius"]).double()
    near_r = (torch.stack([torch.norm(x - locs64[r0], dim=-1) for x in (locs64, bT, bP)]).min(0).values
              < cfg["comp_near_radius"]).double()
    near_p = ((torch.stack([torch.norm(x - mu, dim=-1) for x in (locs64, bT, bP)]).max(0).values
               / anchor_d) > cfg["comp_periph_frac"]).double()
    cost = (piT_b.double() + piS_b.double()
            + cfg["comp_pen_target"] * near_t
            + cfg["comp_pen_reference"] * near_r
            + cfg["comp_pen_periph"] * near_p
            + cfg["comp_w_dcur"] * ddcur + cfg["comp_w_dstart"] * ddst + cfg["comp_w_nn"] * ddnn)
    b = int(torch.where(ok, cost, big).argmin())

    locs_T = locs.clone(); locs_T[a] = a_T.to(locs.dtype); locs_T[b] = bT[b].to(locs.dtype)
    locs_P = locs.clone(); locs_P[a] = a_P.to(locs.dtype); locs_P[b] = bP[b].to(locs.dtype)

    S_T = h02_semantics(locs_T, current, mask)
    S_P = h02_semantics(locs_P, current, mask)
    if float(torch.norm(S_T["mu"] - mu)) >= cfg["centroid_tol"] or \
       float(torch.norm(S_P["mu"] - mu)) >= cfg["centroid_tol"]:
        return None, "centroid_mismatch"
    if S_T["anchor"] != anchor or S_P["anchor"] != anchor or \
       abs(float(S_T["max_d"] - S["max_d"])) >= cfg["maxdist_tol"] or \
       abs(float(S_P["max_d"] - S["max_d"])) >= cfg["maxdist_tol"]:
        return None, "max_anchor_changed"

    moved = torch.stack([locs64[a], locs_T[a].double(), locs_P[a].double(),
                         locs64[b], locs_T[b].double(), locs_P[b].double()])
    rok = mask.clone()
    for j in (a, b, anchor, current, first):
        rok[j] = False
    rok &= (torch.cdist(locs64, moved).min(dim=-1).values >= cfg["ref_min_sep"])
    if not bool(rok.any()):
        return None, "reference_unavailable"
    r = int(piS_b.masked_fill(~rok, -1.0).argmax())

    disp_T = float(torch.norm(locs_T[a].double() - locs64[a]))
    disp_P = float(torch.norm(locs_P[a].double() - locs64[a]))
    disp_bT = float(torch.norm(locs_T[b].double() - locs64[b]))
    disp_bP = float(torch.norm(locs_P[b].double() - locs64[b]))
    nn_a = float(torch.where(mask & (torch.arange(N, device=device) != a),
                             torch.norm(locs64 - locs64[a], dim=-1), big).min())
    dX2 = float(S_T["X2"][a] - S["X2"][a])
    trip = {
        "a": a, "b": b, "r": r, "anchor": anchor, "current": current, "first": first,
        "visited": visited, "mask": mask, "prefix": snap["prefix"],
        "step_i": snap["step_i"], "n_remaining": snap["n_remaining"],
        "locs_base": locs, "locs_T": locs_T, "locs_P": locs_P,
        "phi_deg": math.degrees(phi), "radius": radius,
        "disp_target": disp_T, "disp_target_placebo": disp_P,
        "disp_comp": disp_bT, "disp_comp_placebo": disp_bP,
        "total_disp_T": disp_T + disp_bT, "total_disp_P": disp_P + disp_bP,
        "dd_start_target_T": dd0_T, "dd_start_target_P": dd0_P,
        "comp_ddcur": float(ddcur[b]), "comp_ddstart": float(ddst[b]), "comp_ddnn": float(ddnn[b]),
        "S_base": S, "S_T": S_T, "S_P": S_P,
        "dX2": dX2, "orientation": 1.0 if dX2 > 0 else -1.0,
        "nn_base_target": nn_a,
    }
    return trip, None


def analytic_checks(trip, ev, cfg):
    m2 = cfg["H02_index"]
    a, r = trip["a"], trip["r"]
    S, S_T, S_P = trip["S_base"], trip["S_T"], trip["S_P"]
    e = {}

    e["centroid_T"] = float(torch.norm(S_T["mu"] - S["mu"]))
    e["centroid_P"] = float(torch.norm(S_P["mu"] - S["mu"]))
    assert e["centroid_T"] < cfg["centroid_tol"], f"centroid moved (treatment): {e['centroid_T']:.2e}"
    assert e["centroid_P"] < cfg["centroid_tol"], f"centroid moved (placebo): {e['centroid_P']:.2e}"

    e["dcur_T"] = abs(float(S_T["d_cur"][a] - S["d_cur"][a]))
    e["dcur_P"] = abs(float(S_P["d_cur"][a] - S["d_cur"][a]))
    assert e["dcur_T"] < cfg["dist_tol"], f"d_cur(a) moved (treatment): {e['dcur_T']:.2e}"
    assert e["dcur_P"] < cfg["dist_tol"], f"d_cur(a) moved (placebo): {e['dcur_P']:.2e}"

    e["disp_match"] = abs(trip["total_disp_T"] - trip["total_disp_P"])
    assert e["disp_match"] < cfg["displacement_tol"], f"displacement mismatch: {e['disp_match']:.2e}"
    assert S["anchor"] == S_T["anchor"] == S_P["anchor"], "max-periphery anchor changed"
    e["maxd_T"] = abs(float(S_T["max_d"] - S["max_d"]))
    e["maxd_P"] = abs(float(S_P["max_d"] - S["max_d"]))
    assert e["maxd_T"] < cfg["maxdist_tol"], f"denominator moved (treatment): {e['maxd_T']:.2e}"
    assert e["maxd_P"] < cfg["maxdist_tol"], f"denominator moved (placebo): {e['maxd_P']:.2e}"

    e["X2_placebo"] = abs(float(S_P["X2"][a] - S["X2"][a]))
    assert e["X2_placebo"] < cfg["analytic_tol"], f"placebo changed X2(a): {e['X2_placebo']:.2e}"
    e["raw_placebo_target"] = abs(float(ev["plac"]["raw"][m2][a] - ev["base"]["raw"][m2][a]))
    assert e["raw_placebo_target"] < cfg["analytic_tol"], \
        f"placebo changed the H02 raw logit at a: {e['raw_placebo_target']:.2e}"

    exp_d = float(S_T["X2"][a] - S["X2"][a])
    obs_d = float(ev["treat"]["raw"][m2][a] - ev["base"]["raw"][m2][a])
    e["treat_analytic"] = abs(exp_d - obs_d)
    assert e["treat_analytic"] < cfg["analytic_tol"], \
        f"H02 treatment analytic fail: expected {exp_d:.6f} observed {obs_d:.6f}"
    assert exp_d < 0.0, f"construction should lower X2 but dX2={exp_d:.6f}"

    e["raw_ref_T"] = abs(float(ev["treat"]["raw"][m2][r] - ev["base"]["raw"][m2][r]))
    e["raw_ref_P"] = abs(float(ev["plac"]["raw"][m2][r] - ev["base"]["raw"][m2][r]))
    assert e["raw_ref_T"] < cfg["invariance_tol"], f"reference H02 logit moved (treat): {e['raw_ref_T']:.2e}"
    assert e["raw_ref_P"] < cfg["invariance_tol"], f"reference H02 logit moved (plac): {e['raw_ref_P']:.2e}"

    for k in ("base", "treat", "plac"):
        rc = float((ev[k]["piS"] - (ev[k]["w"][:, None] * ev[k]["V"]).sum(0)).abs().max())
        e[f"recon_{k}"] = rc
        assert rc < cfg["recon_tol"], f"pi_S != sum_m w_m V_m in {k}: {rc:.2e}"
    return e


def triplet_metrics(bank, student, teacher, trip, cfg, device):
    N = trip["locs_base"].shape[0]
    a, r, m2 = trip["a"], trip["r"], cfg["H02_index"]
    eps = cfg["eps_logodds"]
    M = len(bank)
    ev = {}
    for name, L in (("base", trip["locs_base"]), ("treat", trip["locs_T"]), ("plac", trip["locs_P"])):
        ev[name] = eval_state(bank, student, teacher, L, trip["current"], trip["visited"],
                              trip["first"], trip["step_i"], N - 1, device)

    def E_all(v):
        return ev[v]["raw"][:, a] - ev[v]["raw"][:, r]

    def R_all(v):
        lw = torch.log(ev[v]["w"] + eps)
        return lw - (lw.sum() - lw) / (M - 1)

    def C_all(v):
        w, V = ev[v]["w"], ev[v]["V"]
        c = w * V[:, a]
        return c / (c.sum() + 1e-12)

    def E(v):
        return float(ev[v]["raw"][m2][a] - ev[v]["raw"][m2][r])

    def R(v):
        return float(R_all(v)[m2])

    def Cs(v):
        return float(C_all(v)[m2])

    def LO(v, key):
        return math.log(float(ev[v][key][a]) + eps) - math.log(float(ev[v][key][r]) + eps)

    def LO_student_exact(v):
        lw = torch.log(ev[v]["w"])
        lv = ev[v]["logV"]
        return float(torch.logsumexp(lw + lv[:, a], 0) - torch.logsumexp(lw + lv[:, r], 0))

    def LO_teacher_exact(v):
        return float(ev[v]["logitsT"][a] - ev[v]["logitsT"][r])

    def w_rank(v):
        return int((ev[v]["w"] > ev[v]["w"][m2]).sum()) + 1

    def contrib_rank(v):
        cc = ev[v]["w"] * ev[v]["V"][:, a]
        return int((cc > cc[m2]).sum()) + 1

    def top1(v, key):
        return int(ev[v][key].masked_fill(trip["visited"], -1e9).argmax())

    o = trip["orientation"]

    def ctrl(fn):
        return o * (fn("treat") - fn("plac"))

    m = {
        "E_H02":     ctrl(E),
        "R_H02":     ctrl(R),
        "C_H02":     ctrl(Cs),
        "L_student": ctrl(lambda v: LO(v, "piS")),
        "L_teacher": ctrl(lambda v: LO(v, "piT")),
        "L_student_exact": ctrl(LO_student_exact),
        "L_teacher_exact": ctrl(LO_teacher_exact),
        "L_student_raw": LO("treat", "piS") - LO("plac", "piS"),
        "L_teacher_raw": LO("treat", "piT") - LO("plac", "piT"),
        "L_student_exact_raw": LO_student_exact("treat") - LO_student_exact("plac"),
        "L_teacher_exact_raw": LO_teacher_exact("treat") - LO_teacher_exact("plac"),
        "E_H02_base": E("base"), "E_H02_treat": E("treat"), "E_H02_plac": E("plac"),
        "R_H02_base": R("base"), "R_H02_treat": R("treat"), "R_H02_plac": R("plac"),
        "C_H02_base": Cs("base"), "C_H02_treat": Cs("treat"), "C_H02_plac": Cs("plac"),
        "w_H02_base": float(ev["base"]["w"][m2]), "w_H02_treat": float(ev["treat"]["w"][m2]),
        "w_H02_plac": float(ev["plac"]["w"][m2]),
        "w_rank_base": w_rank("base"), "w_rank_treat": w_rank("treat"), "w_rank_plac": w_rank("plac"),
        "contrib_rank_base": contrib_rank("base"),
        "LO_student_base": LO("base", "piS"), "LO_teacher_base": LO("base", "piT"),
        "LO_student_exact_base": LO_student_exact("base"),
        "LO_teacher_exact_base": LO_teacher_exact("base"),
        "base_agree": top1("base", "piS") == top1("base", "piT"),
        "student_top1_stable": len({top1(v, "piS") for v in ev}) == 1,
        "teacher_top1_stable": len({top1(v, "piT") for v in ev}) == 1,
    }

    thr = cfg["primary_contrib_threshold"]
    m["H02_dominant"] = m["C_H02_base"] > thr
    m["pop_primary"] = bool(m["base_agree"] and m["H02_dominant"])
    m["pop_secondary"] = bool(m["base_agree"] and m["contrib_rank_base"] == 1)
    m["pop_secondary_excl_primary"] = bool(m["pop_secondary"] and not m["pop_primary"])
    m["pop_diagnostic"] = bool(not m["base_agree"])

    sE = {v: E_all(v) for v in ("base", "treat", "plac")}
    sR = {v: R_all(v) for v in ("base", "treat", "plac")}
    sC = {v: C_all(v) for v in ("base", "treat", "plac")}
    for mm in range(M):
        m[f"specE_m{mm:02d}"] = float(o * (sE["treat"][mm] - sE["plac"][mm]))
        m[f"specR_m{mm:02d}"] = float(o * (sR["treat"][mm] - sR["plac"][mm]))
        m[f"specC_m{mm:02d}"] = float(o * (sC["treat"][mm] - sC["plac"][mm]))
        m[f"specC_base_m{mm:02d}"] = float(sC["base"][mm])

    w, V, piS = ev["base"]["w"], ev["base"]["V"], ev["base"]["piS"]
    keep = [i for i in range(M) if i != m2]
    w_fix = w[keep] / w[keep].sum()
    pi_fix = (w_fix[:, None] * V[keep]).sum(0)
    batch = decision_state(trip["locs_base"].unsqueeze(0), trip["visited"].unsqueeze(0),
                           torch.tensor([trip["current"]], device=device), torch.tensor([trip["first"]], device=device),
                           trip["step_i"] + 1)
    with torch.no_grad():
        pi_rr, w_rr = student(build_td(batch, device), V[keep].unsqueeze(0))
    pi_rr, w_rr = pi_rr[0], w_rr[0]
    m["del_reroute_equals_fixed"] = float((w_rr - w_fix).abs().max())
    for tag, pi_new in (("fixed", pi_fix), ("rerouted", pi_rr)):
        m[f"del_{tag}_dp_target"] = float(pi_new[a] - piS[a])
        m[f"del_{tag}_dlogodds"] = ((math.log(float(pi_new[a]) + eps) - math.log(float(pi_new[r]) + eps))
                                    - (math.log(float(piS[a]) + eps) - math.log(float(piS[r]) + eps)))
        p = piS.clamp_min(1e-12); q = pi_new.clamp_min(1e-12)
        m[f"del_{tag}_kl"] = float((p * (p.log() - q.log()))[trip["mask"]].sum())
    return m, ev


def run_shard(teacher, bank, student, cfg, device, inst_lo, inst_hi):
    N = cfg["problem_size"]
    valid, rejections, aerrs, events = [], collections.Counter(), [], []
    n_primary = 0
    for inst_id, locs in enumerate(C.tsp_instances(cfg["seed"], N, inst_lo, inst_hi, device), start=inst_lo):
        snaps = natural_states(bank, student, teacher, locs, 0, cfg, device)
        step = max(1, len(snaps) // max(cfg["states_per_instance"], 1))
        snaps = snaps[::step][:cfg["states_per_instance"]]
        for si, snap in enumerate(snaps):
            trip, why = build_triplet(locs, snap, 0, cfg, device)
            if trip is None:
                rejections[why] += 1
                events.append(("r", why))
                continue
            m, ev = triplet_metrics(bank, student, teacher, trip, cfg, device)
            aerrs.append(analytic_checks(trip, ev, cfg))
            hb = {k: hull_vertices(x, trip["mask"]) for k, x in
                  (("base", trip["locs_base"]), ("treat", trip["locs_T"]), ("plac", trip["locs_P"]))}
            rec = dict(m)
            rec.update({
                "instance_id": inst_id, "rollout_id": 0, "state_idx": si, "step": trip["step_i"],
                "step_norm": ev["base"]["step_norm"], "n_remaining": trip["n_remaining"],
                "a": trip["a"], "b": trip["b"], "r": trip["r"], "anchor": trip["anchor"],
                "current_node": trip["current"], "first_node": trip["first"],
                "phi_deg": trip["phi_deg"], "radius": trip["radius"],
                "disp_target": trip["disp_target"], "disp_comp": trip["disp_comp"],
                "total_disp_T": trip["total_disp_T"], "total_disp_P": trip["total_disp_P"],
                "dd_start_target_T": trip["dd_start_target_T"],
                "dd_start_target_P": trip["dd_start_target_P"],
                "comp_ddcur": trip["comp_ddcur"], "comp_ddstart": trip["comp_ddstart"],
                "comp_ddnn": trip["comp_ddnn"], "nn_base_target": trip["nn_base_target"],
                "dX2": trip["dX2"], "orientation": trip["orientation"],
                "hull_base": trip["a"] in hb["base"], "hull_treat": trip["a"] in hb["treat"],
                "hull_plac": trip["a"] in hb["plac"],
                "hull_base_to_interior": (trip["a"] in hb["base"] and trip["a"] in hb["plac"]
                                          and trip["a"] not in hb["treat"]),
                "hull_bonus_a_base": float(trip["S_base"]["hull_bonus"][trip["a"]]),
                "continuity_a_base": float(trip["S_base"]["continuity"][trip["a"]]),
                "X2_a_base": float(trip["S_base"]["X2"][trip["a"]]),
                "d_cur_a_base": float(trip["S_base"]["d_cur"][trip["a"]]),
            })
            events.append(("v", len(valid)))
            valid.append(rec)
            n_primary += int(rec["pop_primary"])
        print(f"  [inst {inst_id}] valid={len(valid)}  primary={n_primary}  "
              f"rejected={sum(rejections.values())}", flush=True)
    print(f"\n  shard [{inst_lo}, {inst_hi}) complete: {len(valid)} valid, "
          f"{sum(rejections.values())} rejected (no stop rule applied)")
    return {"inst_lo": inst_lo, "inst_hi": inst_hi, "valid": valid, "rejections": dict(rejections),
            "aerrs": aerrs, "events": events}


def load_shards(cfg, shard_dir, expected_fp, allow_unverified=False):
    paths = sorted((os.path.join(shard_dir, f) for f in os.listdir(shard_dir)
                    if f.startswith("shard_") and f.endswith(".pt")),
                   key=lambda x: int(os.path.basename(x).split("_")[1]))
    if not paths:
        raise SystemExit(f"STOP: no shard_*.pt found in {shard_dir}")
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    names = [os.path.basename(p) for p in paths]
    for name, sh in zip(names, shards):
        if name != f"shard_{sh['inst_lo']}_{sh['inst_hi']}.pt":
            raise SystemExit(f"STOP: {name} holds instances [{sh['inst_lo']}, {sh['inst_hi']})")
    fp = C.common_fingerprint([sh.get("fingerprint") for sh in shards], names, "shards", allow_unverified)
    if fp is not None:
        C.check_fingerprint(fp, expected_fp, f"the shards in {shard_dir}")
    C.check_tiling([(sh["inst_lo"], sh["inst_hi"], n) for n, sh in zip(names, shards)], *cfg["instances"],
                   "shards", complete=False)
    return shards


def merge_shards(cfg, shards):
    valid, rejections = [], collections.Counter()
    n_primary = 0
    cut = None
    lo_hi = []
    for sh in shards:
        lo, hi = sh["inst_lo"], sh["inst_hi"]
        lo_hi.append((lo, hi, len(sh["valid"])))
        if cut is not None:
            continue
        for kind, payload in sh["events"]:
            if kind == "r":
                rejections[payload] += 1
                continue
            rec = sh["valid"][payload]
            valid.append(rec)
            n_primary += int(rec["pop_primary"])
            hit_primary = cfg.get("stop_on_primary", False) and n_primary >= cfg["target_primary"]
            if hit_primary or len(valid) >= cfg["max_valid"]:
                cut = (rec["instance_id"], rec["state_idx"])
                break
    print("\n" + "=" * 78)
    print("SHARD MERGE  (execution detail only -- instance-ordered replay of the same stream)")
    print("=" * 78)
    for lo, hi, nv in lo_hi:
        print(f"  shard [{lo:>6d}, {hi:>6d})  valid={nv}")
    if cut:
        print(f"  cut at the first point the base-only stop rule fires: "
              f"instance {cut[0]} / state {cut[1]}")
        reason = (f"primary target reached ({n_primary} >= {cfg['target_primary']})"
                  if cfg.get("stop_on_primary", False) and n_primary >= cfg["target_primary"]
                  else f"HARD CAP on valid triplets bound first ({len(valid)} >= "
                       f"{cfg['max_valid']}); primary stopped at {n_primary} of "
                       f"{cfg['target_primary']}")
    else:
        reason = (f"WARNING: stop rule never fired -- the sharded instance range was exhausted "
                  f"with n_valid={len(valid)} (cap {cfg['max_valid']}), n_primary={n_primary} "
                  f"(target {cfg['target_primary']}). Extend the shard range and re-merge.")
        print("\n  *** " + reason + " ***")
    print(f"  kept {len(valid)} valid triplets, {n_primary} strict-primary, "
          f"{sum(rejections.values())} rejections")
    stop = {"n_valid": len(valid), "n_primary": n_primary,
            "n_instances_scanned": lo_hi[-1][1] if lo_hi else 0,
            "target_primary": cfg["target_primary"], "max_valid": cfg["max_valid"],
            "stop_on_primary": cfg.get("stop_on_primary", False),
            "reason": reason, "merged_from_shards": [[a, b, c] for a, b, c in lo_hi],
            "cut_at": list(cut) if cut else None}
    return valid, stop


def summarize_primary(valid, cfg, stop, out_dir):
    thr = cfg["primary_contrib_threshold"]
    prim = [r for r in valid if r["pop_primary"]]
    ids = [r["instance_id"] for r in prim]
    print(f"\n  valid triplets = {len(valid)}   strict-primary = {len(prim)}   stop: {stop['reason']}")
    print(f"\n  PRIMARY -- base teacher/student agree AND C_H02(a,base) > {thr}   "
          f"(N={len(prim)}, instance clusters={len(set(ids))})")
    five = {}
    if len(prim) < 3:
        print("    too few triplets -- reporting N only, no effect estimate")
    else:
        print(f"    {'metric':<17s}{'pred':>5s}{'mean':>11s}{'median':>11s}{'95% CI':>25s}{'dir%':>7s}"
              f"{'inst-mean':>11s}{'p':>9s}      (mean, median, CI: states; p: instance means)")
        for k in FIVE:
            v = [r[k] for r in prim]
            t = torch.tensor(v)
            lo, hi = C.bootstrap_ci(v, ids, "hull", cfg["n_boot"], seed=0)
            p = C.signflip_p(v, ids, "hull", cfg["n_perm"], seed=1)
            mi = C.instance_mean(v, ids, "hull")
            dc = float((t > 0).float().mean())
            print(f"    {k:<17s}{PRED[k]:>5s}{float(t.mean()):>11.5f}{float(t.median()):>11.5f}"
                  f"   [{lo:>9.5f},{hi:>9.5f}]{100*dc:>6.1f}%{mi:>11.5f}{p:>9.4f}")
            five[k] = {"pred": PRED[k], "n": len(v), "mean_state": float(t.mean()),
                       "median_state": float(t.median()), "ci95_state": [lo, hi], "frac_positive": dc,
                       "mean_instance": mi, "p_signflip_instance_mean": p}
    out = {"stop_rule": stop, "n_valid": len(valid), "n_primary": len(prim),
           "n_instance_clusters": len(set(ids)), "five_values": five}
    C.write_json(os.path.join(out_dir, "five_values.json"), out)
    return out


def run(a, cfg):
    shard_dir = os.path.join(a.out, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    fp = C.experiment_fingerprint(NAME, PROTOCOL, ANALYSIS_KEYS, a, cfg)
    print(f"[fingerprint] {C.fingerprint_id(fp)}  {fp}")
    if not a.merge:
        device = torch.device(a.device)
        torch.manual_seed(cfg["seed"])
        teacher, bank, student = load(cfg, a, device)
        if a.inst_lo is not None or a.inst_hi is not None:
            lo, hi = a.inst_lo or 0, a.inst_hi if a.inst_hi is not None else cfg["instances"][1]
            ranges, merge = [(lo, hi)], False
        else:
            lo, hi = cfg["instances"]
            size = a.shard_size or cfg["shard_size"]
            ranges, merge = [(s, min(s + size, hi)) for s in range(lo, hi, size)], True
        for lo, hi in ranges:
            path = os.path.join(shard_dir, f"shard_{lo}_{hi}.pt")
            if merge and os.path.exists(path):
                old = torch.load(path, map_location="cpu", weights_only=False)
                C.check_fingerprint(old.get("fingerprint"), fp, path, a.allow_unverified)
                print(f"  {path} exists (same experiment), skipping")
                continue
            sh = run_shard(teacher, bank, student, cfg, device, lo, hi)
            sh["fingerprint"] = fp
            torch.save(sh, path)
            print(f"  shard done: {len(sh['valid'])} valid, {sum(sh['rejections'].values())} rejected -> {path}")
        if not merge:
            return
    valid, stop = merge_shards(cfg, load_shards(cfg, shard_dir, fp, a.allow_unverified))
    summarize_primary(valid, cfg, stop, a.out)
