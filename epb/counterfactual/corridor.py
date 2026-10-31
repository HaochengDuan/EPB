import collections
import csv
import json
import math
import os
import time

import numpy as np
import torch

from . import common as C

NAME = "corridor"
SETTING = "pomo_cvrp"
STUDENT = "pomo_cvrp50"
CONFIG = os.path.join(C.HERE, "corridor.json")
PROTOCOL = 1
ANALYSIS_KEYS = ("batches", "primary", "secondary_load_thresholds", "share_grid", "paper_id_of_bank", "n_boot",
                 "n_perm", "null_tolerance")
PREDICTED_SIGN = {"g_E_opp": "-", "g_C_opp": None}


def predicted_sign(key):
    return PREDICTED_SIGN.get(key, "+")


def add_args(ap):
    ap.add_argument("--batch", choices=["A", "B"], help="collect one batch only (default: A, B, then --analyze)")
    ap.add_argument("--max-instances", type=int,
                    help="only the first N instances of each batch (a multiple of rollout_batch_instances)")
    ap.add_argument("--analyze", action="store_true", help="only the analysis of OUT/out_A and OUT/out_B")


def load(cfg, a, device):
    student, bank, ck, bank_json = C.load_student(SETTING, a.student, device, bank=a.bank)
    raw = C.bank_entries(bank_json)
    M = len(bank)
    assert len(raw) == M
    sets = cfg["carriers"] + cfg["opposing_family"] + cfg["leak_set"]
    assert M > max(sets)
    assert not (set(cfg["carriers"]) & set(cfg["leak_set"])), "a carrier cannot be in the leak set"
    assert not (set(cfg["opposing_family"]) & set(cfg["leak_set"])), "an opposing reader cannot be in the leak set"
    teacher = C.load_teacher(SETTING, a.teacher_ckpt, device).model
    return teacher, bank, student, ck["args"], dict(epoch=ck.get("epoch"), M=M)


def build(st, a, r, cfg):
    locs = st["locs"].double().numpy(); cur = st["current_node"]
    c, o = locs[cur], locs[0]
    v, u = locs[a] - c, o - c
    rho, s = float(np.linalg.norm(v)), float(np.linalg.norm(u))
    if rho < 1e-6 or s < 1e-6:
        return None, "degenerate"
    al = C.signed_angle(u, v)
    if not (math.radians(cfg["alpha_min_deg"]) <= abs(al) <= math.radians(cfg["alpha_max_deg"])):
        return None, "alpha_outside_window"
    proj = rho * math.cos(al)
    if not (0.0 <= proj <= s):
        return None, "target_not_on_segment"
    sgn = 1.0 if al >= 0 else -1.0
    aT, aP = c + C.rotate2(v, +2.0 * abs(al) * sgn), c + C.rotate2(v, -2.0 * abs(al) * sgn)
    for x in (aT, aP):
        if not (cfg["coord_lo"] <= x.min() and x.max() <= cfg["coord_hi"]):
            return None, "out_of_bounds"
        d = np.linalg.norm(locs - x, axis=1); d[a] = np.inf
        if d.min() < cfg["min_node_sep"]:
            return None, "node_collision"
    return {"aT": aT, "aP": aP, "alpha": al, "rho": rho, "s": s, "d_a_depot": float(np.linalg.norm(locs[a] - o)),
            "d_aT_depot": float(np.linalg.norm(aT - o)), "chord": 2 * rho * math.sin(abs(al))}, None


def analytic_pred(t, cfg):
    al = abs(t["alpha"])
    m1, m2 = cfg["carriers"]
    return {m1: 3.0 * t["rho"] * (abs(math.sin(3 * al)) - abs(math.sin(al))),
            m2: 6.0 * (t["d_aT_depot"] - t["d_a_depot"])}


@torch.no_grad()
def eval_arm(teacher, bank, student, states, device, bs, V_base=None, clamps=None):
    n = len(states)
    res = None
    for lo in range(0, n, bs):
        sub = states[lo:lo + bs]; hi = lo + len(sub)
        b, mask = C.cvrp_stack_states(sub, device)
        V = bank.compute_probs(b)[0]
        piS, w = student(b, V)
        piT = C.cvrp_teacher_probs(teacher, b, mask)
        raw = C.cvrp_raw_scores(bank, b, mask)
        cur = {"raw": raw, "V": V, "w": w, "piS": piS, "piT": piT, "mask": mask}
        if clamps:
            Vb = V_base[lo:lo + bs].to(device)
            for name, hold in clamps.items():
                Vc = V.clone(); Vc[:, hold] = Vb[:, hold]
                pc, _ = student(b, Vc); cur["piS_clamp_" + name] = pc
        if res is None:
            res = {k: torch.empty((n,) + tuple(v.shape[1:]), dtype=v.dtype) for k, v in cur.items()}
        for k, v in cur.items():
            res[k][lo:hi] = v.cpu()
    return res


def lo_eps(p, a, r, eps):
    return math.log(float(p[a]) + eps) - math.log(float(p[r]) + eps)


def readouts(ev, k, a, r, cfg):
    raw = ev["raw"][k].double(); w = ev["w"][k].double(); V = ev["V"][k].double()
    pS, pT = ev["piS"][k].double(), ev["piT"][k].double()
    fam = cfg["carriers"]; opp = cfg["opposing_family"]; non = [m for m in range(raw.shape[0]) if m not in fam]
    Cm = w * (V[:, a] - V[:, r]); eps = cfg["eps_clip"]
    d = {"M_S": float(pS[a] - pS[r]), "M_T": float(pT[a] - pT[r]), "L_S": lo_eps(pS, a, r, 1e-9),
         "L_S3": lo_eps(pS, a, r, eps), "L_T": lo_eps(pT, a, r, 1e-9), "L_T3": lo_eps(pT, a, r, eps),
         "E_fam": float(sum(raw[m, a] - raw[m, r] for m in fam)),
         "E_opp": float(sum(raw[m, a] - raw[m, r] for m in opp)),
         "R_fam": float(torch.log(w[fam].sum() + 1e-12) - torch.log(w[non].sum() + 1e-12)),
         "C_fam": float(Cm[fam].sum()), "C_opp": float(Cm[opp].sum()), "w_fam": float(w[fam].sum()),
         "w_opp": float(w[opp].sum()), "VD_fam": float(sum(V[m, a] - V[m, r] for m in fam)),
         "argmax_S": int(pS.argmax()), "argmax_T": int(pT.argmax())}
    for m in range(raw.shape[0]):
        d[f"E_m{m:02d}"] = float(raw[m, a] - raw[m, r])
    for m in fam:
        d[f"C_m{m:02d}"] = float(Cm[m]); d[f"w_m{m:02d}"] = float(w[m]); d[f"Vdepot_m{m:02d}"] = float(V[m, 0])
    for key in ev:
        if key.startswith("piS_clamp_"):
            pc = ev[key][k].double(); tag = key[len("piS_clamp_"):]
            d["M_S_" + tag] = float(pc[a] - pc[r]); d["L_S3_" + tag] = lo_eps(pc, a, r, eps)
    return d


@torch.no_grad()
def run_batch(cfg, teacher, bank, student, info, device, batch, out_dir, max_instances=None, fp=None):
    seed, off, (lo_i, hi_i) = cfg["seed"], cfg["instance_id_offset"], cfg["batches"][batch]
    if max_instances is not None:
        hi_i = min(hi_i, lo_i + max_instances)
    os.makedirs(out_dir, exist_ok=True)
    N, B, keep_mod = cfg["problem_size"], cfg["rollout_batch_instances"], cfg["keep_mod"]
    t0 = time.time()
    states, n_all = C.cvrp_collect_states(teacher, cfg, device, C.cvrp_instances(seed, lo_i, hi_i, B, N),
                                          off + lo_i, keep_mod=keep_mod)
    print(f"[{batch}] instances [{lo_i},{hi_i}) seed {seed} states kept {len(states)} of {n_all} "
          f"({time.time()-t0:.0f}s)", flush=True)
    ev0 = eval_arm(teacher, bank, student, states, device, cfg["eval_batch"])
    trips, rej = [], collections.Counter()
    for k, st in enumerate(states):
        mask = ev0["mask"][k]; pS = ev0["piS"][k]; a = int(pS.argmax())
        if a == 0 or not bool(mask[a]):
            rej["target_depot_or_infeasible"] += 1; continue
        locs = st["locs"].double().numpy(); c = locs[st["current_node"]]
        d = np.linalg.norm(locs - c, axis=1); d[0] = np.inf; d[~mask.numpy()] = np.inf; d[a] = np.inf
        if not np.isfinite(d).any():
            rej["no_reference"] += 1; continue
        r = int(d.argmin())
        con, why = build(st, a, r, cfg)
        if why:
            rej[why] += 1; continue
        trips.append({"k": k, "a": a, "r": r, **con})
    print(f"[{batch}] constructed {len(trips)} / {len(states)}  rejections {dict(rej.most_common(6))} "
          f"({time.time()-t0:.0f}s)", flush=True)
    idx = [t["k"] for t in trips]
    ev0 = {key: val[idx] for key, val in ev0.items()}
    V_base = ev0["V"]
    M = len(bank)
    clamps = {"leak": cfg["leak_set"], "famonly": [m for m in range(M) if m not in cfg["carriers"]]}
    arms = {}
    for arm in ("T", "P"):
        sts = []
        for t in trips:
            st = dict(states[t["k"]]); L = st["locs"].clone()
            L[t["a"]] = torch.tensor(t["aT"] if arm == "T" else t["aP"], dtype=L.dtype); st["locs"] = L
            sts.append(st)
        arms[arm] = eval_arm(teacher, bank, student, sts, device, cfg["eval_batch"], V_base=V_base, clamps=clamps)
    print(f"[{batch}] arms evaluated ({time.time()-t0:.0f}s)", flush=True)
    rows, inv_rej, diag = [], collections.Counter(), collections.Counter()
    max_res = {m: 0.0 for m in cfg["carriers"]}
    for j, t in enumerate(trips):
        st = states[t["k"]]
        bse = readouts(ev0, j, t["a"], t["r"], cfg)
        T = readouts(arms["T"], j, t["a"], t["r"], cfg)
        P = readouts(arms["P"], j, t["a"], t["r"], cfg)
        fails = []
        for m in cfg["carriers"]:
            if abs(P[f"E_m{m:02d}"] - bse[f"E_m{m:02d}"]) > cfg["carrier_inert_tol"]:
                fails.append(f"carrier_{m}_not_inert_in_P")
        for arm, ev in (("T", arms["T"]), ("P", arms["P"])):
            if not torch.equal(ev["mask"][j], ev0["mask"][j]):
                fails.append(f"mask_changed_{arm}")
        if fails:
            for f in fails:
                inv_rej[f] += 1
            continue
        pred = analytic_pred(t, cfg)
        row = {"instance_id": st["instance_id"], "pomo": st["pomo"], "step_i": st["step_i"], "load": st["load"],
               "a": t["a"], "r": t["r"],
               "alpha_deg": math.degrees(t["alpha"]), "rho": t["rho"], "s": t["s"], "chord": t["chord"],
               "d_a_depot": t["d_a_depot"], "d_aT_depot": t["d_aT_depot"],
               "n_unvisited": int((st["visited"] < 0.5).sum()) - 1, "n_feasible_cust": int(ev0["mask"][j][1:].sum()),
               "teacher_agrees": int(bse["argmax_T"] == t["a"]),
               "share_fam": bse["C_fam"] / bse["M_S"] if abs(bse["M_S"]) > 1e-12 else float("nan"),
               "share_opp": bse["C_opp"] / bse["M_S"] if abs(bse["M_S"]) > 1e-12 else float("nan"),
               "base_M_S": bse["M_S"], "base_M_T": bse["M_T"], "base_w_fam": bse["w_fam"], "base_w_opp": bse["w_opp"],
               "base_L_S3": bse["L_S3"], "base_L_T3": bse["L_T3"],
               "top1_S_T": T["argmax_S"], "top1_S_P": P["argmax_S"], "top1_T_T": T["argmax_T"],
               "top1_T_P": P["argmax_T"]}
        for key in ("E_fam", "E_opp", "R_fam", "C_fam", "C_opp", "w_fam", "M_S", "L_S", "L_S3", "M_T", "L_T", "L_T3",
                    "M_S_leak", "L_S3_leak", "M_S_famonly", "L_S3_famonly"):
            row["g_" + key] = P[key] - T[key]
        for m in cfg["carriers"]:
            row[f"base_C_m{m:02d}"] = bse[f"C_m{m:02d}"]; row[f"base_w_m{m:02d}"] = bse[f"w_m{m:02d}"]
            row[f"base_Vdepot_m{m:02d}"] = bse[f"Vdepot_m{m:02d}"]
            row[f"gC_m{m:02d}"] = P[f"C_m{m:02d}"] - T[f"C_m{m:02d}"]
            row[f"gE_pred_m{m:02d}"] = pred[m]
            res = (P[f"E_m{m:02d}"] - T[f"E_m{m:02d}"]) - pred[m]
            row[f"res_pred_m{m:02d}"] = res; max_res[m] = max(max_res[m], abs(res))
            if abs(res) > cfg["analytic_pred_tol"]:
                diag[f"carrier_{m}_analytic_residual_gt_tol"] += 1
        row["VPinert_fam"] = P["VD_fam"] - bse["VD_fam"]; row["CPinert_fam"] = P["C_fam"] - bse["C_fam"]
        for m in range(M):
            row[f"gE_m{m:02d}"] = P[f"E_m{m:02d}"] - T[f"E_m{m:02d}"]
            row[f"Pinert_m{m:02d}"] = P[f"E_m{m:02d}"] - bse[f"E_m{m:02d}"]
        dfull = (arms["P"]["raw"][j].double() - ev0["raw"][j].double()).abs()
        dfull[:, t["a"]] = 0.0
        dfull[:, ~ev0["mask"][j]] = 0.0
        for m in range(M):
            row[f"Pinert_full_m{m:02d}"] = float(dfull[m].max())
        rows.append(row)
    print(f"[{batch}] accepted {len(rows)}  invariant rejections {dict(inv_rej)}  diagnostics {dict(diag)}  "
          f"max analytic residual {max_res}", flush=True)
    C.write_csv(os.path.join(out_dir, "triplets.csv"), rows, list(rows[0].keys()))
    C.write_json(os.path.join(out_dir, "funnel.json"),
                 {"batch": batch, "seed": seed, "instance_id_offset": off, "instances": [lo_i, hi_i],
                  "n_states_all": n_all, "n_states_kept": len(states), "n_constructed": len(trips),
                  "n_accepted": len(rows), "rejections": dict(rej), "invariant_rejections": dict(inv_rej),
                  "diagnostics": dict(diag), "max_abs_analytic_residual": {str(k): v for k, v in max_res.items()},
                  "student": info, "elapsed_s": time.time() - t0, "fingerprint": fp})


def load_rows(path):
    def fnum(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")
    rows = []
    with open(path) as f:
        for d in csv.DictReader(f):
            rows.append({k: (v == "True") if v in ("True", "False") else v if k == "tier" else fnum(v)
                         for k, v in d.items()})
    return rows


def describe(rows, key, cfg, seed=0):
    v = [r[key] for r in rows if r.get(key) == r.get(key)]
    ids = [r["instance_id"] for r in rows if r.get(key) == r.get(key)]
    if not v:
        return None
    tol = cfg["null_tolerance"]
    n = len(v)
    mean = sum(v) / n
    lo, hi = C.bootstrap_ci(v, ids, "corridor", cfg["n_boot"], seed=seed)
    q1, med, q3 = ([float(torch.quantile(torch.tensor(v, dtype=torch.float64), q)) for q in (0.25, 0.5, 0.75)])
    pos = sum(1 for x in v if x > tol)
    neg = sum(1 for x in v if x < -tol)
    nul = n - pos - neg
    sgn = 1.0 if mean >= 0 else -1.0
    pred = predicted_sign(key)
    return {"n": n, "n_instances": len(set(ids)), "mean_state": mean, "ci95_state_lo": lo, "ci95_state_hi": hi,
            "q1": q1, "median": med, "q3": q3,
            "pct_positive": 100.0 * pos / n, "pct_null": 100.0 * nul / n,
            "pct_negative": 100.0 * neg / n,
            "predicted_sign": pred,
            "pct_predicted_direction": (None if pred is None else 100.0 * (pos if pred == "+" else neg) / n),
            "pct_consistent_with_observed_sign": 100.0 * (pos if sgn > 0 else neg) / n,
            "observed_mean_sign": ("+" if sgn > 0 else "-"),
            "mean_instance": C.instance_mean(v, ids, "corridor"),
            "p_signflip_instance_mean": C.signflip_p(v, ids, "corridor", cfg["n_perm"], seed=seed + 1)}


def _pct(v):
    p = v["pct_predicted_direction"]
    return "n/a" if p is None else f"{p:.0f}%"


def table_rows(cfg):
    pid = cfg["paper_id_of_bank"]
    car = cfg["carriers"]
    names = "+".join(pid[str(m)] for m in car)
    rows = [("g_E_fam", f"1. carrier score ({names}, analytic)"), ("g_R_fam", "2. family relative routing log-weight"),
            ("g_C_fam", "3. family contribution"),
            ("g_L_S3", "4. student log-odds (clipped)"), ("g_L_T3", "5. teacher log-odds (clipped)"),
            ("g_M_T", "5m. teacher probability margin"), ("g_L_T", "5r. teacher log-odds eps=1e-9"),
            ("g_M_S", "4m. student margin"),
            ("g_L_S3_leak", "4c. student log-odds, leak set clamped"),
            ("g_L_S3_famonly", "4f. student log-odds, carriers only free"),
            ("g_E_opp", "opp. savings-family score"), ("g_C_opp", "opp. savings-family contribution"),
            ("g_L_S", "4r. student log-odds eps=1e-9")]
    extra = ([(f"gC_m{m:02d}", f"3{'ab'[k]}. contribution, bank {m} ({pid[str(m)]})") for k, m in enumerate(car)]
             + [(f"gE_m{m:02d}", f"1{'ab'[k]}. score, bank {m} ({pid[str(m)]})") for k, m in enumerate(car)])
    return rows, extra


def pops(cfg):
    P = cfg["primary"]; out = []
    out.append(("PRIMARY", lambda r: r["load"] < P["load_max"] and r["teacher_agrees"] == 1
                and r["share_fam"] >= P["share_min"] and r["base_M_S"] >= P["ms_floor"]))
    out.append(("no-agreement", lambda r: r["load"] < P["load_max"] and r["share_fam"] >= P["share_min"]
                and r["base_M_S"] >= P["ms_floor"]))
    for th in cfg["share_grid"]:
        out.append((f"share>={th}", lambda r, th=th: r["load"] < P["load_max"] and r["teacher_agrees"] == 1
                    and r["share_fam"] >= th and r["base_M_S"] >= P["ms_floor"]))
    out.append(("w_fam>=0.3", lambda r: r["load"] < P["load_max"] and r["teacher_agrees"] == 1
                and r["base_w_fam"] >= 0.3 and r["base_M_S"] >= P["ms_floor"]))
    for lt in cfg["secondary_load_thresholds"]:
        out.append((f"load<{lt}", lambda r, lt=lt: r["load"] < lt and r["teacher_agrees"] == 1
                    and r["share_fam"] >= P["share_min"] and r["base_M_S"] >= P["ms_floor"]))
    out.append(("all-load share>=0.5", lambda r: r["teacher_agrees"] == 1 and r["share_fam"] >= P["share_min"]
                and r["base_M_S"] >= P["ms_floor"]))
    out.append(("ALL constructed", lambda r: True))
    out.append(("opposing-dominant (share_opp>=0.5, low load)", lambda r: r["load"] < P["load_max"]
                and r["share_opp"] >= 0.5 and r["base_M_S"] >= P["ms_floor"]))
    return out


def diagnostics(rows, cfg):
    M = len([k for k in rows[0] if k.startswith("Pinert_m")])
    d = {"n": len(rows)}
    for m in range(M):
        v = [abs(r[f"Pinert_m{m:02d}"]) for r in rows]
        g = [r[f"gE_m{m:02d}"] for r in rows]
        vf = [r[f"Pinert_full_m{m:02d}"] for r in rows]
        d[f"m{m:02d}"] = {"max_abs_Pinert_target_vs_ref": max(v),
                          "frac_Pinert_gt_1e-5": sum(x > 1e-5 for x in v) / len(v),
                          "max_Pinert_nontarget_feasible": max(vf),
                          "gE_mean": sum(g) / len(g), "gE_pct_pos": 100.0 * sum(x > 1e-9 for x in g) / len(g),
                          "gE_pct_neg": 100.0 * sum(x < -1e-9 for x in g) / len(g)}
    for m in cfg["carriers"]:
        res = [abs(r[f"res_pred_m{m:02d}"]) for r in rows]
        d[f"carrier_m{m:02d}_max_abs_analytic_residual"] = max(res)
    d["max_abs_VPinert_fam"] = max(abs(r["VPinert_fam"]) for r in rows)
    d["max_abs_CPinert_fam"] = max(abs(r["CPinert_fam"]) for r in rows)
    return d


def check_batches(cfg, run_dirs, expected_fp, allow_unverified=False):
    funnels = [json.load(open(os.path.join(d, "funnel.json"))) for d in run_dirs]
    names = [os.path.basename(os.path.normpath(d)) for d in run_dirs]
    fp = C.common_fingerprint([f.get("fingerprint") for f in funnels], names, "batch outputs", allow_unverified)
    if fp is not None:
        C.check_fingerprint(fp, expected_fp, "the batch outputs")
    for name, f in zip(names, funnels):
        lo, hi = cfg["batches"][f["batch"]]
        got = list(f["instances"])
        if name != f"out_{f['batch']}" or got[0] != lo or not lo < got[1] <= hi:
            raise SystemExit(f"STOP: {name} holds batch {f['batch']} instances {got}, configured [{lo}, {hi})")
        if got[1] != hi:
            print(f"  NOTE: {name} covers instances [{lo}, {got[1]}) of the configured [{lo}, {hi}) (--max-instances)")


def analyze(cfg, run_dirs, out):
    cfg = dict(cfg); cfg.setdefault("null_tolerance", 1e-9)
    ROWS, EXTRA = table_rows(cfg)
    car, pid = cfg["carriers"], cfg["paper_id_of_bank"]
    os.makedirs(out, exist_ok=True)
    runs = {os.path.basename(os.path.normpath(p)): load_rows(os.path.join(p, "triplets.csv")) for p in run_dirs}
    pooled = [r for rs in runs.values() for r in rs]; runs["pooled"] = pooled
    P = cfg["primary"]
    res, md = {}, ["# corridor experiment (POMO CVRP-50 student, M=20; carriers "
                   + ", ".join(f"bank {m} = {pid[str(m)]}" for m in car) + ") -- g = P - T\n",
                   f"PRIMARY: load < {P['load_max']} AND teacher top-1 == a AND C_fam/M_S >= {P['share_min']} "
                   f"AND M_S >= {P['ms_floor']}\n"]
    base_keys = (("load", "base_M_S", "base_M_T", "share_fam", "base_w_fam")
                 + tuple(f"base_C_m{m:02d}" for m in car) + tuple(f"base_w_m{m:02d}" for m in car)
                 + (f"base_Vdepot_m{car[0]:02d}", "teacher_agrees", "alpha_deg", "rho", "chord"))
    for tag, rows in runs.items():
        res[tag] = {"_diagnostics_all_constructed": diagnostics(rows, cfg)}
        for pname, f in pops(cfg):
            pop = [r for r in rows if f(r)]
            if len(pop) < 8:
                continue
            blk = {}
            for key, lab in ROWS + EXTRA:
                d = describe(pop, key, cfg, seed=C.seed_of(tag, pname, key))
                if d:
                    blk[key] = {"label": lab, **d}
            blk["_base"] = {k: sum(r[k] for r in pop) / len(pop) for k in base_keys}
            blk["_base"]["abs_alpha_deg"] = sum(abs(r["alpha_deg"]) for r in pop) / len(pop)
            for lo, hi, dl in ((8, 14, "dose_lo"), (14, 22, "dose_mid"), (22, 30.1, "dose_hi")):
                sub = [r for r in pop if lo <= abs(r["alpha_deg"]) < hi]
                if len(sub) >= 8:
                    blk[dl] = {k: describe(sub, k, cfg, seed=C.seed_of(tag, pname, dl, k))
                               for k in ("g_E_fam", "g_L_S3", "g_L_T3", "g_M_T")}
            res[tag][pname] = blk
            md.append(f"\n## {tag} | {pname}   n = {blk['g_L_S3']['n']} / {blk['g_L_S3']['n_instances']} instances\n\n"
                      f"| row | pred. | mean (states) | 95% CI (states) | dir% (pred.) | mean (instances) | "
                      f"p (instance mean) |\n|---|---|---|---|---|---|---|")
            for key, lab in ROWS + EXTRA:
                v = blk[key]
                md.append(f"| {lab} | {v['predicted_sign'] or 'none'} | {v['mean_state']:+.3f} | "
                          f"[{v['ci95_state_lo']:+.3f}, {v['ci95_state_hi']:+.3f}] | {_pct(v)} | "
                          f"{v['mean_instance']:+.3f} | {v['p_signflip_instance_mean']:.2g} |")
            dl_txt = []
            for dl in ("dose_lo", "dose_mid", "dose_hi"):
                if dl in blk:
                    b2 = blk[dl]
                    dl_txt.append(f"{dl} n={b2['g_E_fam']['n']}: E {b2['g_E_fam']['mean_state']:+.3f}, "
                                  f"L_S3 {b2['g_L_S3']['mean_state']:+.3f}, L_T3 {b2['g_L_T3']['mean_state']:+.3f}, "
                                  f"M_T {b2['g_M_T']['mean_state']:+.3f}")
            if dl_txt:
                md.append("\n|alpha| terciles: " + "; ".join(dl_txt))
    C.write_json(os.path.join(out, "analysis_corridor.json"), res, indent=1)
    open(os.path.join(out, "FIVE_ROW_TABLE_corridor.md"), "w").write("\n".join(md) + "\n")
    for tag in runs:
        b = res[tag].get("PRIMARY")
        if not b:
            print(f"{tag}: PRIMARY too small"); continue
        print(f"\n=== {tag} PRIMARY  n={b['g_L_S3']['n']}/{b['g_L_S3']['n_instances']} inst ===")
        for key, lab in ROWS[:6] + ROWS[8:11] + EXTRA[:2]:
            v = b[key]
            print(f"  {lab:<44}{v['mean_state']:+.3f} [{v['ci95_state_lo']:+.3f},{v['ci95_state_hi']:+.3f}]  "
                  f"{_pct(v)} ({v['predicted_sign'] or 'none'})  inst-mean {v['mean_instance']:+.3f}  "
                  f"p={v['p_signflip_instance_mean']:.2g}")
    fv = {tag: {key: {k2: res[tag]["PRIMARY"][key][k2]
                      for k2 in ("label", "mean_state", "ci95_state_lo", "ci95_state_hi", "predicted_sign",
                                 "pct_predicted_direction", "pct_consistent_with_observed_sign", "mean_instance",
                                 "p_signflip_instance_mean", "n", "n_instances")}
                for key, _ in ROWS + EXTRA} for tag in runs if "PRIMARY" in res[tag]}
    C.write_json(os.path.join(out, "five_values.json"), fv)


def run(a, cfg):
    if a.inst_lo is not None or a.inst_hi is not None:
        raise SystemExit("corridor: the instance ranges are the config's batches (--batch, --max-instances)")
    if len(cfg["carriers"]) != 2:
        raise SystemExit("corridor: analytic_pred is written for the two carriers of the config")
    fp = C.experiment_fingerprint(NAME, PROTOCOL, ANALYSIS_KEYS, a, cfg)
    print(f"[fingerprint] {C.fingerprint_id(fp)}  {fp}")
    if not a.analyze:
        device = torch.device(a.device)
        torch.manual_seed(cfg["seed"])
        teacher, bank, student, sargs, info = load(cfg, a, device)
        print(f"student {info} tau_h {sargs['heuristic_temperature']} tau_r {sargs['routing_temperature']}", flush=True)
        for batch in ([a.batch] if a.batch else sorted(cfg["batches"])):
            run_batch(cfg, teacher, bank, student, info, device, batch, os.path.join(a.out, f"out_{batch}"),
                      a.max_instances, fp=fp)
        if a.batch:
            return
    runs = [os.path.join(a.out, f"out_{b}") for b in sorted(cfg["batches"])]
    runs = [d for d in runs if os.path.exists(os.path.join(d, "triplets.csv"))]
    if not runs:
        raise SystemExit(f"no OUT/out_<batch>/triplets.csv under {a.out}")
    check_batches(cfg, runs, fp, a.allow_unverified)
    analyze(cfg, runs, a.out)
