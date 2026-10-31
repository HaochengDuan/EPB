import collections
import hashlib
import json
import math
import os
import warnings
import zlib
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .. import settings as S

HERE = os.path.dirname(os.path.abspath(__file__))
STUDENTS = os.path.join(os.path.dirname(HERE), "students")


def student_files(student: str, bank: Optional[str] = None) -> Tuple[str, str]:
    if not os.path.exists(student) and os.path.isdir(os.path.join(STUDENTS, student)):
        student = os.path.join(STUDENTS, student)
    if os.path.isdir(student):
        ckpt = os.path.join(student, "best_model.pt")
        bank = bank or os.path.join(student, "bank.json")
    else:
        ckpt = student
    if not bank:
        raise ValueError(f"--student {student}: a checkpoint file needs --bank")
    for p in (ckpt, bank):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    return ckpt, bank


def guard_fallbacks() -> None:
    warnings.filterwarnings("error", message=".*runtime error.*")
    warnings.filterwarnings("error", message=".*out of (GPU )?memory.*")


def _eval_args(setting: str, device, **kw) -> SimpleNamespace:
    a = dict(ckpt=None, bank=None, teacher_ckpt=None, nco_root=None, tau_h=None, k_mode=None, v_mode=None, drop=[],
             enc_layers=None, device=str(device), search=S.get(setting).search_default, batch=None)
    a.update(kw)
    return SimpleNamespace(**a)


def load_student(setting: str, student: str, device, bank: Optional[str] = None, tau_h: Optional[float] = None):
    ckpt, bank_json = student_files(student, bank)
    policy = S.get(setting).eval_policy("epb", _eval_args(setting, device, ckpt=ckpt, bank=bank_json, tau_h=tau_h))
    guard_fallbacks()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    args = ck.get("args") or {}
    ck["args"] = args if isinstance(args, dict) else vars(args)
    return policy.model, policy.bank, ck, bank_json


def load_teacher(setting: str, ckpt: str, device, nco_root: Optional[str] = None):
    if not ckpt:
        raise ValueError("--teacher-ckpt is required")
    return S.get(setting).eval_policy("teacher", _eval_args(setting, device, teacher_ckpt=ckpt, nco_root=nco_root))


def bank_entries(bank_json: str) -> List[Dict]:
    hs = json.load(open(bank_json))["heuristics"]
    for i, h in enumerate(hs):
        if int(h.get("index", i)) != i:
            raise SystemExit(f"STOP: bank list slot {i} carries JSON index {h['index']} -- the slot mapping is not "
                             f"the identity, refusing to continue")
    return hs


def tsp_instances(seed: int, N: int, lo: int, hi: int, device) -> List[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    for _ in range(lo):
        torch.rand(N, 2, generator=g)
    return [torch.rand(N, 2, generator=g).to(device) for _ in range(hi - lo)]


LEHD_HELDOUT = dict(n_instances=100_000, problem_size=100, batch=256, seed=42, val_ratio=0.1, split_seed=42)
LEHD_HELDOUT_SHA256 = "27dd932f1526ef53e8333e05d464bc6b8d00ca2177809d03a4bc2dd4fa65feea"
LEHD_TOUR_BATCH = 1024


def _cuda_instances(device, n, problem_size, batch, seed) -> torch.Tensor:
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("the held-out instances are drawn from the CUDA generator: a CUDA device is required "
                           "(or pass a file written earlier with --heldout-cache)")
    idx = device.index if device.index is not None else torch.cuda.current_device()
    out, total = [], 0
    with torch.random.fork_rng(devices=[idx]):
        torch.manual_seed(seed)
        while total < n:
            bs = min(batch, n - total)
            out.append(torch.rand(bs, problem_size, 2, device=device).cpu())
            total += bs
    return torch.cat(out, dim=0)


def sha256_of(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def heldout_fingerprint(teacher_ckpt: str, upto: int) -> Dict:
    return fingerprint("lehd_heldout", HELDOUT_PROTOCOL, teacher=teacher_ckpt,
                       extra={"heldout": dict(LEHD_HELDOUT, sha256=LEHD_HELDOUT_SHA256),
                              "tour_batch": LEHD_TOUR_BATCH, "tour_instances": int(upto)})


@torch.no_grad()
def lehd_heldout(teacher_model, device, upto: int, cache: Optional[str] = None, fp: Optional[Dict] = None,
                 allow_unverified: bool = False) -> Dict[str, torch.Tensor]:
    if cache and os.path.exists(cache):
        held = torch.load(cache, map_location="cpu", weights_only=False)
        if fp is not None:
            check_fingerprint(held.get("fingerprint"), fp, f"held-out cache {cache}", allow_unverified)
        if held["instance_locs"].size(0) < upto:
            raise ValueError(f"{cache} holds {held['instance_locs'].size(0)} held-out instances, the run needs {upto}")
        return held
    from ..settings.lehd_tsp import split_train_val
    H = LEHD_HELDOUT
    locs = _cuda_instances(device, H["n_instances"], H["problem_size"], H["batch"], H["seed"])
    _, val_idx = split_train_val(H["n_instances"], H["val_ratio"], H["split_seed"])
    held_locs = locs.index_select(0, val_idx).contiguous()
    if sha256_of(held_locs) != LEHD_HELDOUT_SHA256:
        warnings.warn("held-out instances differ from the reference instance set (sha256 mismatch): the CUDA random "
                      "stream differs on this setup")
    upto = min(int(upto), held_locs.size(0))
    held_locs = held_locs[:upto].contiguous()
    held = {"val_idx": val_idx[:upto].clone(), "instance_locs": held_locs,
            "tour": lehd_greedy_tours(teacher_model, held_locs, device)}
    if fp is not None:
        held["fingerprint"] = fp
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)) or ".", exist_ok=True)
        torch.save(held, cache)
        print(f"  saved held-out instances + teacher tours to {cache}")
    return held


@torch.no_grad()
def pomo_tsp_greedy_tour(teacher, locs: torch.Tensor, first: int = 0) -> List[int]:
    from ..settings.pomo_tsp import construct
    tours, _ = construct(teacher, locs.unsqueeze(0), torch.full((1, 1), first, dtype=torch.long, device=locs.device))
    return tours[0, 0].tolist()


@torch.no_grad()
def lehd_greedy_tours(model, locs: torch.Tensor, device, batch_size: int = LEHD_TOUR_BATCH) -> torch.Tensor:
    from ..settings.lehd_tsp import lehd_step
    out = []
    for b0 in range(0, locs.size(0), batch_size):
        x = locs[b0:b0 + batch_size].to(device)
        encoded = model.encoder(x)
        selected = torch.zeros(x.size(0), 1, dtype=torch.long, device=device)
        for _ in range(1, x.size(1)):
            nxt_local, unsel_idx = lehd_step(model, encoded, selected)
            selected = torch.cat([selected, unsel_idx.gather(1, nxt_local.unsqueeze(1))], dim=1)
        out.append(selected.cpu())
    return torch.cat(out)


def lehd_prefix_states(held: Dict[str, torch.Tensor], inst_ids: List[int], step: int, device) -> Dict:
    N = held["instance_locs"].size(1)
    ids = torch.tensor(inst_ids, dtype=torch.long)
    locs = held["instance_locs"].index_select(0, ids).float().to(device)
    tours = held["tour"].index_select(0, ids).long().to(device)
    B, K = len(inst_ids), N - step
    prefix = tours[:, :step]
    nl = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
    bidx = torch.arange(B, device=device).unsqueeze(1).expand(B, step)
    nl[bidx, prefix] = -2
    unsel_idx = nl[nl >= 0].view(B, K)
    return {"inst_ids": inst_ids, "step": step, "K": K, "B": B, "locs": locs,
            "prefix": prefix, "unsel_idx": unsel_idx,
            "unsel_locs": locs.gather(1, unsel_idx.unsqueeze(-1).expand(B, K, 2)),
            "first_loc": locs.gather(1, tours[:, 0:1].unsqueeze(-1).expand(B, 1, 2)).squeeze(1),
            "current_loc": locs.gather(1, tours[:, step - 1:step].unsqueeze(-1).expand(B, 1, 2)).squeeze(1)}


def lehd_sub_batch(unsel_locs, first_loc, current_loc, unsel_idx, N: int) -> Dict:
    mask = torch.zeros(unsel_locs.size(0), N, dtype=torch.bool, device=unsel_locs.device)
    mask.scatter_(1, unsel_idx, True)
    return {"mask": mask, "unselected_idx": unsel_idx, "unselected_locs": unsel_locs, "first_loc": first_loc,
            "current_loc": current_loc}


@torch.no_grad()
def lehd_teacher_probs(model, locs: torch.Tensor, selected: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    from ..settings.lehd_tsp import lehd_step
    return lehd_step(model, model.encoder(locs), selected, return_probs=True)


CVRP_EPS_FEAS = 1e-5
CVRP_DEMAND_SCALER = {20: 30, 50: 40, 100: 50}


def cvrp_instances(seed: int, lo: int, hi: int, batch: int, N: int):
    assert lo % batch == 0 and hi % batch == 0, "instance bounds must be multiples of the rollout batch"
    g = torch.Generator().manual_seed(seed)
    done = 0
    while done < hi:
        bs = min(batch, hi - done)
        depot_xy = torch.rand(size=(bs, 1, 2), generator=g)
        node_xy = torch.rand(size=(bs, N, 2), generator=g)
        node_demand = torch.randint(1, 10, size=(bs, N), generator=g) / float(CVRP_DEMAND_SCALER[N])
        if done >= lo:
            yield depot_xy, node_xy, node_demand
        done += bs


def cvrp_keep_hash(instance_ids: torch.Tensor, P: int, step: int, keep_mod: int, device) -> torch.Tensor:
    inst = instance_ids.to(device=device, dtype=torch.long)[:, None]
    pom = torch.arange(P, device=device, dtype=torch.long)[None, :]
    return ((inst * 7919 + pom * 104729 + step * 31) % keep_mod) == 0


@torch.no_grad()
def cvrp_collect_states(teacher, cfg, device, batches, id_offset: int, keep_mod: Optional[int] = None,
                        on_batch=None):
    N, P = cfg["problem_size"], cfg["pomo_size"]
    num_nodes = N + 1
    states = []
    n_seen = 0
    done_inst = 0
    for depot_xy, node_xy, node_demand in batches:
        batch_states = []
        bs = depot_xy.size(0)
        depot_xy, node_xy, node_demand = (depot_xy.to(device), node_xy.to(device), node_demand.to(device))
        all_locs = torch.cat([depot_xy, node_xy], dim=1)
        all_demand = torch.cat([torch.zeros(bs, 1, device=device), node_demand], 1)
        teacher.pre_forward(depot_xy, node_xy, node_demand)
        locs_cpu = all_locs.cpu()
        dem_cpu = all_demand.cpu()
        inst_ids = torch.arange(bs, dtype=torch.long) + (id_offset + done_inst)

        load = torch.ones(bs, P, device=device)
        visited_flag = torch.zeros(bs, P, num_nodes, device=device)
        visited_ninf = torch.zeros(bs, P, num_nodes, device=device)
        finished = torch.zeros(bs, P, dtype=torch.bool, device=device)
        bidx = torch.arange(bs, device=device)[:, None].expand(bs, P)
        pidx = torch.arange(P, device=device)[None, :].expand(bs, P)
        demand_BP = all_demand[:, None, :].expand(bs, P, num_nodes)

        first_customer = torch.arange(1, P + 1, device=device)[None, :].expand(bs, P)
        current_node = first_customer.clone()
        first_tracker = first_customer.clone()
        load -= demand_BP.gather(2, current_node[:, :, None]).squeeze(2)
        visited_flag[bidx, pidx, current_node] = 1.0
        visited_ninf[bidx, pidx, current_node] = float("-inf")
        at_depot = (current_node == 0)
        visited_ninf[:, :, 0][~at_depot] = 0
        ninf_mask = visited_ninf.clone()
        ninf_mask[load[:, :, None] + CVRP_EPS_FEAS < demand_BP] = float("-inf")
        finished |= (visited_ninf == float("-inf")).all(dim=2)
        ninf_mask[:, :, 0][finished] = 0

        step = 2
        max_steps = N * 3
        while not finished.all() and step < max_steps:
            sel, _, _ = teacher(current_node, load, ninf_mask)
            feas = (ninf_mask == 0)
            n_feas_cust = feas[:, :, 1:].sum(-1)
            keep = ((~finished) & (current_node != 0)
                    & (load > 0) & (load <= cfg["load_max"])
                    & (n_feas_cust >= cfg["n_feasible_min"]))
            n_seen += int(keep.sum())
            if keep_mod is not None:
                keep = keep & cvrp_keep_hash(inst_ids, P, step, keep_mod, device)
            if keep.any():
                bi, pi = keep.nonzero(as_tuple=True)
                vf = visited_flag[bi, pi].cpu()
                mk = feas[bi, pi].float().cpu()
                cn = current_node[bi, pi].cpu()
                fn = first_tracker[bi, pi].cpu()
                ld = load[bi, pi].cpu()
                bi_c = bi.cpu()
                pi_c = pi.cpu()
                for k in range(bi.numel()):
                    b_ = int(bi_c[k])
                    batch_states.append({
                        "instance_id": id_offset + done_inst + b_,
                        "pomo": int(pi_c[k]), "step_i": step,
                        "locs": locs_cpu[b_], "demand": dem_cpu[b_],
                        "visited": vf[k], "mask": mk[k],
                        "current_node": int(cn[k]), "first_node": int(fn[k]),
                        "load": float(ld[k]), "load_t": ld[k:k + 1].reshape(()),
                    })
            current_node = sel
            at_depot = (sel == 0)
            load -= demand_BP.gather(2, sel[:, :, None]).squeeze(2)
            load[at_depot] = 1.0
            visited_ninf[bidx, pidx, sel] = float("-inf")
            visited_ninf[:, :, 0][~at_depot] = 0
            visited_flag[bidx, pidx, sel] = 1.0
            visited_flag[:, :, 0] = 0
            ninf_mask = visited_ninf.clone()
            ninf_mask[load[:, :, None] + CVRP_EPS_FEAS < demand_BP] = float("-inf")
            finished |= (visited_ninf == float("-inf")).all(dim=2)
            ninf_mask[:, :, 0][finished] = 0
            step += 1
        done_inst += bs
        states.extend(on_batch(batch_states) if on_batch is not None else batch_states)
        del batch_states
    return states, n_seen


def cvrp_feasibility_mask(demand, visited, current_node, load) -> torch.Tensor:
    n = demand.numel()
    ok = torch.ones(n, dtype=torch.bool)
    ok[1:] = visited[1:] < 0.5
    ok &= ~(demand > load + CVRP_EPS_FEAS)
    ok[0] = current_node != 0
    return ok


def cvrp_stack_states(states, device, demand_key: str = "demand"):
    locs = torch.stack([s["locs"] for s in states]).to(device)
    dem = torch.stack([s[demand_key] for s in states]).to(device)
    vis = torch.stack([s["visited"] for s in states]).to(device)
    cur = torch.tensor([s["current_node"] for s in states], device=device, dtype=torch.long)
    fst = torch.tensor([s["first_node"] for s in states], device=device, dtype=torch.long)
    load = torch.stack([s["load_t"] for s in states]).to(device).view(-1).float()
    mask = torch.stack([cvrp_feasibility_mask(s[demand_key], s["visited"], s["current_node"], s["load"])
                        for s in states]).to(device)
    return {"locs": locs, "demand": dem, "visited": vis, "mask": mask.float(),
            "current_node": cur, "first_node": fst, "load": load}, mask


@torch.no_grad()
def cvrp_raw_scores(bank, b, mask) -> torch.Tensor:
    args = (b["locs"], b["demand"], b["current_node"], mask, b["visited"], b["load"], b["first_node"])
    return torch.stack([p.fn(*[x.clone() for x in args]).float() for p in bank.programs], dim=1)


@torch.no_grad()
def cvrp_teacher_probs(teacher, b, mask) -> torch.Tensor:
    B = b["locs"].shape[0]
    teacher.pre_forward(b["locs"][:, :1, :], b["locs"][:, 1:, :], b["demand"][:, 1:])
    ninf = torch.zeros(B, 1, b["demand"].shape[1], device=b["locs"].device)
    ninf[:, 0][~mask] = float("-inf")
    enc = teacher.encoded_nodes.gather(1, b["current_node"][:, None, None].expand(B, 1, teacher.encoded_nodes.size(2)))
    piT, _ = teacher.decoder(enc, b["load"][:, None], ninf)
    return piT[:, 0]


def rotate2(v, theta: float):
    c, s = math.cos(theta), math.sin(theta)
    if torch.is_tensor(v):
        return torch.stack([c * v[0] - s * v[1], s * v[0] + c * v[1]])
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def polar_angle(v) -> float:
    return math.atan2(float(v[1]), float(v[0]))


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def signed_angle(u, v) -> float:
    return math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])


FINGERPRINT_VERSION = 1
HELDOUT_PROTOCOL = 1


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def student_sha256(ckpt: str) -> str:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k]
        if torch.is_tensor(t):
            t = t.detach().cpu().contiguous()
            h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(f"{k}|{t!r}|".encode())
    args = ck.get("args") or {}
    args = args if isinstance(args, dict) else vars(args)
    h.update(json.dumps(args, sort_keys=True, default=str).encode())
    return h.hexdigest()


def bank_sha256(bank_json: str) -> str:
    hs = json.load(open(bank_json))["heuristics"]
    return _sha256(json.dumps([h["code"] for h in hs]).encode())


def config_sha256(cfg: Dict, exclude=()) -> str:
    keep = {k: v for k, v in cfg.items() if not k.startswith("_") and k not in exclude}
    return _sha256(json.dumps(keep, sort_keys=True).encode())


def fingerprint(test: str, protocol: int, student: Optional[str] = None, bank: Optional[str] = None,
                teacher: Optional[str] = None, cfg: Optional[Dict] = None, exclude=(),
                extra: Optional[Dict] = None) -> Dict:
    fp = {"version": FINGERPRINT_VERSION, "test": test, "protocol": int(protocol)}
    if student:
        fp["student"] = student_sha256(student)
    if bank:
        fp["bank"] = bank_sha256(bank)
    if teacher:
        fp["teacher"] = file_sha256(teacher)
    if cfg is not None:
        fp["config"] = config_sha256(cfg, exclude)
    if extra:
        fp.update(extra)
    return fp


def experiment_fingerprint(name: str, protocol: int, analysis_keys, a, cfg: Dict) -> Dict:
    ckpt, bank_json = student_files(a.student, a.bank)
    return fingerprint(name, protocol, student=ckpt, bank=bank_json, teacher=a.teacher_ckpt or None,
                       cfg=cfg, exclude=analysis_keys)


def fingerprint_id(fp: Optional[Dict]) -> str:
    return _sha256(json.dumps(fp, sort_keys=True).encode())[:12] if fp else "none"


def check_fingerprint(found: Optional[Dict], expected: Dict, what: str, allow_unverified: bool = False) -> None:
    if not found:
        if allow_unverified:
            print(f"  WARNING: {what} carries no experiment fingerprint; used unverified (--allow-unverified)")
            return
        raise SystemExit(f"STOP: {what} carries no experiment fingerprint, so it cannot be matched to this run. "
                         f"Re-run it, or pass --allow-unverified to use it unchecked.")
    diff = [k for k in expected if found.get(k) != expected[k]]
    if diff:
        raise SystemExit(f"STOP: experiment mismatch for {what}; differing components: "
                         + "; ".join(f"{k}: {found.get(k)!r} (found) vs {expected[k]!r} (expected)" for k in diff))


def common_fingerprint(fps: List[Optional[Dict]], names: List[str], what: str,
                       allow_unverified: bool = False) -> Optional[Dict]:
    have = [f for f in fps if f]
    if not have:
        if allow_unverified:
            print(f"  WARNING: none of the {len(fps)} {what} carries an experiment fingerprint; merged unverified "
                  f"(--allow-unverified)")
            return None
        raise SystemExit(f"STOP: the {what} carry no experiment fingerprint, so they cannot be matched. Re-run them, "
                         f"or pass --allow-unverified to merge them unchecked.")
    if len(have) != len(fps):
        missing = [n for n, f in zip(names, fps) if not f]
        raise SystemExit(f"STOP: some {what} carry no experiment fingerprint: {missing}")
    for n, f in zip(names, fps):
        if f != fps[0]:
            diff = sorted(k for k in set(f) | set(fps[0]) if f.get(k) != fps[0].get(k))
            raise SystemExit(f"STOP: {what} {n} and {names[0]} come from different experiments (differing "
                             f"components: {', '.join(diff)})")
    return fps[0]


def check_tiling(ranges: List[Tuple[int, int, str]], lo: int, hi: int, what: str, complete: bool = True) -> int:
    if not ranges:
        raise SystemExit(f"STOP: no {what} found")
    pos = lo
    for a, b, name in ranges:
        if b <= a:
            raise SystemExit(f"STOP: {what} {name} has an empty range [{a}, {b})")
        if a != pos:
            kind = "gap" if a > pos else "overlap"
            raise SystemExit(f"STOP: {what} {kind}: {name} starts at {a}, the covered range ends at {pos} "
                             f"(the {what} must tile [{lo}, {hi}) contiguously)")
        pos = b
    if pos > hi:
        raise SystemExit(f"STOP: {what} reach {pos}, past the end of the target range [{lo}, {hi})")
    if complete and pos != hi:
        raise SystemExit(f"STOP: {what} cover [{lo}, {pos}) of the target range [{lo}, {hi}); the rest is missing")
    return pos


SCHEMES = ("hull", "corridor", "isolation", "demand")


def seed_of(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode()) % (2 ** 31 - 1)


def clusters(vals, ids) -> List[List[float]]:
    by = collections.OrderedDict()
    for v, i in zip(vals, ids):
        by.setdefault(i, []).append(v)
    return list(by.values())


def _np_finite(vals, ids):
    vals, ids = np.asarray(vals, float), np.asarray(ids)
    ok = np.isfinite(vals)
    return vals[ok], ids[ok]


def _np_cluster_means(vals, ids) -> np.ndarray:
    _, inv = np.unique(ids, return_inverse=True)
    return np.array([vals[inv == g].mean() for g in range(inv.max() + 1)])


def cluster_mean(vals, ids) -> float:
    vals, ids = _np_finite(vals, ids)
    return float(_np_cluster_means(vals, ids).mean())


def instance_mean(vals, ids, scheme: str) -> float:
    if scheme in ("hull", "corridor"):
        cl = clusters(vals, ids)
        if not cl:
            return float("nan")
        return float(torch.tensor([sum(c) / len(c) for c in cl], dtype=torch.float64).mean())
    if scheme == "isolation":
        return cluster_mean(vals, ids)
    if scheme == "demand":
        vals, ids = np.asarray(vals, float), np.asarray(ids)
        u, inv = np.unique(ids, return_inverse=True)
        return float((np.bincount(inv, weights=vals) / np.bincount(inv)).mean())
    raise ValueError(f"unknown statistics scheme {scheme!r}; one of {SCHEMES}")


def bootstrap_ci(vals, ids, scheme: str, n_boot: int, seed: int = 0, rng: Optional[np.random.Generator] = None,
                 alpha: float = 0.05) -> Tuple[float, float]:
    if scheme in ("hull", "corridor"):
        cl = clusters(vals, ids)
        K = len(cl)
        g = torch.Generator().manual_seed(seed)
        if scheme == "hull":
            idx = torch.randint(0, K, (n_boot, K), generator=g).tolist()
            means = []
            for row in idx:
                pool = [x for j in row for x in cl[j]]
                means.append(sum(pool) / len(pool))
            t = torch.tensor(means)
            return float(torch.quantile(t, alpha / 2)), float(torch.quantile(t, 1 - alpha / 2))
        if K < 2:
            return float("nan"), float("nan")
        s = torch.tensor([float(sum(c)) for c in cl], dtype=torch.float64)
        n = torch.tensor([float(len(c)) for c in cl], dtype=torch.float64)
        idx = torch.randint(0, K, (n_boot, K), generator=g)
        mu = s[idx].sum(1) / n[idx].sum(1)
        return float(torch.quantile(mu, alpha / 2)), float(torch.quantile(mu, 1 - alpha / 2))
    if scheme == "isolation":
        vals, ids = _np_finite(vals, ids)
        uniq, inv = np.unique(ids, return_inverse=True)
        groups = [np.flatnonzero(inv == k) for k in range(len(uniq))]
        r = np.random.default_rng(seed)
        out = np.empty(n_boot)
        for b in range(n_boot):
            pick = r.integers(0, len(groups), len(groups))
            out[b] = np.mean(vals[np.concatenate([groups[p] for p in pick])])
        return float(np.percentile(out, 100 * alpha / 2)), float(np.percentile(out, 100 * (1 - alpha / 2)))
    if scheme == "demand":
        u, inv = np.unique(ids, return_inverse=True)
        s, n, K = np.bincount(inv, weights=vals), np.bincount(inv), len(u)
        out = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, K, K)
            out[b] = s[pick].sum() / n[pick].sum()
        return float(np.percentile(out, 100 * alpha / 2)), float(np.percentile(out, 100 * (1 - alpha / 2)))
    raise ValueError(f"unknown statistics scheme {scheme!r}; one of {SCHEMES}")


def signflip_p(vals, ids, scheme: str, n_perm: int, seed: int = 1, rng: Optional[np.random.Generator] = None) -> float:
    if scheme in ("hull", "corridor"):
        cm = torch.tensor([sum(c) / len(c) for c in clusters(vals, ids)], dtype=torch.float64)
        K = len(cm)
        if scheme == "corridor" and K < 2:
            return float("nan")
        obs = abs(float(cm.mean()))
        g = torch.Generator().manual_seed(seed)
        s = torch.randint(0, 2, (n_perm, K), generator=g).double() * 2 - 1
        stat = (s * cm).mean(dim=1).abs()
        if scheme == "hull":
            return float((stat >= obs).sum().item() + 1) / (n_perm + 1)
        return (int((stat >= obs - 1e-15).sum()) + 1) / (n_perm + 1)
    if scheme == "isolation":
        vals, ids = _np_finite(vals, ids)
        cm = _np_cluster_means(vals, ids)
        obs = cm.mean()
        r = np.random.default_rng(seed)
        signs = r.choice([-1.0, 1.0], size=(n_perm, len(cm)))
        null = (signs * cm[None, :]).mean(axis=1)
        return max(float((np.abs(null) >= abs(obs)).mean()), 1.0 / n_perm)
    if scheme == "demand":
        u, inv = np.unique(ids, return_inverse=True)
        s, K = np.bincount(inv, weights=vals), len(u)
        obs = abs(vals.mean())
        sg = rng.integers(0, 2, (n_perm, K)) * 2 - 1
        return (int((np.abs((sg * s).sum(1) / len(vals)) >= obs - 1e-15).sum()) + 1) / (n_perm + 1)
    raise ValueError(f"unknown statistics scheme {scheme!r}; one of {SCHEMES}")


def write_json(path: str, obj, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
    print(f"  wrote {path}")


def write_csv(path: str, rows: List[Dict], keys: List[str]) -> None:
    import csv
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for d in rows:
            w.writerow({k: d.get(k) for k in keys})
    os.replace(tmp, path)
