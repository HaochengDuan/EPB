"""Evaluate a policy on a fixed instance set: teacher-forced top-1, greedy cost and search cost.

  python -m epb.eval --setting pomo_cvrp --n 50 --method epb --ckpt epb/students/pomo_cvrp50 \\
      --teacher-ckpt <POMO_CVRP50_CKPT> --metrics top1,greedy,search
  python -m epb.eval --setting pomo_cvrp --n 50 --method teacher --teacher-ckpt <POMO_CVRP50_CKPT>
  python -m epb.eval --setting lehd_tsp --instances n500 --method epb --ckpt epb/students/lehd_tsp100 \\
      --nco-root <NCO_code clone> --slice 0:32
  python -m epb.eval --table results/

Methods: teacher (--teacher-ckpt), epb / expert / frozen (--ckpt: a directory with best_model.pt, bank.json and
decode.json, the decoding options of the paper, or an epb.train run directory with best_model.pt,
heuristics_best.json and the decode.json the run wrote; a flag overrides decode.json), nam / moet / tree / agent
(--pkg: the model directory under epb/baselines; tree also --depth). Search: POMO settings take the best of all N
first nodes (ms), LEHD settings run random re-construct (rrc) for --budget iterations. top-1 needs --teacher-ckpt
for every method.

Every run writes <out>/<setting><N>[_<instances>]/<method>[_<tag>][@lo-hi].pt (per-instance costs and top-1 counts)
and a .json summary. --table merges slices, computes the gaps (ratio of means against the teacher on the same
instances) and the top-1 rates with instance-clustered bootstrap CIs, and writes table.md / table.json.
From Python, build_args(argv) resolves a command line the same way for a setting's eval_policy / eval_greedy /
eval_search / eval_top1.
"""
import argparse
import glob
import json
import os
import re
import time
from typing import Dict, List

import torch

from . import instances as I
from . import settings as S

METHODS = ("teacher", "epb", "expert", "frozen", "nam", "moet", "tree", "agent")
METRICS = ("top1", "greedy", "search")
BOOT_REPS, BOOT_SEED = 2000, 0


def parser(setting=None) -> argparse.ArgumentParser:
    """The command line of epb.eval; with a setting (epb.settings.get(name)), also that setting's evaluation flags."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", metavar="DIR", help="merge the results under DIR into tables and exit")
    ap.add_argument("--setting", choices=S.NAMES)
    ap.add_argument("--n", type=int, help="problem size of the main-table set (main) / check for other sets")
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--ckpt", help="router-bank method: directory (best_model.pt, bank.json, decode.json) or .pt")
    ap.add_argument("--bank", help="bank.json (default: next to --ckpt)")
    ap.add_argument("--pkg", help="nam / moet / tree / agent: model directory")
    ap.add_argument("--depth", help="tree: depth variant (d12 | staged)")
    ap.add_argument("--teacher-ckpt", help="teacher checkpoint (POMO: yd-kwon/POMO format; LEHD: NCO_code format)")
    ap.add_argument("--nco-root", help="clone of CIAM-Group/NCO_code (LEHD settings)")
    ap.add_argument("--instances", default="main", help="main | n500 | band:<set> | file:<path>")
    ap.add_argument("--slice", help="lo:hi, evaluate a contiguous slice of the set (shards)")
    ap.add_argument("--metrics", default="greedy,search", help=f"comma list of {METRICS}")
    ap.add_argument("--search", choices=("ms", "rrc"), help="default: the setting's (POMO ms, LEHD rrc)")
    ap.add_argument("--budget", type=int, help="RRC iterations (default: the main table's)")
    ap.add_argument("--k-mode", choices=("soft", "onehot"))
    ap.add_argument("--v-mode", choices=("soft", "onehot"))
    ap.add_argument("--tau-h", type=float, help="program temperature (default: decode.json, then the checkpoint's "
                                                 "heuristic_temperature, then the setting's)")
    ap.add_argument("--state-convention", choices=S.STATE_CONVENTIONS,
                    help="decision states of the router / bank: train (the generator's rows) or legacy (the evaluator "
                         "of the released results); default: decode.json, then the checkpoint, then legacy")
    ap.add_argument("--drop", type=lambda s: [int(x) for x in s.split(",") if x != ""], help="bank slots to drop")
    ap.add_argument("--batch", type=int, help="instances per batch (default: the setting's)")
    ap.add_argument("--device", help="default cuda (decode.json may say otherwise)")
    ap.add_argument("--threads", type=int, help="torch CPU threads")
    ap.add_argument("--tag", default="", help="suffix of the result file name")
    ap.add_argument("--out", default="results")
    if setting is not None:
        setting.eval_args(ap)
    return ap


BANK_FILES = ("bank.json", "heuristics_best.json")     # the bank next to best_model.pt: released dir, epb.train run


def resolve_ckpt(a) -> None:
    """--ckpt DIR -> DIR/best_model.pt and DIR/bank.json (a released student) or DIR/heuristics_best.json (an
    epb.train run directory). The defaults of decode.json (in the --ckpt or --pkg directory) fill the unset flags; an
    entry named after the setting (e.g. "lehd_tsp") and then one named after the instance set (e.g. "n500") override
    the top-level values. A checkpoint that records its bank's sha (epb.train) must match the bank: an error for the
    bank found next to it, a warning for an explicit --bank (an edited bank on a trained router)."""
    d = None
    explicit_bank = getattr(a, "bank", None) is not None
    if a.ckpt and os.path.isdir(a.ckpt):
        d = a.ckpt
        a.ckpt = os.path.join(d, "best_model.pt")
        if a.bank is None:
            a.bank = next((os.path.join(d, f) for f in BANK_FILES if os.path.exists(os.path.join(d, f))), None)
    elif a.pkg and os.path.isdir(a.pkg):
        d = a.pkg
    dec = os.path.join(d, "decode.json") if d else None
    if dec and os.path.exists(dec):
        opts = json.load(open(dec))
        opts = {**{k: v for k, v in opts.items() if not isinstance(v, dict)}, **opts.get(a.setting, {}),
                **opts.get(a.instances, {})}
        for k, v in opts.items():
            if getattr(a, k, None) is None:
                setattr(a, k, v)
    if a.device is None:
        a.device = "cuda"
    if a.ckpt and a.bank and os.path.isfile(a.ckpt) and os.path.isfile(a.bank):
        check_bank_pairing(a.ckpt, a.bank, strict=not explicit_bank)


def check_bank_pairing(ckpt: str, bank: str, strict: bool = True) -> None:
    """The checkpoint's bank_sha (written by epb.train) against the sha of the bank's programs; nothing to check for a
    checkpoint without one (the released students)."""
    import warnings
    from .bank import bank_file_sha
    want = torch.load(ckpt, map_location="cpu", weights_only=False).get("bank_sha")
    if not want:
        return
    got = bank_file_sha(bank)
    if got != want:
        msg = (f"{bank} (bank sha {got}) is not the bank {ckpt} was trained with (bank_sha {want}): the router's "
               f"programs differ from its training bank")
        if strict:
            raise ValueError(msg + "; pass the matching bank (or --bank to evaluate this router with another bank)")
        warnings.warn(msg + " (--bank given explicitly: evaluating anyway)")


def build_args(argv: List[str]) -> argparse.Namespace:
    """The arguments of an epb.eval command line given as a list (with --setting and --method), resolved as a run
    resolves them: parsed with the setting's flags, --ckpt / --pkg resolved by resolve_ckpt (decode.json fills the
    unset flags) and the search set to the setting's default unless given. For driving a setting's evaluation
    interface from Python:

      a = build_args(["--setting", "pomo_tsp", "--method", "epb", "--ckpt", "epb/students/pomo_tsp100"])
      setting = epb.settings.get(a.setting)
      policy = setting.eval_policy(a.method, a)        # then setting.eval_greedy(policy, instances, a), ...
    """
    first = parser().parse_known_args(argv)[0]
    if not first.setting or not first.method:
        raise ValueError("build_args: the command line needs --setting and --method")
    setting = S.get(first.setting)
    a = parser(setting).parse_args(argv)
    resolve_ckpt(a)
    a.search = a.search or setting.search_default
    return a


def result_dir(out: str, setting: str, n: int, instances: str) -> str:
    inst = instances
    if inst.startswith("file:"):
        inst = os.path.splitext(os.path.basename(inst[5:]))[0]
    inst = "" if inst == "main" else "_" + re.sub(r"[^A-Za-z0-9]+", "-", inst).strip("-")
    return os.path.join(out, f"{setting}{n}{inst}")


def _out_path(a, n: int) -> str:
    d = result_dir(a.out, a.setting, n, a.instances)
    os.makedirs(d, exist_ok=True)
    name = a.method + (f"_{a.tag}" if a.tag else "") + (f"@{a.slice.replace(':', '-')}" if a.slice else "")
    return os.path.join(d, name + ".pt")


def run(a) -> Dict:
    setting = S.get(a.setting)
    if a.threads:
        torch.set_num_threads(a.threads)
    resolve_ckpt(a)
    metrics = [m for m in a.metrics.split(",") if m]
    bad = [m for m in metrics if m not in METRICS]
    if bad:
        raise ValueError(f"unknown metrics {bad}; choose from {METRICS}")
    if "top1" in metrics and not a.teacher_ckpt:
        raise ValueError("top1 needs --teacher-ckpt")
    a.search = a.search or setting.search_default
    if a.method == "agent" and not a.tag and a.pkg:
        a.tag = os.path.basename(os.path.normpath(a.pkg))      # agent_fable51, agent_gpt6
    sl = tuple(int(x) for x in a.slice.split(":")) if a.slice else None
    inst, sha = I.load(a.instances, a.setting, a.n, sl)
    first = next(iter(inst.values()))
    n_inst, N = first.size(0), (inst["locs"] if "locs" in inst else inst["node_xy"]).size(1)
    if a.n is not None and a.n != N:
        raise ValueError(f"--n {a.n} but the instances have N={N}")
    if a.budget is None and a.search == "rrc":
        a.budget = setting.budget_default.get(N)
        if a.budget is None:
            raise ValueError(f"no default RRC budget for N={N}; pass --budget")
    t0 = time.time()
    policy = setting.eval_policy(a.method, a)
    rec = {"setting": a.setting, "N": N, "method": a.method, "tag": a.tag, "instances": a.instances,
           "instances_sha256": sha, "slice": list(sl) if sl else None, "n": n_inst,
           "search_name": f"{a.search}-{a.budget}" if a.search == "rrc" else a.search,
           "identity": _identity(a, metrics),
           "config": {k: v for k, v in vars(a).items() if k not in ("table",)}}
    if "greedy" in metrics:
        rec["greedy"] = setting.eval_greedy(policy, inst, a).double().cpu()
    if "search" in metrics:
        rec["search"] = setting.eval_search(policy, inst, a).double().cpu()
    if "top1" in metrics:
        teacher = policy if a.method == "teacher" else setting.eval_policy("teacher", a)
        rec["top1"] = {k: v.double().cpu() for k, v in setting.eval_top1(teacher, policy, inst, a).items()}
    rec["seconds"] = time.time() - t0
    path = _out_path(a, N)
    torch.save(rec, path)
    summary = _summary(rec)
    json.dump(summary, open(path[:-3] + ".json", "w"), indent=1)
    print(json.dumps(summary), flush=True)
    return rec


def _file_sha(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _identity(a, metrics: List[str]) -> Dict:
    """Content hashes of what defines the evaluated model: router checkpoint, bank, teacher (when used), package."""
    ident = {}
    for k in ("ckpt", "bank"):
        p = getattr(a, k, None)
        ident[k] = _file_sha(p) if p and os.path.isfile(p) else None
    uses_teacher = a.method == "teacher" or "top1" in metrics
    t = getattr(a, "teacher_ckpt", None)
    ident["teacher_ckpt"] = _file_sha(t) if uses_teacher and t and os.path.isfile(t) else None
    pkg = getattr(a, "pkg", None)
    ident["pkg"] = os.path.basename(os.path.normpath(pkg)) if pkg else None
    return ident


def _summary(rec: Dict) -> Dict:
    s = {k: rec[k] for k in ("setting", "N", "method", "tag", "instances", "slice", "n", "search_name")}
    for k in ("greedy", "search"):
        if k in rec:
            s[f"{k}_mean"] = rec[k].mean().item()
    if "top1" in rec:
        t = rec["top1"]
        s["top1"] = (t["hits"].sum() / t["states"].sum()).item()
        s["top1_all"] = (t["hits_all"].sum() / t["states_all"].sum()).item()
    s["seconds"] = round(rec["seconds"], 1)
    return s


# ═══════════════════════════════════════════════════════════════════════════════════════════════ tables

def bootstrap_ci(hits: torch.Tensor, states: torch.Tensor, reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> List[float]:
    """Instance-clustered bootstrap of the pooled hit fraction sum(hits) / sum(states) (per-instance counts)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, hits.numel(), (reps, hits.numel()), generator=g)
    f = hits.double()[idx].sum(1) / states.double()[idx].sum(1)
    return torch.quantile(f, torch.tensor([0.025, 0.975], dtype=torch.double)).tolist()


# What must agree between the slices of one result: the record fields, the content hashes of `identity` and these
# resolved arguments (decoding options, batch layout, device, search). Paths (ckpt, bank, pkg, teacher_ckpt) are
# compared only for records written before `identity` existed.
FINGERPRINT_RECORD = ("setting", "N", "method", "tag", "instances", "instances_sha256", "search_name")
FINGERPRINT_CONFIG = ("depth", "k_mode", "v_mode", "tau_h", "drop", "enc_layers", "batch", "ms_chunk",
                      "starts_per_chunk", "device", "search", "budget", "rrc_seed", "sub_min", "state_convention")
FINGERPRINT_PATHS = ("ckpt", "bank", "pkg", "teacher_ckpt")


def fingerprint(rec: Dict) -> Dict:
    """The experiment a result record belongs to (everything but the slice): see FINGERPRINT_*."""
    cfg = rec.get("config") or {}
    fp = {k: rec.get(k) for k in FINGERPRINT_RECORD}
    fp.update({f"config.{k}": cfg.get(k) for k in FINGERPRINT_CONFIG})
    if "identity" in rec:
        fp.update({f"identity.{k}": v for k, v in rec["identity"].items()})
    else:
        fp.update({f"config.{k}": cfg.get(k) for k in FINGERPRINT_PATHS})
    fp["metrics"] = sorted(k for k in METRICS if k in rec)
    if "top1" in rec:
        fp["top1_keys"] = sorted(rec["top1"])
    return fp


def _check_lengths(rec: Dict, where: str) -> None:
    n = rec["n"]
    if rec["slice"] is not None and rec["slice"][1] - rec["slice"][0] != n:
        raise ValueError(f"{where}: slice {rec['slice']} but n = {n}")
    for k in ("greedy", "search"):
        if k in rec and rec[k].numel() != n:
            raise ValueError(f"{where}: {k} has {rec[k].numel()} entries, n = {n}")
    for k, v in (rec.get("top1") or {}).items():
        if v.numel() != n:
            raise ValueError(f"{where}: top1 {k} has {v.numel()} entries, n = {n}")


def _merge(files: List[str]) -> Dict:
    """One record: the full-set file if there is one, else the slices concatenated from instance 0. The slices must
    come from one experiment (identical fingerprint(): setting, N, method, tag, instance set and its sha, model
    identity, decoding options, search and budget, the same metrics), hold as many entries as their range, not
    overlap and form one chain from 0 without holes; anything else is an error. A chain that stops early covers the
    first `n` instances only (the gaps of the table then need a teacher on the same first n)."""
    recs = [torch.load(f, weights_only=False) for f in files]
    for f, r in zip(files, recs):
        _check_lengths(r, f)
    full = [r for r in recs if r["slice"] is None]
    if full:
        return full[0]
    order = sorted(range(len(recs)), key=lambda i: tuple(recs[i]["slice"]))
    fp0, f0 = fingerprint(recs[order[0]]), files[order[0]]
    for i in order[1:]:
        fp = fingerprint(recs[i])
        if fp != fp0:
            bad = {k: (fp0.get(k), fp.get(k)) for k in sorted(set(fp0) | set(fp)) if fp0.get(k) != fp.get(k)}
            raise ValueError(f"slices of different experiments: {f0} vs {files[i]} differ in {bad}")
    pos = 0
    for i in order:
        lo, hi = recs[i]["slice"]
        if lo < pos:
            raise ValueError(f"{files[i]}: slice {lo}:{hi} overlaps the slices before it (they end at {pos})")
        if lo > pos:
            raise ValueError(f"{files[i]}: slice {lo}:{hi} leaves instances {pos}:{lo} uncovered")
        pos = hi
    recs = [recs[i] for i in order]
    out = dict(recs[0], slice=None, n=pos, seconds=sum(r["seconds"] for r in recs))
    for k in ("greedy", "search"):
        if k in recs[0]:
            out[k] = torch.cat([r[k] for r in recs])
    if "top1" in recs[0]:
        out["top1"] = {k: torch.cat([r["top1"][k] for r in recs]) for k in recs[0]["top1"]}
    return out


def table(root: str) -> Dict:
    out = {}
    lines = []
    for d in sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)):
        groups: Dict[str, List[str]] = {}
        for f in glob.glob(os.path.join(d, "*.pt")):
            groups.setdefault(os.path.basename(f)[:-3].split("@")[0], []).append(f)
        recs = {name: _merge(fs) for name, fs in sorted(groups.items())}
        tch = recs.get("teacher")
        rows = {}
        for name, r in recs.items():
            row = {"n": r["n"], "search_name": r["search_name"]}
            for k in ("greedy", "search"):
                if k in r:
                    row[k] = r[k].mean().item()
                    if tch is not None and k in tch and name != "teacher":
                        if tch["instances_sha256"] != r["instances_sha256"] or tch["n"] != r["n"]:
                            row["note"] = "not on the teacher's instances: no gap"
                            continue
                        if k == "search" and tch["search_name"] != r["search_name"]:
                            row["note"] = f"search {r['search_name']} vs the teacher's {tch['search_name']}: no gap"
                            continue
                        row[f"{k}_gap"] = 100 * (r[k].mean() / tch[k].mean() - 1).item()
            if "top1" in r:
                t = r["top1"]
                row["top1"] = 100 * (t["hits"].sum() / t["states"].sum()).item()
                row["top1_ci95"] = [100 * x for x in bootstrap_ci(t["hits"], t["states"])]
            rows[name] = row
        out[os.path.basename(d)] = rows
        lines += [f"## {os.path.basename(d)}", "", "| method | n | greedy | gap % | search | gap % | top-1 % [95% CI] |",
                  "|---|---|---|---|---|---|---|"]
        for name, row in rows.items():
            f = lambda k, p=4: f"{row[k]:.{p}f}" if k in row else ""          # noqa: E731
            t1 = f"{row['top1']:.2f} [{row['top1_ci95'][0]:.2f}, {row['top1_ci95'][1]:.2f}]" if "top1" in row else ""
            search = f"{f('search')} ({row['search_name']})" if "search" in row else ""
            lines.append(f"| {name} | {row['n']} | {f('greedy')} | {f('greedy_gap', 2)} | {search} | "
                         f"{f('search_gap', 2)} | {t1} |")
        lines.append("")
    json.dump(out, open(os.path.join(root, "table.json"), "w"), indent=1)
    open(os.path.join(root, "table.md"), "w").write("\n".join(lines))
    print("\n".join(lines))
    return out


def main():
    first = parser().parse_known_args()[0]
    if first.table:
        table(first.table)
        return
    if not first.setting or not first.method:
        parser().error("--setting and --method are required (or --table DIR)")
    a = parser(S.get(first.setting)).parse_args()
    run(a)


if __name__ == "__main__":
    main()
