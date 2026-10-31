import json
import os
import time
import warnings
from typing import Dict

import torch

from .. import eval as E
from .. import instances as I
from .. import settings as S

HERE = os.path.dirname(os.path.abspath(__file__))


def load_rule(path: str, name: str = "heuristic"):
    ns = {}
    exec(compile(open(path).read(), path, "exec"), ns)   # noqa: S102
    return ns[name]


def bank_functions(student_dir: str) -> Dict:
    ns = {}
    exec(json.load(open(os.path.join(student_dir, "bank.json")))["heuristics"][0]["code"], ns)   # noqa: S102
    return ns


def eval_args(case, method: str, device: str, layout: Dict, *extra: str):
    argv = ["--setting", case.SETTING, "--method", method, "--device", device, *extra]
    for k, v in layout.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return E.build_args(argv)


def save_costs(case, out: str, name: str, sha: str, method: str, tag: str, greedy, search) -> None:
    spec = "band:" + name
    d = E.result_dir(out, case.SETTING, case.N, spec)
    os.makedirs(d, exist_ok=True)
    rec = {"setting": case.SETTING, "N": case.N, "method": method, "tag": tag, "instances": spec,
           "instances_sha256": sha, "slice": None, "n": greedy.numel(), "search_name": "ms",
           "greedy": greedy.double().cpu(), "search": search.double().cpu()}
    torch.save(rec, os.path.join(d, f"{method}_{tag}.pt"))


def cells(spec: str):
    return [tuple(float(g) for g in c.split("/")) for c in spec.split(",") if c]


def cell_tag(gammas) -> str:
    return "g" + "-".join(f"{g:g}" for g in gammas)


def run(case, a) -> None:
    parts = [p for p in a.parts.split(",") if p]
    bad = [p for p in parts if p not in case.PARTS]
    if bad:
        raise ValueError(f"unknown parts {bad}; choose from {case.PARTS}")
    if "teacher_rule" in parts and not a.teacher_ckpt:
        raise ValueError("teacher_rule needs --teacher-ckpt")
    setting = S.get(case.SETTING)
    inst = {n: I.load("band:" + n, case.SETTING) for n in dict.fromkeys(case.HELDOUT + case.DIVERGENCE_SETS)}
    for part in parts:
        t0 = time.time()
        if part == "teacher_rule":
            ta = eval_args(case, "teacher", a.device, case.LAYOUT["teacher_rule"], "--teacher-ckpt", a.teacher_ckpt)
            base = setting.eval_policy("teacher", ta)
            rules = case.TEACHER_RULES[a.teacher_rule]
            for gs in cells(a.gammas):
                if len(gs) != len(rules):
                    raise ValueError(f"gammas {gs}: one weight per rule of {a.teacher_rule!r} ({len(rules)})")
                pol = case.teacher_rule(base, rules, gs)
                for n in case.HELDOUT:
                    x = inst[n][0]
                    save_costs(case, a.out, n, inst[n][1], "teacher_rule", f"{a.teacher_rule}_{cell_tag(gs)}",
                               setting.eval_greedy(pol, x, ta), setting.eval_search(pol, x, ta))
                print(f"[{case.NAME}] teacher_rule {a.teacher_rule} {cell_tag(gs)} ({time.time() - t0:.0f}s)", flush=True)
        elif part == "rules_alone":
            ra = eval_args(case, "epb", a.device, case.LAYOUT["rules_alone"])
            pol = case.rules_alone(a.device)
            for n in case.HELDOUT:
                x = inst[n][0]
                save_costs(case, a.out, n, inst[n][1], "rules_alone", "band",
                           setting.eval_greedy(pol, x, ra), setting.eval_search(pol, x, ra))
            print(f"[{case.NAME}] rules_alone ({time.time() - t0:.0f}s)", flush=True)
        elif part == "divergence":
            da = eval_args(case, "epb", a.device, case.LAYOUT["divergence"], "--ckpt", case.EDITED)
            pol = setting.eval_policy("epb", da)
            warnings.filterwarnings("error", message=".*runtime error.*")
            counts = {n: case.divergence(setting, pol, inst[n][0], da) for n in case.DIVERGENCE_SETS}
            report = divergence_report(case, counts)
            os.makedirs(a.out, exist_ok=True)
            json.dump({"counts": counts, "divergence": report},
                      open(os.path.join(a.out, f"{case.NAME}_divergence.json"), "w"), indent=1)
            for g, v in report.items():
                print(f"[{case.NAME}] divergence {g}: " + "; ".join(
                    f"{r} {100 * c['divergence']:.2f}% of {c['active']} active steps" for r, c in v.items()), flush=True)


def divergence_report(case, counts: Dict) -> Dict:
    groups = {n: (n,) for n in counts}
    groups.update({p: s for p, s in case.POOLS.items() if all(x in counts for x in s)})
    out = {}
    for g, sets in groups.items():
        for r in counts[sets[0]]:
            c = {k: sum(counts[s][r][k] for s in sets) for k in ("diverge", "active", "steps")}
            out.setdefault(g, {})[r] = dict(c, divergence=c["diverge"] / max(c["active"], 1),
                                            active_share=c["active"] / max(c["steps"], 1))
    return out
