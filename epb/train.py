"""Train an EPB student: a bank of LLM-written programs plus an attention router, distilled from a teacher's
decision-step data.

  python -m epb.train --config epb/configs/pomo_cvrp50.yaml [--set train.lr=3e-4 ...] [--force-slot-event]
  python -m epb.train --resume RUN_DIR [--set train.epochs=30]

Per epoch: router training (loss = KL(teacher || student), optional focal and routing terms), full validation, and on
event epochs (textgrad.start_epoch + k * textgrad.every_n_epochs, not during a finetune, not the last epoch) the
bank events:
  * plateau check on the val top-1 curve (OLS trend detector, `plateau.window`, `plateau.min_gain`);
  * no plateau -> TextGrad rewrites of the programs (if textgrad.enabled);
  * plateau -> slot drop (numeric leave-one-out) and slot add (standalone round, bank.add_round); after an add the
    router trains for `add.finetune_epochs` without events, then `add.cooldown_epochs` without drop / add (a failed
    or empty add starts the cooldown directly).
When the LLM budget (llm.max_cost_usd) runs out, TextGrad and add stop; router training and drops continue.
best_model.pt / heuristics_best.json are written together at the moment of the best val KL; last_model.pt /
heuristics_last.json every epoch (with the state --resume needs). Both checkpoints record their bank's sha
(bank_sha): --resume refuses a heuristics_last.json that is not last_model.pt's bank, and epb.eval refuses a
best_model.pt with another bank. decode.json (written at the start) tells `python -m epb.eval --ckpt RUN_DIR` how to
evaluate the run as it was trained: tau_h, soft keys and values and the "train" state convention (the checkpoints'
args record it too). Every config key must be known (unknown keys are an error) and is recorded when read; at the end
of the run the keys the yaml / --set gave that were never read are printed (all unread keys go to metrics.jsonl).
"""
import argparse
import copy
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml

from . import bank as B
from . import settings as S
from .llm import ABORTS, LLM, BudgetExhausted, Blind, Prompts, import_textgrad

HERE = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════════════════════════ config

DEFAULTS = {
    "setting": {"name": None},
    "run": {"output_dir": None, "seed": 42},
    "data": {"path": None, "val_ratio": 0.05, "num_workers": 4},
    "generate": {"teacher_ckpt": None, "num_instances": None, "seed": 1234, "nco_root": None},
    "init": {"bank": None, "router": None},
    "bank": {"num_programs": 8, "min_programs": 4, "max_programs": 20, "tau_h": 0.1},
    "router": {},
    "train": {"epochs": 20, "batch_size": 1024, "lr": 1e-4},
    "loss": {"focal": 0.0, "routing": 0.0},
    "textgrad": {"enabled": True, "start_epoch": 3, "every_n_epochs": 3, "backward_every_n": 200, "buffer_size": 10,
                 "rounds": 3, "min_improvement": 0.005, "eval_batches": 10},
    "plateau": {"window": 8, "min_gain": 0.001},
    "drop": {"enabled": True, "grace_epochs": 20, "top1_tol": 0.005, "kl_tol": 0.005, "max_per_event": 3,
             "cooldown_epochs": 9},
    "add": {"enabled": True, "n_candidates": 6, "min_gain": 0.001, "min_faults": 200, "pretrain_epochs": 5,
            "finetune_epochs": 3, "cooldown_epochs": 3},
    "llm": {"model": None, "reasoning_effort": None, "max_tokens": None, "max_workers": 16, "max_cost_usd": None,
            "price_input": None, "price_output": None, "price_cache_read": None},
    "prompts": {"file": None, "teacher": None},
    "blind": {"enabled": False},
    "program_budget": {"tier": "off", "max_agreement": 0.0},
}
ROUTING_TARGET_TAU = 0.05     # temperature of the routing soft target (loss.routing > 0)
STATE_CONVENTION = "train"    # epb.eval: decision states exactly as the training rows (settings.state_convention)
FOCAL_GAMMA = 2.0
PLATEAU_T_CRIT, PLATEAU_DECEL = 2.0, 0.3
BLIND_TEACHER = "expert policy"


class Section:
    """One config section; attribute reads are recorded so unread keys can be reported."""

    def __init__(self, name: str, values: Dict, used: set):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_used", used)

    def __getattr__(self, key):
        values = object.__getattribute__(self, "_values")
        if key not in values:
            raise AttributeError(f"config has no key {object.__getattribute__(self, '_name')}.{key}")
        object.__getattribute__(self, "_used").add(f"{object.__getattribute__(self, '_name')}.{key}")
        return values[key]

    def __setattr__(self, key, value):
        self._values[key] = value

    def get(self, key, default=None):
        return getattr(self, key) if key in self._values else default

    def to_dict(self) -> Dict:
        for k in self._values:
            self._used.add(f"{self._name}.{k}")
        return dict(self._values)


class Config:
    def __init__(self, tree: Dict, explicit: set = frozenset()):
        self._tree, self._used, self.explicit = tree, set(), set(explicit)   # explicit: keys the yaml / --set gave
        for name, values in tree.items():
            setattr(self, name, Section(name, values, self._used))

    def unread(self) -> List[str]:
        return sorted(f"{s}.{k}" for s, sec in self._tree.items() for k in sec if f"{s}.{k}" not in self._used)

    def dump(self, path: str):
        with open(path, "w") as f:
            yaml.safe_dump(self._tree, f, sort_keys=False)


def _merge(base: Dict, over: Dict, where: str, allow_new: bool) -> Dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out and not allow_new:
            raise ValueError(f"unknown config key {where}{k}")
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v, f"{where}{k}.", allow_new)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str], sets: List[str], resolved: Optional[Dict] = None) -> Config:
    """defaults < the setting's defaults < the experiment yaml < --set key=value (value parsed as YAML)."""
    tree = resolved
    if tree is None:
        user = yaml.safe_load(open(path)) or {}
        name = (user.get("setting") or {}).get("name")
        setting = S.get(name)
        tree = _merge(DEFAULTS, setting.defaults, "", allow_new=True)   # a setting may add keys (router, ...)
        tree = _merge(tree, user, "", allow_new=False)
        explicit = {f"{sec}.{k}" for sec, vals in user.items() if isinstance(vals, dict) for k in vals}
    else:
        explicit = set()
    for s in sets:
        key, _, val = s.partition("=")
        node = tree
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node:
                raise ValueError(f"unknown config key {key}")
            node = node[p]
        if parts[-1] not in node:
            raise ValueError(f"unknown config key {key}")
        node[parts[-1]] = yaml.safe_load(val)
        explicit.add(key)
    return Config(tree, explicit)


# ═══════════════════════════════════════════════════════════════════════════════════════════════ plateau

class Plateau:
    """Plateau of the val top-1 curve since the last reset: OLS slope over the last `window` epochs. A plateau is a
    declining or noise-level trend (|t| < t_crit), a projected gain over the next TextGrad period below `min_gain`,
    or a slope that fell below PLATEAU_DECEL of its peak."""

    def __init__(self, window: int, min_gain: float):
        self.window, self.min_gain = window, min_gain
        self.reset()

    def reset(self):
        self.curve: List[Tuple[int, float]] = []
        self.peak = 0.0

    def add(self, epoch: int, top1: float):
        self.curve.append((epoch, top1))

    def check(self, period: int) -> Tuple[bool, Dict]:
        win = self.curve[-self.window:]
        if len(win) < max(4, self.window // 2):
            return False, {"reason": "warmup", "n_points": len(win)}
        xs, ys = [float(e) for e, _ in win], [y for _, y in win]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        s2 = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        se = math.sqrt(s2 / sxx) if s2 > 0 else 0.0
        t = slope / se if se > 0 else (math.inf if slope > 0 else -math.inf)
        gain = slope * period
        self.peak = max(self.peak, slope)
        if t <= -PLATEAU_T_CRIT:
            reason = "declining"
        elif t < PLATEAU_T_CRIT:
            reason = "trend_within_noise"
        elif gain < self.min_gain:
            reason = "gain_below_floor"
        elif self.peak > 0 and slope < PLATEAU_DECEL * self.peak:
            reason = "decelerated"
        else:
            reason = "growing"
        return reason != "growing", {"reason": reason, "slope": slope, "t": t, "gain": gain, "n_points": n}

    def state_dict(self):
        return {"curve": self.curve, "peak": self.peak}

    def load_state_dict(self, d):
        self.curve, self.peak = [tuple(x) for x in d["curve"]], d["peak"]


# ═══════════════════════════════════════════════════════════════════════════════════════════════ training

class Predict:
    """The router as a fixed function of (batch, V_full, V_K) -> (y_hat, w), for TextGrad / drop evaluations."""

    def __init__(self, setting, model, amp: bool, device):
        self.setting, self.model, self.amp, self.device = setting, model, amp, device

    def __call__(self, batch, V_full, V_K):
        self.model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            y_hat, w, _ = self.setting.router_forward(self.model, batch, V_full, V_K)
        return y_hat.float(), w.float()


def run_epoch(setting, model, bank: B.Bank, loader, device, cfg, amp: bool, optimizer=None,
              tg: Optional[B.TextGrad] = None, log=None, epoch: int = 0, keep: int = 0, seed: int = 0) -> Dict:
    """One pass. keep > 0: also return `keep` batches drawn uniformly from the pass (reservoir sampling with a fixed
    seed, so the same batches each epoch when the loader order is fixed) for the TextGrad / drop evaluations."""
    train = optimizer is not None
    model.train(train)
    kept, rng, budget_hit = [], random.Random(seed), None
    tot = {"kl": 0.0, "focal": 0.0, "routing": 0.0, "top1": 0, "oracle": 0, "rows": 0, "valid": 0}
    lam_f, lam_r = cfg.loss.focal, cfg.loss.routing
    every = cfg.textgrad.backward_every_n
    for i, batch in enumerate(loader):
        if keep:
            if len(kept) < keep:
                kept.append(batch)
            elif rng.random() < keep / (i + 1):
                kept[rng.randrange(keep)] = batch
        batch = B.to_device(batch, device)
        teacher, mask = setting.target(batch)
        with torch.no_grad():
            V_full, V_K, _ = bank.compute_probs(batch)
        with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            y_hat, w, scores = setting.router_forward(model, batch, V_full, V_K)
        y_hat, w, scores = y_hat.float(), w.float(), scores.float()
        kl = B.kl_rows(teacher, y_hat, mask)
        loss = kl.mean()
        t1 = B.masked_argmax(teacher, mask)
        if lam_f > 0:
            pt = y_hat.clamp(min=B.EPS).gather(-1, t1[:, None]).squeeze(-1)
            focal = -((1 - pt) ** FOCAL_GAMMA * pt.log()).mean()
            loss = loss + lam_f * focal
            tot["focal"] += float(focal) * kl.numel()
        if lam_r > 0:
            with torch.no_grad():
                kl_m = torch.stack([B.kl_rows(teacher, V_full[:, m], mask) for m in range(V_full.size(1))], -1)
                right = B.masked_argmax(V_full, mask.unsqueeze(1)) == t1.unsqueeze(1)
                target = F.softmax(-kl_m.masked_fill(~right, float("inf")) / ROUTING_TARGET_TAU, -1)
                none = ~right.any(-1)
                if none.any():
                    target[none] = F.softmax(-kl_m[none] / ROUTING_TARGET_TAU, -1)
            routing = -(target * F.log_softmax(scores, -1)).sum(-1).mean()
            loss = loss + lam_r * routing
            tot["routing"] += float(routing) * kl.numel()
        if train:
            optimizer.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        with torch.no_grad():
            valid = mask.any(-1)
            tot["kl"] += float(kl.sum())
            tot["rows"] += kl.numel()
            tot["valid"] += int(valid.sum())
            tot["top1"] += int(((B.masked_argmax(y_hat, mask) == t1) & valid).sum())
            tot["oracle"] += int(((B.masked_argmax(V_full, mask.unsqueeze(1)) == t1.unsqueeze(1)).any(-1) & valid).sum())
        if tg is not None:
            tg.accumulate(w.detach(), V_full, teacher, mask, batch, setting.row_ids(batch))
            if (i + 1) % every == 0:
                try:
                    with B.fork_rng():
                        tg.backward()
                except BudgetExhausted as e:     # the epoch's router training goes on without TextGrad
                    budget_hit, tg = e, None
        if log is not None and train and (i + 1) % 50 == 0:
            log({"type": "batch", "epoch": epoch, "batch": i, "kl": kl.mean().item(), "loss": loss.item()})
    n, v = max(tot["rows"], 1), max(tot["valid"], 1)
    return {"kl": tot["kl"] / n, "focal": tot["focal"] / n, "routing": tot["routing"] / n, "top1": tot["top1"] / v,
            "oracle_cov": tot["oracle"] / v, "rows": tot["rows"], "kept": kept, "budget_hit": budget_hit}


# ═══════════════════════════════════════════════════════════════════════════════════════════════ checkpoints

def _atomic_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class Best:
    """best_model.pt and heuristics_best.json, written together when the val KL improves."""

    def __init__(self, run_dir: str, best_kl: float = math.inf):
        self.run_dir, self.kl = run_dir, best_kl

    def offer(self, val: Dict, source: str, epoch: int, model, bank: B.Bank, ckpt_args: Dict, log) -> bool:
        if not val["kl"] < self.kl:
            return False
        self.kl = val["kl"]
        sha = bank.sha()
        _atomic_save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_kl": val["kl"],
                      "val_top1": val["top1"], "val_source": source, "bank_sha": sha, "args": ckpt_args},
                     os.path.join(self.run_dir, "best_model.pt"))
        bank.save(os.path.join(self.run_dir, "heuristics_best.json"),
                  {"epoch": epoch, "val_kl": val["kl"], "val_top1": val["top1"], "val_source": source, "bank_sha": sha})
        log({"type": "best", "epoch": epoch, "val_kl": val["kl"], "source": source})
        return True


def check_pair(ck: Dict, sha: str, ck_name: str, bank_name: str, strict: bool) -> None:
    """A checkpoint's bank_sha against the sha of the bank file written with it."""
    want = ck.get("bank_sha")
    if want is None or want == sha:
        return
    msg = f"{ck_name} was written with the bank {want}, but {bank_name} holds the bank {sha}"
    if strict:
        raise RuntimeError(msg + ": the run directory is inconsistent, refusing to resume")
    print(f"[resume] warning: {msg} (rewritten at the next val KL improvement; epb.eval refuses this pair)")


def _rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s):
    random.setstate(s["python"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


# ═══════════════════════════════════════════════════════════════════════════════════════════════ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="experiment yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one config key")
    ap.add_argument("--resume", metavar="RUN_DIR", help="continue an interrupted run from its last_model.pt")
    ap.add_argument("--force-slot-event", action="store_true", help="treat the next TextGrad epoch as a plateau")
    a = ap.parse_args()
    if bool(a.config) == bool(a.resume):
        ap.error("give exactly one of --config / --resume")
    if a.resume:
        cfg = load_config(None, a.set, yaml.safe_load(open(os.path.join(a.resume, "config.yaml"))))
    else:
        cfg = load_config(a.config, a.set)
    setting = S.get(cfg.setting.name)
    run_dir = a.resume or cfg.run.output_dir
    if not run_dir:
        raise ValueError("run.output_dir is not set")
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    def log(rec: Dict):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")

    seed = cfg.run.seed
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"      # set by the setting's defaults

    # data (the split is saved and reused on resume)
    split_path = os.path.join(run_dir, "split.pt")
    split = torch.load(split_path, weights_only=False) if a.resume and os.path.exists(split_path) else None
    data = setting.build_data(cfg, split)
    if split is None:
        torch.save(data.split, split_path)
    print(f"[data] {cfg.data.path}: N={data.problem_size}")

    # prompts, LLM, teacher-blind
    blind = None
    if cfg.blind.enabled:
        cfg.prompts.teacher = BLIND_TEACHER
        blind = Blind(replacement=BLIND_TEACHER, mode="abort", log_path=os.path.join(run_dir, "blind_log.jsonl"))
    prompt_file = cfg.prompts.file or os.path.join(HERE, "prompts", f"{setting.name}.yaml")
    from . import program_budget
    budget = program_budget.get(cfg.program_budget.tier, setting.name,
                                max_agreement=cfg.program_budget.max_agreement)
    prompts = Prompts(prompt_file, setting.prompt_vars(cfg, data.problem_size))
    if budget is not None:
        prompts.apply_budget(budget)
        print(f"[budget] {budget}")
    prompts.validate()
    prompts.lock(os.path.join(run_dir, "prompts.lock.json"))
    if blind is not None:
        print(blind.audit(prompts, os.path.join(run_dir, "blind_audit.txt")))
    needs_llm = cfg.textgrad.enabled or cfg.add.enabled or not (cfg.init.bank or a.resume)
    llm = None
    if needs_llm:
        import_textgrad(log_dir=os.path.join(run_dir, "textgrad_logs"))
        llm = LLM(cfg.llm.to_dict(), log_path=os.path.join(run_dir, "api_metrics.jsonl"), blind=blind)

    # bank
    bank = B.Bank(setting, cfg.bank.tau_h, budget, B.Bank.probe_chunks(data.val))
    ck_resume = None
    if a.resume:
        bank.load(os.path.join(run_dir, "heuristics_last.json"))
        ck_resume = torch.load(os.path.join(run_dir, "last_model.pt"), map_location="cpu", weights_only=False)
        check_pair(ck_resume, bank.sha(), "last_model.pt", "heuristics_last.json", strict=True)
        best_path, best_bank = os.path.join(run_dir, "best_model.pt"), os.path.join(run_dir, "heuristics_best.json")
        if os.path.exists(best_path) and os.path.exists(best_bank):
            check_pair(torch.load(best_path, map_location="cpu", weights_only=False), B.bank_file_sha(best_bank),
                       "best_model.pt", "heuristics_best.json", strict=False)
    elif cfg.init.bank:
        bank.load(cfg.init.bank)
    else:
        print(f"[bank] generating {cfg.bank.num_programs} programs with {cfg.llm.model}")
        with B.fork_rng():
            B.propose_bank(bank, cfg.bank.num_programs, llm, prompts)
        bank.save(os.path.join(run_dir, "heuristics_init.json"), {"stage": "init"})
    if blind is not None:
        blind.check_bank([{"code": p.code, "description": p.description} for p in bank.programs])
    missing = [p for p in bank.programs if not p.description]
    if missing and llm is not None:         # released banks carry no descriptions; TextGrad and the add round need them
        for p, d in zip(missing, llm.map(lambda p: B.describe(llm, prompts, p.code), missing)):
            p.description = d if isinstance(d, str) else ""
        print(f"[bank] described {len(missing)} programs without a description")
    print(f"[bank] {len(bank)} programs")

    # router
    model = setting.build_router(cfg, len(bank)).to(device)
    if cfg.init.router and not a.resume:
        ck = torch.load(cfg.init.router, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        print(f"[router] initialised from {cfg.init.router}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
                 if cfg.train.get("scheduler", "none") == "cosine" else None)
    predict = Predict(setting, model, amp, device)
    resize = getattr(model, "set_num_programs", None)

    # trainer state
    plateau = Plateau(cfg.plateau.window, cfg.plateau.min_gain)
    st = {"epoch": 0, "finetune": 0, "cooldown": 0, "last_drop": -10 ** 9, "llm_off": False,
          "force": bool(a.force_slot_event)}
    best = Best(run_dir)
    st_eval: List[Dict] = []      # val batches of the TextGrad / drop evaluations (drawn by validate)
    if a.resume:
        ck = ck_resume
        if resize:
            resize(len(bank))
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        if scheduler is not None and ck.get("scheduler_state_dict"):
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        st.update(ck["trainer_state"])
        st["force"] = bool(a.force_slot_event)
        plateau.load_state_dict(ck["plateau"])
        best.kl = ck.get("best_kl", math.inf)
        if llm is not None and ck.get("llm"):
            llm.load_state_dict(ck["llm"])
        _set_rng_state(ck["rng"])
        print(f"[resume] from epoch {st['epoch']}")
        log({"type": "run_resume", "epoch": st["epoch"]})
    else:
        cfg.dump(os.path.join(run_dir, "config.yaml"))
        log({"type": "run_start", "config": cfg._tree})

    with open(os.path.join(run_dir, "decode.json"), "w") as f:     # how epb.eval --ckpt RUN_DIR decodes this run
        json.dump({"tau_h": float(cfg.bank.tau_h), "k_mode": "soft", "v_mode": "soft",
                   "state_convention": STATE_CONVENTION}, f)

    tgc, dc, ac = cfg.textgrad, cfg.drop, cfg.add
    tg = B.TextGrad(bank, llm, prompts, tgc, log) if llm is not None else None
    args_of = lambda: {**setting.ckpt_args(cfg, len(bank)), "state_convention": STATE_CONVENTION}   # noqa: E731

    def bank_changed():
        if resize:
            resize(len(bank))

    n_eval = max(tgc.eval_batches, B.DROP_EVAL_BATCHES)

    def validate(tag: str, epoch: int) -> Dict:
        with torch.no_grad():
            v = run_epoch(setting, model, bank, data.val, device, cfg, amp, keep=n_eval, seed=seed)
        st_eval[:] = v.pop("kept")
        log({"type": "val", "epoch": epoch, "source": tag, **v, "M": len(bank)})
        best.offer(v, tag, epoch, model, bank, args_of(), log)
        return v

    def llm_ok() -> bool:
        return llm is not None and not st["llm_off"]

    def budget_stop(e, epoch):
        st["llm_off"] = True
        print(f"[llm] {e}: no more TextGrad / add from now on (router training and numeric drops continue)")
        log({"type": "api_budget_stop", "epoch": epoch, "spent_usd": llm.spent_usd})

    for epoch in range(st["epoch"] + 1, cfg.train.epochs + 1):
        t0 = time.time()
        data.set_epoch(epoch)
        in_finetune, in_cooldown = st["finetune"] > 0, st["cooldown"] > 0
        event_epoch = (not in_finetune and epoch >= tgc.start_epoch and (epoch - tgc.start_epoch) % tgc.every_n_epochs == 0
                       and epoch < cfg.train.epochs)       # no bank change the router would never train on
        tg_epoch = event_epoch and tgc.enabled and llm_ok()
        if tg_epoch:
            tg.reset()
        tr = run_epoch(setting, model, bank, data.train, device, cfg, amp, optimizer,
                       tg if tg_epoch else None, log, epoch)
        tr.pop("kept")
        if tr.pop("budget_hit") is not None:
            budget_stop("API budget exhausted", epoch)
            tg_epoch = False
        if scheduler is not None:
            scheduler.step()
        val = validate("pre_event", epoch)
        plateau.add(epoch, val["top1"])
        print(f"epoch {epoch:3d} | M={len(bank)} | train KL {tr.get('kl', float('nan')):.4f} | val KL {val['kl']:.4f} "
              f"top-1 {val['top1']:.4f} oracle {val['oracle_cov']:.4f} | {time.time() - t0:.0f}s")

        if event_epoch and (tg_epoch or dc.enabled or (ac.enabled and llm_ok())):
            is_plateau, diag = plateau.check(tgc.every_n_epochs)
            if st["force"]:
                is_plateau, diag, st["force"] = True, {**diag, "reason": "forced"}, False
            log({"type": "plateau_check", "epoch": epoch, "plateau": is_plateau, **diag})
            drop_ok = (dc.enabled and not in_cooldown and epoch - st["last_drop"] >= dc.cooldown_epochs
                       and len(bank) > cfg.bank.min_programs)
            add_ok = ac.enabled and llm_ok() and not in_cooldown and len(bank) < cfg.bank.max_programs
            try:
                if tg_epoch and not (is_plateau and (drop_ok or add_ok)):
                    with B.fork_rng():
                        n_acc = tg.update(epoch, predict, st_eval[:tgc.eval_batches])
                    bank.save(os.path.join(run_dir, f"heuristics_epoch{epoch}.json"), {"epoch": epoch, "event": "textgrad"})
                    if n_acc:
                        validate("post_textgrad", epoch)
                if is_plateau and drop_ok:
                    st["last_drop"] = epoch
                    res = B.drop(bank, epoch, predict, st_eval[:B.DROP_EVAL_BATCHES], data.val, dc, cfg.bank.min_programs)
                    log({"type": "drop", "epoch": epoch, **res})
                    print(f"  [drop] {res['reason']}" + (f": {res['dropped'] or res['candidates']}, val KL "
                          f"{res['pre_kl']:.4f} -> {res['post_kl']:.4f}, top-1 {res['pre_top1']:.4f} -> "
                          f"{res['post_top1']:.4f}" if "post_kl" in res else ""))
                    if res["dropped"]:
                        bank_changed()
                        bank.save(os.path.join(run_dir, f"heuristics_epoch{epoch}_drop.json"),
                                  {"epoch": epoch, "event": "drop", "dropped": res["dropped"]})
                        validate("post_drop", epoch)
                        plateau.reset()
                if is_plateau and add_ok:
                    with B.fork_rng():
                        new = B.add_round(bank, epoch, data, llm, prompts, ac, tgc,
                                          cfg.bank.max_programs - len(bank), device,
                                          os.path.join(run_dir, f"add_epoch{epoch}"), log, seed + epoch)
                    if new:
                        bank.add(new)
                        bank_changed()
                        bank.save(os.path.join(run_dir, f"heuristics_epoch{epoch}_add.json"),
                                  {"epoch": epoch, "event": "add", "added": len(new)})
                        validate("post_add", epoch)
                        st["finetune"] = ac.finetune_epochs
                    else:
                        st["cooldown"] = ac.cooldown_epochs
                    plateau.reset()
            except BudgetExhausted as e:
                budget_stop(e, epoch)
            except ABORTS:
                raise
        elif in_finetune:
            st["finetune"] -= 1
            if st["finetune"] == 0:
                st["cooldown"] = ac.cooldown_epochs
                plateau.reset()
        if in_cooldown:             # counts every epoch, TextGrad epochs included
            st["cooldown"] -= 1

        st["epoch"] = epoch
        bank.save(os.path.join(run_dir, "heuristics_last.json"), {"epoch": epoch, "bank_sha": bank.sha()})
        _atomic_save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                      "epoch": epoch, "val_kl": val["kl"], "val_top1": val["top1"], "bank_sha": bank.sha(),
                      "args": args_of(), "trainer_state": {k: v for k, v in st.items() if k != "force"},
                      "plateau": plateau.state_dict(), "best_kl": best.kl, "rng": _rng_state(),
                      "llm": llm.state_dict() if llm is not None else None},
                     os.path.join(run_dir, "last_model.pt"))
        log({"type": "epoch", "epoch": epoch, "M": len(bank), "train": tr, "val": val,
             "spent_usd": llm.spent_usd if llm is not None else 0.0, "seconds": time.time() - t0})

    unread = [k for k in cfg.unread() if not k.startswith("generate.")]    # generate.* belongs to epb.generate
    log({"type": "run_end", "best_kl": best.kl, "unread_config_keys": unread})
    no_effect = [k for k in unread if k in cfg.explicit]
    if no_effect:
        print(f"[config] set but never used in this run: {no_effect}")
    print(f"done: best val KL {best.kl:.4f}")


if __name__ == "__main__":
    main()
