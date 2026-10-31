import os

import torch

from ..settings.pomo_cvrp import EvalPolicy
from .common import HERE, bank_functions, load_rule

NAME, SETTING, N = "band_cvrp50", "pomo_cvrp", 50
HELDOUT = ("diag_band_s3", "diag_band_s4", "diag_band_s1", "diag_band_s2")
POOLS = {"seeds 3+4": ("diag_band_s3", "diag_band_s4"), "seeds 1+2": ("diag_band_s1", "diag_band_s2")}
PARTS = ("teacher_rule", "rules_alone", "divergence")
EDITED = os.path.join(HERE, "pomo_cvrp50_band")
LAYOUT = {part: dict(batch=64) for part in PARTS}
GAMMAS = "32/32/32,32/16/32"
RULE_ACTIVE = 1e-6
TIE = 1e-4

_bank = bank_functions(EDITED)
RULES = {"start": _bank["_rule_heuristic_L2"], "route": _bank["_rule_heuristic_L1"],
         "no_early_return": _bank["_rule_heuristic"]}
TEACHER_RULES = {"same": [RULES[r] for r in ("route", "start", "no_early_return")]}
DEFAULT_TEACHER_RULE = "same"
DIVERGENCE_SETS = HELDOUT
FULL_RULE = os.path.join(HERE, "rules", "band_cvrp50_full.py")


def _scores(fn, *args):
    rs = fn(*args).float()
    if rs.shape != args[3].shape or not torch.isfinite(rs).all():
        raise RuntimeError(f"rule {fn.__name__}: bad output {tuple(rs.shape)}")
    return rs


def _spread(rs, fm, keepdim=False):
    return rs.masked_fill(~fm, float("-inf")).amax(-1, keepdim=keepdim) - \
        rs.masked_fill(~fm, float("inf")).amin(-1, keepdim=keepdim)


def _z(rs, fm):
    n = fm.sum(-1, keepdim=True).clamp(min=1).float()
    mean = rs.masked_fill(~fm, 0.0).sum(-1, keepdim=True) / n
    std = (((rs - mean) ** 2).masked_fill(~fm, 0.0).sum(-1, keepdim=True) / n).sqrt().clamp(min=1e-6)
    act = (_spread(rs, fm, keepdim=True) > RULE_ACTIVE).float()
    return (((rs - mean) / std).clamp(-3.0, 3.0) * act).masked_fill(~fm, 0.0)


class _RuleTeacherModel:

    def __init__(self, model, rules, gammas):
        self.model, self.rules, self.gammas = model, rules, gammas

    def pre_forward(self, depot_xy, node_xy, node_demand):
        self.model.pre_forward(depot_xy, node_xy, node_demand)
        self.locs = torch.cat([depot_xy, node_xy], dim=1)
        self.demand = torch.cat([torch.zeros_like(node_demand[:, :1]), node_demand], dim=1)
        self.vis = None

    def __call__(self, current_node, load, ninf_mask):
        sel, probs, logits = self.model(current_node, load, ninf_mask)
        B, P, N1 = ninf_mask.shape
        if self.vis is None:
            self.vis, self.first = torch.zeros_like(ninf_mask), current_node.clone()
        self.vis.scatter_(2, current_node[:, :, None], 1.0)
        self.vis[:, :, 0] = 0.0
        feas = ninf_mask == 0
        fm = feas.reshape(B * P, N1)
        args = (self.locs[:, None].expand(B, P, N1, 2).reshape(B * P, N1, 2),
                self.demand[:, None].expand(B, P, N1).reshape(B * P, N1), current_node.reshape(B * P), fm,
                self.vis.reshape(B * P, N1), load.reshape(B * P), self.first.reshape(B * P))
        logit = torch.log(probs.clamp(min=1e-12))
        for fn, gamma in zip(self.rules, self.gammas):
            if gamma != 0:
                logit = logit + gamma * _z(_scores(fn, *args), fm).view(B, P, N1)
        return logit.masked_fill(~feas, float("-inf")).argmax(dim=-1), probs, logits


def teacher_rule(base: EvalPolicy, rules, gammas) -> EvalPolicy:
    return EvalPolicy("teacher", _RuleTeacherModel(base.model, rules, gammas), base.device)


class _RulesAlone:

    def __init__(self, band, no_early_return):
        self.band, self.ner = band, no_early_return

    def compute_probs(self, flat):
        args = (flat["locs"], flat["demand"], flat["current_node"], flat["mask"], flat["visited"], flat["load"],
                flat["first_node"])
        sc = self.band(*args).float().clone()
        r = self.ner(*args).float()
        sc[:, 0] = torch.where(r[:, 0] < 0, torch.full_like(sc[:, 0], -1e3), sc[:, 0])
        return (sc[:, None, :],)

    def __call__(self, flat, V, V_out=None):
        N1 = flat["locs"].size(1)
        P = N1 - 1
        locs = flat["locs"][::P].contiguous()
        B = locs.size(0)
        dcur = torch.cdist(locs, locs).gather(1, flat["current_node"].view(B, P, 1).expand(B, P, N1))
        return (V[:, 0].view(B, P, N1) - TIE * dcur).view(B * P, N1), None


def rules_alone(device: str) -> EvalPolicy:
    pol = _RulesAlone(load_rule(FULL_RULE), RULES["no_early_return"])
    return EvalPolicy("router", pol, torch.device(device), bank=pol)


class _DeclineTracker:

    def __init__(self, model, rules):
        self.model, self.rules, self.steps = model, rules, 0
        self.counts = {r: {"diverge": 0, "active": 0} for r in rules}

    def __call__(self, flat, V, V_out=None):
        y_hat, w = self.model(flat, V, V_out=V_out)
        P = flat["locs"].size(1) - 1
        f = {k: v[::P].contiguous() for k, v in flat.items()}
        fm = f["mask"] > 0.5
        sel = y_hat[::P].masked_fill(~fm, float("-inf")).argmax(-1)
        active = ~((f["visited"][:, 1:] > 0).all(-1) & (f["current_node"] == 0))
        args = (f["locs"], f["demand"], f["current_node"], f["mask"], f["visited"], f["load"], f["first_node"])
        for r, fn in self.rules.items():
            rs = _scores(fn, *args)
            act = (_spread(rs, fm) > RULE_ACTIVE) & active
            self.counts[r]["diverge"] += int(((sel != rs.masked_fill(~fm, float("-inf")).argmax(-1)) & act).sum())
            self.counts[r]["active"] += int(act.sum())
        self.steps += int(active.sum())
        return y_hat, w


def divergence(setting, policy: EvalPolicy, inst, a):
    tr = _DeclineTracker(policy.model, RULES)
    pol = EvalPolicy("router", tr, policy.device, bank=policy.bank, k_onehot=policy.k_onehot,
                     v_onehot=policy.v_onehot)
    setting.eval_greedy(pol, inst, a)
    return {r: dict(c, steps=tr.steps) for r, c in tr.counts.items()}
