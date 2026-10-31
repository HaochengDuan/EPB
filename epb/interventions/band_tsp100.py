import os

import torch

from ..settings.pomo_tsp import PomoTeacher, RouterPolicy, decision_state, repeat_rows
from .common import HERE, bank_functions, load_rule

NAME, SETTING, N = "band_tsp100", "pomo_tsp", 100
HELDOUT = ("diag_band_s3", "diag_band_s5")
POOLS = {"pooled": HELDOUT}
PARTS = ("teacher_rule", "divergence")
EDITED = os.path.join(HERE, "pomo_tsp100_band")
LAYOUT = {"teacher_rule": dict(batch=128, ms_chunk=128, starts_per_chunk=25), "divergence": dict(batch=128)}
GAMMAS = "0,1,2,4,8,16,32,64,1000"
RULE_ACTIVE = 1e-6

RULES = {"injected": bank_functions(EDITED)["_rule_heuristic"],
         "recipe": load_rule(os.path.join(HERE, "rules", "band_tsp100_recipe.py"))}
TEACHER_RULES = {k: [f] for k, f in RULES.items()}
DEFAULT_TEACHER_RULE = "recipe"
DIVERGENCE_SETS = ("diag_band_s0",) + HELDOUT


class RuleTeacher(PomoTeacher):

    def __init__(self, base: PomoTeacher, rules, gammas):
        self.model, self.device, self.rules, self.gammas = base.model, base.device, rules, gammas

    def begin(self, locs, first):
        super().begin(locs, first)
        self.rows, self.first = repeat_rows(locs, first.size(1)), first.reshape(-1)

    def choose(self, visited, cur, step):
        B, S, N = visited.shape
        probs, _ = self.decode(visited, cur)
        vis = visited.reshape(B * S, N)
        fm = ~vis
        step_norm = torch.full((B * S,), float(step - 1) / (N - 1), device=self.device)
        n = fm.sum(-1, keepdim=True).clamp(min=1).float()
        logit = torch.log(probs.clamp(min=1e-12))
        for rule, gamma in zip(self.rules, self.gammas):
            rs = rule(self.rows, cur.reshape(-1), fm, vis.float(), step_norm, self.first).float()
            mean = rs.masked_fill(vis, 0.0).sum(-1, keepdim=True) / n
            std = (((rs - mean) ** 2).masked_fill(vis, 0.0).sum(-1, keepdim=True) / n).sqrt().clamp(min=1e-6)
            logit = logit + gamma * ((rs - mean) / std).masked_fill(vis, 0.0).reshape(B, S, N)
        return logit.masked_fill(visited, float("-inf")).argmax(dim=2)


teacher_rule = RuleTeacher


class RuleTracker(RouterPolicy):

    def __init__(self, policy: RouterPolicy, rules):
        self.__dict__.update(vars(policy))
        self.rules, self.steps = rules, 0
        self.counts = {r: {"diverge": 0, "active": 0} for r in rules}

    def pick(self, vis, cur, step):
        nxt = super().pick(vis, cur, step)
        b = decision_state(self.locs, vis, cur, self.first, step)
        step_norm = b["step_i"] / b["step_max"].clamp_min(1.0)
        for r, fn in self.rules.items():
            rs = fn(self.locs, cur, ~vis, vis.float(), step_norm, self.first).float()
            spread = rs.masked_fill(vis, float("-inf")).amax(-1) - rs.masked_fill(vis, float("inf")).amin(-1)
            act = spread > RULE_ACTIVE
            self.counts[r]["diverge"] += int(((nxt != rs.masked_fill(vis, float("-inf")).argmax(-1)) & act).sum())
            self.counts[r]["active"] += int(act.sum())
        self.steps += vis.size(0)
        return nxt


def divergence(setting, policy: RouterPolicy, inst, a):
    tr = RuleTracker(policy, RULES)
    setting.eval_greedy(tr, inst, a)
    return {r: dict(c, steps=tr.steps) for r, c in tr.counts.items()}
