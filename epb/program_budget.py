"""Program-complexity budget: the ablation "how simple can each program be?" (pomo_tsp and lehd_tsp only).

A tier fixes, per program, the maximum number of distinct feature primitives (concepts, epb/feats.py), statement lines
and branches (`if` / `torch.where` / `masked_fill`). `Budget.check` is an AST gate applied wherever code enters a bank
(initial bank, TextGrad rewrites, add candidates, JSON load): a violation is rejected like code that fails to run, and
the reason is fed back to the LLM. Optional anti-homogenisation rule: no program whose argmax agrees with a bank
program on >= `max_agreement` of a fixed synthetic probe set (0 = off).

The prompt texts of a tier (block, module template, fallback, seed directions) live in prompts/<setting>.yaml under
`budget:`; `Budget.prompt_vars()` gives the values they are filled with (llm.Prompts.apply_budget). Nothing here
calls an LLM.
"""
import ast
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch

from . import feats      # registers `tsp_feats` / `cand_feats` (alias `lehd_feats`), the modules programs import

TIERS = {           # tier: (max_concepts, max_lines, max_branches)
    "tiny":   (1, 8, 0),
    "simple": (2, 15, 1),
    "medium": (3, 30, 2),
}
SETTINGS = {        # setting: (raw inputs = the program's signature, primitives that are constant across actions, noun)
    "pomo_tsp": (("locs", "current_node", "mask", "visited", "step_norm", "first_node"),
                 frozenset({"progress", "n_unvisited"}), "per-node"),
    "lehd_tsp": (("unselected_locs", "first_loc", "current_loc"), frozenset({"n_unvisited"}), "per-candidate"),
}
ALLOWED_TORCH = {
    "exp", "log", "log1p", "sqrt", "abs", "clamp", "clamp_min", "clamp_max", "minimum", "maximum",
    "where", "sigmoid", "tanh", "pow", "square", "relu", "zeros_like", "ones_like", "full_like",
    "mean", "std", "sum", "min", "max", "amin", "amax", "logical_and", "logical_or", "logical_not",
    "float", "tensor", "finfo", "float32",
}
BRANCH_FUNCS = {"where", "masked_fill"}
FORBIDDEN_METHODS = {"gather", "scatter", "scatter_", "topk", "sort", "argsort", "cdist", "matmul", "bmm", "mm",
                     "einsum", "index_select", "cummax", "cummin", "kthvalue", "unfold", "roll", "flip"}


def get(tier: Optional[str], setting: str, **kw) -> Optional["Budget"]:
    """The budget of config `program_budget.tier` (None for "off")."""
    return None if tier in (None, "off") else Budget(tier, setting, **kw)


def program_concepts(code: str) -> FrozenSet[str]:
    """The concept set of a program: the names of the feat.* primitives it calls."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return frozenset()
    return frozenset(n.func.attr for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and isinstance(n.func.value, ast.Name) and n.func.value.id == "feat")


# ---- behaviour signature ------------------------------------------------------------------------------------------
_PROBES: Dict[str, list] = {}


def probe_states(setting: str) -> List[Tuple[tuple, Optional[torch.Tensor]]]:
    """Fixed synthetic decision states (CPU, seed 0) as [(program args, mask of the selectable actions or None)].
    pomo_tsp: 512 random TSP-100 partial tours (random visit order, >= 1 visited, >= 2 left);
    lehd_tsp: 256 random sub-problems at each of K = 10, 40, 80 candidates (variable-K signature)."""
    if setting not in _PROBES:
        g = torch.Generator().manual_seed(0)
        if setting == "pomo_tsp":
            B, N = 512, 100
            locs = torch.rand(B, N, 2, generator=g)
            n_vis = torch.randint(1, N - 1, (B,), generator=g)
            order = torch.argsort(torch.rand(B, N, generator=g), dim=1)          # random visit order
            mask = torch.ones(B, N, dtype=torch.bool).scatter(1, order, torch.arange(N).expand(B, N) >= n_vis[:, None])
            cur = order[torch.arange(B), n_vis - 1]
            _PROBES[setting] = [((locs, cur, mask, ~mask, n_vis.float() / N, order[:, 0]), mask)]
        elif setting == "lehd_tsp":
            probes = []
            for K in (10, 40, 80):
                unsel = torch.rand(256, K, 2, generator=g)
                first = torch.rand(256, 2, generator=g)
                probes.append(((unsel, first, torch.rand(256, 2, generator=g)), None))
            _PROBES[setting] = probes
        else:
            raise ValueError(f"no probe states for setting {setting!r}")
    return _PROBES[setting]


def behaviour_signature(fn: Callable, probes) -> Optional[torch.Tensor]:
    """argmax over the selectable actions of every probe state (concatenated), or None if the program fails."""
    try:
        sigs = []
        with torch.no_grad():
            for args, mask in probes:
                out = torch.nan_to_num(fn(*args).float(), nan=-1e9, posinf=1e9, neginf=-1e9)
                if mask is not None:
                    out = out.masked_fill(~mask, -float("inf"))
                sigs.append(out.argmax(dim=1))
        return torch.cat(sigs)
    except Exception:
        return None


def is_duplicate(sig: Optional[torch.Tensor], others: Sequence[Optional[torch.Tensor]], max_agreement: float,
                 exclude: Optional[int] = None) -> Optional[Tuple[int, float]]:
    """(index, agreement) of the signature in `others` that agrees most with `sig` (first on ties, skipping index
    `exclude` and None), if that agreement is >= max_agreement; else None. max_agreement <= 0 = off."""
    if max_agreement <= 0 or sig is None:
        return None
    best = None
    for i, osig in enumerate(others):
        if i == exclude or osig is None:
            continue
        agree = float((osig == sig).float().mean())
        if best is None or agree > best[1]:
            best = (i, agree)
    return best if best is not None and best[1] >= max_agreement else None


# ---- the gate -----------------------------------------------------------------------------------------------------
class _Parents(ast.NodeVisitor):
    def __init__(self):
        self.parent = {}

    def generic_visit(self, node):
        for child in ast.iter_child_nodes(node):
            self.parent[child] = node
        super().generic_visit(node)


def _call_name(node: ast.Call):
    f = node.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return f.value.id, f.attr
    if isinstance(f, ast.Attribute):
        return None, f.attr
    if isinstance(f, ast.Name):
        return None, f.id
    return None, None


class Budget:
    """The budget of one tier in one setting. Bank context is passed per call as `bank` = the bank's codes in index
    order; `exclude` = the index of the program being rewritten (TextGrad), which is not counted against itself."""

    def __init__(self, tier: str, setting: str, *, tg_retries: int = 2, max_agreement: float = 0.0):
        if setting not in SETTINGS:
            raise ValueError(f"the program budget is defined for {sorted(SETTINGS)} only, not {setting!r}")
        if tier not in TIERS:
            raise ValueError(f"unknown program-budget tier {tier!r}; one of {sorted(TIERS)} (or 'off')")
        self.tier, self.setting = tier, setting
        self.max_concepts, self.max_lines, self.max_branches = TIERS[tier]
        self.raw_inputs, self.state_only, self._per = SETTINGS[setting]
        self.module, self.primitives = feats.MODULES[setting], feats.PRIMITIVES[setting]   # module: the prompts'
        self.import_names = feats.import_names(setting)      # accepted by the gate (older banks: lehd_feats)
        self.tg_retries = int(tg_retries)                     # TextGrad rounds re-tried with the gate's reason
        self.max_agreement = float(max_agreement)
        self._sigs: Dict[str, torch.Tensor] = {}

    def __repr__(self):
        return f"Budget({self.tier!r}, {self.setting!r}: {TIERS[self.tier]} (max concepts, lines, branches))"

    # -- public API --
    def check(self, code: str, bank: Optional[Sequence[str]] = None, exclude: Optional[int] = None
              ) -> Tuple[bool, str]:
        """(ok, reason): the admission gate. Without `bank` only the per-program rules apply (JSON load). With `bank`,
        also the behavioural-duplicate rule (skipped for a code that is already in the bank verbatim)."""
        if bank is None:
            reason = self._reason(code, exclude=exclude)
        else:
            bank = list(bank)
            reason = self._reason(code, codes={c.strip() for c in bank}, others=bank, exclude=exclude)
        return reason is None, reason or ""

    def signature(self, code: str) -> Optional[torch.Tensor]:
        """Behaviour signature of a program's code on this setting's probe states (cached by code), None if it fails."""
        if code in self._sigs:
            return self._sigs[code]
        ns: dict = {}
        try:
            exec(compile(code, "<budget-probe>", "exec"), ns)  # noqa: S102  (code already passed the gate)
            fn = ns["heuristic"]
        except Exception:
            return None
        sig = behaviour_signature(fn, probe_states(self.setting))
        if sig is not None:
            self._sigs[code] = sig
        return sig

    def duplicate_of(self, code: str, bank: Sequence[str], exclude: Optional[int] = None):
        """(index, agreement) of the bank program `code` behaves like (agreement >= max_agreement), else None."""
        if self.max_agreement <= 0:
            return None
        sig = self.signature(code)
        if sig is None:
            return None
        return is_duplicate(sig, [self.signature(c) for c in bank], self.max_agreement, exclude)

    def prompt_vars(self) -> Dict:
        """Values for the `budget:` prompt templates of prompts/<setting>.yaml."""
        return {"tier": self.tier, "max_concepts": self.max_concepts, "max_lines": self.max_lines,
                "max_branches": self.max_branches, "feats_module": self.module,
                "raw_inputs": ", ".join(self.raw_inputs),
                "primitives": "\n".join(f"  - {v}" for v in self.primitives.values())}

    # -- the rules --
    def _reason(self, code: str, codes: Optional[set] = None, others: Optional[Sequence[str]] = None,
                exclude: Optional[int] = None) -> Optional[str]:
        """None if `code` satisfies the tier, else a short reason. codes: stripped bank codes exempt from the
        cross-bank rule; others: bank codes for the duplicate check."""
        tier, raw, (max_c, max_l, max_b) = self.tier, self.raw_inputs, TIERS[self.tier]
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"syntax error: {e.msg}"
        except ValueError as e:      # null bytes
            return f"syntax error: {e}"
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if len(fns) != 1 or fns[0].name != "heuristic":
            return "module must define exactly one function, `heuristic`"
        fn = fns[0]
        if [a.arg for a in fn.args.args] != list(raw):
            return "wrong signature"
        for n in tree.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in n.names]
                mod = n.module if isinstance(n, ast.ImportFrom) else None
                ok = (isinstance(n, ast.Import) and all(x in ("torch", "math") for x in names)) or \
                     (mod in self.import_names and names == ["feat"])
                if not ok:
                    return f"only `import torch`, `import math`, `from {self.module} import feat` are allowed"
            elif not isinstance(n, ast.FunctionDef):
                return "no module-level code besides imports and the function"
        pm = _Parents(); pm.visit(tree); parent = pm.parent
        concepts, branches = set(), 0
        for node in ast.walk(fn):
            if isinstance(node, (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                                 ast.Lambda, ast.Try, ast.With, ast.ClassDef)):
                return f"{type(node).__name__} is not allowed"
            if isinstance(node, ast.FunctionDef) and node is not fn:
                return "nested functions are not allowed"
            if isinstance(node, (ast.If, ast.IfExp)):
                branches += 1
            if isinstance(node, ast.Call):
                base, name = _call_name(node)
                if base == "feat":
                    if name not in self.primitives:
                        return f"unknown primitive feat.{name}"
                    concepts.add(name)
                elif base == "torch":
                    if name not in ALLOWED_TORCH:
                        return f"torch.{name} is not allowed (use the feat.* primitives)"
                    if name in BRANCH_FUNCS:
                        branches += 1
                elif base == "math":
                    pass
                else:
                    # method call on a tensor (or a plain call): count branches, ban re-implementing
                    # geometry / ranking on the raw inputs' derivatives.
                    if name in BRANCH_FUNCS:
                        branches += 1
                    if name in FORBIDDEN_METHODS:
                        return f".{name}() is not allowed (use the feat.* primitives)"
            if isinstance(node, ast.Name) and node.id in raw:
                p = parent.get(node)
                direct_arg = isinstance(p, ast.Call) and node in p.args and _call_name(p)[0] == "feat"
                if not direct_arg:
                    return f"raw input `{node.id}` may only be passed to feat.* primitives"
            if isinstance(node, ast.Subscript):
                v = node.value
                if isinstance(v, ast.Name) and v.id in raw:
                    return f"indexing `{v.id}` is not allowed"
        if len(concepts) > max_c:
            return f"{len(concepts)} concepts used ({sorted(concepts)}), tier '{tier}' allows {max_c}"
        if len(concepts) == 0:
            return "the program uses no feat.* primitive"
        if concepts <= self.state_only:
            per_node = ", ".join(p for p in self.primitives if p not in self.state_only)
            return (f"{sorted(concepts)} are constant across cities; the program needs at least one "
                    f"{self._per} primitive ({per_node})")
        already_in_bank = codes is not None and code.strip() in codes
        if branches > max_b:
            return f"{branches} branches, tier '{tier}' allows {max_b}"
        body = fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr)
                               and isinstance(getattr(fn.body[0], "value", None), ast.Constant)
                               and isinstance(fn.body[0].value.value, str)) else fn.body
        lines = set()
        for st in body:
            for ln in range(st.lineno, getattr(st, "end_lineno", st.lineno) + 1):
                lines.add(ln)
        src = code.splitlines()
        n_lines = sum(1 for ln in lines if ln - 1 < len(src) and src[ln - 1].strip()
                      and not src[ln - 1].strip().startswith("#"))
        if n_lines > max_l:
            return f"{n_lines} statement lines, tier '{tier}' allows {max_l}"
        if not already_in_bank and others is not None:
            dup = self.duplicate_of(code, others, exclude)
            if dup is not None:
                return (f"behaviourally identical to existing program H{dup[0]} (argmax agreement "
                        f"{100 * dup[1]:.1f}% on the probe set); a different coefficient on the same formula "
                        f"is not a new heuristic — change the formula or the primitives")
        return None
