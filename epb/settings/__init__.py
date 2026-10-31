"""Settings: everything that depends on the teacher family (POMO / LEHD) and the problem (TSP / CVRP).

The generic code (train.py, bank.py) talks to a setting only through the `Setting` interface below. A setting
module (`settings/<name>.py`) defines one subclass and registers it under its name; `get(name)` returns an
instance. Batches are dicts of tensors, on the CPU when they leave a loader and moved to the device by train.py.

Shapes: B rows; A = size of the action space (TSP: N nodes; CVRP: N + 1, depot at index 0); K = number of
candidates of a LEHD sub-problem (for POMO settings V_K is V_full and K = A).
"""
import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

NAMES = ["pomo_tsp", "pomo_cvrp", "lehd_tsp", "lehd_cvrp"]


@dataclass
class Data:
    """Loaders of one dataset split into train / val (the split is by instance, so no instance is on both sides)."""
    train: Iterable[Dict]                   # shuffled batches of the train split; reshuffled by set_epoch
    val: Iterable[Dict]                     # batches of the val split, fixed order
    problem_size: int                       # N, read from the data
    scan: Callable[[int], Iterable[Dict]]   # seed -> shuffled batches over the TRAIN split only (fault scan of an add)
    subset: Callable[[torch.Tensor, int, bool, int], Iterable[Dict]]
    #                                       (row_ids, batch_size, shuffle, seed) -> batches of those TRAIN rows
    set_epoch: Callable[[int], None] = lambda epoch: None
    split: Dict = field(default_factory=dict)   # what defines the split; saved in the run dir and passed back on resume


class Setting:
    name: str = ""
    problem: str = ""                       # "tsp" | "cvrp"
    defaults: Dict = {}                     # setting-level config defaults, same sections as the config (router, bank, train ...)

    # ---- data ---------------------------------------------------------------------------------------------------
    def build_data(self, cfg, split: Optional[Dict] = None) -> Data:
        """cfg.data.path / val_ratio / num_workers, cfg.train.batch_size, cfg.run.seed. With `split` (from a previous
        run) the same split is rebuilt instead of drawn."""
        raise NotImplementedError

    def row_ids(self, batch: Dict) -> torch.Tensor:
        """[B] int64 id of each row, unique within the dataset; train ids are accepted by Data.subset."""
        raise NotImplementedError

    def instance_ids(self, batch: Dict) -> torch.Tensor:
        """[B] int64 id of the instance each row belongs to (rows of one instance share it; used to split the faults
        of an add round by instance)."""
        raise NotImplementedError

    def target(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """(teacher probabilities [B, A], feasibility mask [B, A] bool) on the batch's device."""
        raise NotImplementedError

    # ---- programs -----------------------------------------------------------------------------------------------
    def program_args(self, batch: Dict) -> tuple:
        """Positional arguments of heuristic(...) for this batch (same device as the batch)."""
        raise NotImplementedError

    def program_probs(self, fn: Callable, batch: Dict, tau_h: float) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        """Run one program on the batch and turn its output into probabilities. An exception or a wrong shape gives the
        setting's fallback and ok=False; how non-finite values are treated (sanitised with ok=True, or a fallback) is
        the setting's rule. Returns (V_full [B, A], V_K [B, K] or None, ok). torch.cuda.OutOfMemoryError is re-raised
        (the bank handles it)."""
        raise NotImplementedError

    def check_program(self, fn: Callable) -> None:
        """CPU checks of a new program before it may run on the GPU (synthetic shapes and geometric edge cases):
        raise ValueError(reason) if it raises or returns a wrong shape."""
        raise NotImplementedError

    # ---- router -------------------------------------------------------------------------------------------------
    def build_router(self, cfg, num_programs: int) -> nn.Module:
        raise NotImplementedError

    def router_forward(self, model: nn.Module, batch: Dict, V_full: torch.Tensor, V_K: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(y_hat [B, A], routing weights w [B, M], routing scores [B, M])."""
        raise NotImplementedError

    def ckpt_args(self, cfg, num_programs: int) -> Dict:
        """The flat `args` dict stored in checkpoints (the architecture fields the evaluators read)."""
        raise NotImplementedError

    # ---- prompts ------------------------------------------------------------------------------------------------
    def prompt_vars(self, cfg, problem_size: int) -> Dict[str, str]:
        """Setting variables of the prompt templates (problem_size, n_plus_1, tau_h, teacher, ...)."""
        raise NotImplementedError

    def failure_case(self, row: Dict, V_m: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor, kind: str) -> str:
        """Text of one failure case for a TextGrad backward prompt (`${case}`); kind is "code" or "desc".
        row: one row of a batch (tensors without the batch dimension, CPU); V_m, teacher, mask: [A]."""
        raise NotImplementedError

    def fault_case(self, row: Dict, V_bank: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> str:
        """Text of one bank fault (no program has the teacher's top-1) for the add round's cluster prompt.
        V_bank: [M, A] probabilities of every program."""
        raise NotImplementedError

    def fault_summary(self, rows: List[Dict], V_banks: List[torch.Tensor], teachers: List[torch.Tensor],
                      masks: List[torch.Tensor]) -> str:
        """Optional aggregate statistics over the sampled faults for the cluster prompt (`${summary}`)."""
        return ""

    # ---- data generation ----------------------------------------------------------------------------------------
    def generate(self, cfg) -> None:
        """Write the decision-step dataset of cfg.generate (teacher checkpoint, instances, seed) to cfg.data.path."""
        raise NotImplementedError

    # ---- evaluation (epb.eval) ----------------------------------------------------------------------------------
    # inst: the instance tensors of epb.instances (CPU). a: the parsed epb.eval arguments (a.method, a.ckpt, a.bank,
    # a.pkg, a.teacher_ckpt, a.nco_root, a.k_mode, a.v_mode, a.tau_h, a.drop, a.state_convention, a.budget,
    # a.search, a.batch, a.device, a.threads, plus the setting's own eval_args). Options a caller leaves unset
    # (None or absent) follow eval_option / state_convention below: tau_h from the checkpoint's heuristic_temperature,
    # then the setting's bank.tau_h. Costs are tour lengths, one per instance, in instance order, computed in the
    # precision of the setting's rollout (epb.eval stores them as float64).
    search_default: str = ""                # "ms" (POMO: best of all N first nodes) | "rrc" (LEHD: random re-construct)
    budget_default: Dict[int, int] = {}     # N -> RRC budget of the main table (LEHD settings)

    def eval_args(self, parser) -> None:
        """Setting-specific epb.eval arguments (optional)."""

    def eval_policy(self, method: str, a):
        """The policy of `method`: teacher | epb | expert | frozen (router + bank) | nam | moet | tree | agent."""
        raise NotImplementedError

    def eval_greedy(self, policy, inst: Dict[str, torch.Tensor], a) -> torch.Tensor:
        """[n] cost of the policy's greedy solution (from the setting's fixed first node / the depot)."""
        raise NotImplementedError

    def eval_search(self, policy, inst: Dict[str, torch.Tensor], a) -> torch.Tensor:
        """[n] cost of the setting's search: a.search == "ms" or "rrc" (budget a.budget)."""
        raise NotImplementedError

    def eval_top1(self, teacher, policy, inst: Dict[str, torch.Tensor], a) -> Dict[str, torch.Tensor]:
        """Teacher-forced top-1: along the teacher's greedy trajectory, the policy's argmax at every teacher state vs
        the teacher's next action. Per instance [n] (float64): hits / states over the primary states (not forced:
        more than one feasible action; TSP also drops the first step) and hits_all / states_all over all states."""
        raise NotImplementedError


STATE_CONVENTIONS = ("legacy", "train")


def eval_option(a, name: str, ckpt_args: Optional[Dict] = None, ckpt_key: Optional[str] = None, default=None):
    """An evaluation option with the precedence of epb.eval: command-line flag > decode.json (both already in `a`:
    resolve_ckpt fills the unset flags from decode.json) > the checkpoint's `args[ckpt_key]` > `default`."""
    v = getattr(a, name, None)
    if v is None and ckpt_args and ckpt_key:
        v = ckpt_args.get(ckpt_key)
    return default if v is None else v


def state_convention(a, ckpt_args: Optional[Dict] = None) -> str:
    """How the evaluator builds the decision states a router / bank sees (decode key `state_convention`, flag
    --state-convention): "train" = exactly the rows of the setting's data generator (as epb.generate runs it);
    "legacy" = the evaluator of the released results. They differ in POMO-TSP only (legacy: step_i = t - 1 at decision
    t, train: step_i = t); POMO-CVRP, LEHD-TSP and LEHD-CVRP build the same states under both. Precedence: flag >
    decode.json > the checkpoint's args["state_convention"] (epb.train writes "train") > "legacy" (checkpoints
    without the key)."""
    v = eval_option(a, "state_convention", ckpt_args, "state_convention", "legacy")
    if v not in STATE_CONVENTIONS:
        raise ValueError(f"state_convention must be one of {STATE_CONVENTIONS}, got {v!r}")
    return v


_REGISTRY: Dict[str, type] = {}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> Setting:
    if name not in NAMES:
        raise ValueError(f"unknown setting {name!r}; one of {NAMES}")
    if name not in _REGISTRY:
        importlib.import_module(f"{__name__}.{name}")
    return _REGISTRY[name]()
