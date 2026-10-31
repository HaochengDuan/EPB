"""The program bank and everything that changes it.

  Bank           LLM-written heuristic programs: compile and check new code (CPU checks of the setting, then a chunked
                 CPU pre-flight on real batches, so a bad program can never reach the GPU), compute_probs -> V.
  TextGrad       failure buffers filled during training, text gradients, code rewrites accepted only if the bank-level
                 val KL (router fixed) improves.
  drop()         numeric leave-one-out drop, verified on the full val split.
  add_round()    fault scan of the train split -> LLM clustering into failure modes -> one candidate per mode ->
                 TextGrad pretrain of the candidates on their faults -> greedy selection by val oracle-coverage gain.
  propose_bank() the initial bank (ideas from seed directions, then implementations).

The setting (settings/<name>.py) supplies the program signature, the output -> probability rules and the text of
failure cases; prompts come from prompts/<setting>.yaml through llm.Prompts.
"""
import ast
import json
import os
import random
import time
import warnings
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

import torch

from . import feats  # noqa: F401  registers tsp_feats / cand_feats (lehd_feats), the budget-tier primitive modules
from .llm import LLMCallError, import_textgrad, make_optimizer, optimizer_step, strip_code_fences

EPS = 1e-9
PREFLIGHT_ROWS = 128            # rows per pre-flight chunk ...
PREFLIGHT_CHUNKS = 32           # ... of real val rows every new program runs on (CPU) before it may run on the GPU
PREFLIGHT_SECONDS = 2.0         # a chunk slower than this rejects the program
IMPLEMENT_RETRIES = 12          # attempts to turn one strategy into a program that passes the checks
BUDGET_REJECT = "program budget: "   # prefix of a program-budget gate rejection
GRAD_KL_THRESHOLD = 0.05        # minimum failure score for the TextGrad buffer
MAX_GRADS_PER_BATCH = 2         # failures per program per backward call
MOMENTUM_WINDOW = 2             # TGD momentum
MIN_WEIGHTED_KL = 1.0           # a program is skipped by an update if its weighted KL is below this ...
KL_GOOD_FRACTION = 0.5          # ... and below this fraction of the mean over programs
RE_BACKWARD = 3                 # gradients regenerated from the candidate's failures after a rejected rewrite
REIMPLEMENT_ATTEMPTS = 3        # phase 2: implementations of the rewritten description
MAX_FAULTS = 20000              # fault pool of an add round
CLUSTER_CASES = 80              # faults shown to the cluster prompt
DEMOS_PER_MODE = 4              # cases of a mode shown when implementing it
SANITY_MIN_HIT = 0.05           # target hit rate of an add candidate on its mode's held-out faults (teacher's top-1):
SANITY_ATTEMPTS = 4             # below it, re-implemented up to SANITY_ATTEMPTS times; the best attempt is kept even
#                                 below the target (the selection by val coverage gain filters useless candidates)
FAULT_VAL_RATIO = 0.1           # instances of the fault pool held out to accept candidate rewrites
FAULT_EVAL_BATCHES = 10
PRETRAIN_BACKWARD_EVERY = 4     # batches between candidate rewrites in the add pretrain
PRETRAIN_BATCH = 1024
DROP_EVAL_BATCHES = 5           # val batches of the leave-one-out scan (the result is verified on the full val split)


# ═══════════════════════════════════════════════════════════════════════════════════════════════ helpers

def kl_rows(teacher: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL(teacher || pred) per row, summed over the feasible nodes. [B]"""
    p = teacher.clamp(min=EPS)
    q = pred.clamp(min=EPS)
    return (p * (p.log() - q.log())).masked_fill(~mask, 0.0).sum(-1)


def masked_argmax(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x.masked_fill(~mask, float("-inf")).argmax(-1)


def to_device(batch: Dict, device) -> Dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def slice_batch(batch: Dict, idx) -> Dict:
    """Rows idx (slice or index tensor) of a batch; tensors without a batch dimension are kept as they are."""
    B = batch_size(batch)
    return {k: (v[idx] if torch.is_tensor(v) and v.dim() > 0 and v.size(0) == B else v) for k, v in batch.items()}


def batch_size(batch: Dict) -> int:
    return max(v.size(0) for v in batch.values() if torch.is_tensor(v) and v.dim() > 0)


def row_of(batch: Dict, b: int) -> Dict:
    """Row b of a batch on the CPU, tensors without the batch dimension."""
    B = batch_size(batch)
    return {k: (v[b].detach().cpu() if torch.is_tensor(v) and v.dim() > 0 and v.size(0) == B else v)
            for k, v in batch.items()}


@contextmanager
def fork_rng():
    """Run an LLM event without moving the RNG streams of the training loop."""
    state = (random.getstate(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        yield
    finally:
        random.setstate(state[0])
        torch.set_rng_state(state[1])
        if state[2] is not None:
            torch.cuda.set_rng_state_all(state[2])


def _raise(*args, **kwargs):
    raise RuntimeError("program disabled")


# ═══════════════════════════════════════════════════════════════════════════════════════════════ bank

def codes_sha(codes: List[str]) -> str:
    import hashlib
    return hashlib.sha256("\x00".join(codes).encode()).hexdigest()[:16]


def bank_file_sha(path: str) -> str:
    """Bank.sha() of a bank JSON (bank.json / heuristics_*.json), without compiling its programs."""
    return codes_sha([h["code"] for h in json.load(open(path))["heuristics"]])


class Program:
    __slots__ = ("uid", "code", "description", "fn", "update_mode", "added_at")

    def __init__(self, uid, code, description, fn, update_mode, added_at=None):
        self.uid, self.code, self.description, self.fn = uid, code, description, fn
        self.update_mode, self.added_at = update_mode, added_at


class Bank:
    def __init__(self, setting, tau_h: float, budget=None, probe_rows: Optional[List[Dict]] = None):
        self.setting, self.tau_h, self.budget = setting, tau_h, budget
        self.probe_rows = probe_rows or []           # CPU chunks of val rows for the pre-flight of new programs
        self.programs: List[Program] = []
        self._next_uid = 0

    def __len__(self):
        return len(self.programs)

    # ---- compile / admit ----------------------------------------------------------------------------------------
    def compile(self, code: str, exclude: Optional[int] = None, gate: bool = True) -> Callable:
        """Parse, gate (program budget; exclude = position being rewritten; gate=False only for the tier's own
        fallback), exec and run the setting's CPU checks; raises ValueError(reason)."""
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"syntax error: {e}")
        if self.budget is not None and gate:
            ok, reason = self.budget.check(code, bank=[p.code for p in self.programs], exclude=exclude)
            if not ok:
                raise ValueError(BUDGET_REJECT + reason)
        ns: Dict = {}
        try:
            exec(compile(code, "<heuristic>", "exec"), ns)  # noqa: S102
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"exec failed: {e}")
        fn = ns.get("heuristic")
        if not callable(fn):
            raise ValueError("no function named heuristic")
        try:
            self.setting.check_program(fn)
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"check failed: {e}")
        return fn

    def preflight(self, fn: Callable) -> None:
        """Run fn on the CPU over the probe batches in small chunks; raise ValueError if it fails or is too slow."""
        with fork_rng(), torch.no_grad():
            n_threads = torch.get_num_threads()
            torch.set_num_threads(min(4, n_threads))
            try:
                for sub in self.probe_rows:
                    t0 = time.time()
                    _, _, ok = self.setting.program_probs(fn, sub, self.tau_h)
                    if not ok:
                        raise ValueError("pre-flight: the program fails on real states")
                    if time.time() - t0 > PREFLIGHT_SECONDS:
                        raise ValueError(f"pre-flight: {PREFLIGHT_ROWS} rows took over {PREFLIGHT_SECONDS:.0f} s")
            finally:
                torch.set_num_threads(n_threads)

    @staticmethod
    def probe_chunks(val_loader) -> List[Dict]:
        """PREFLIGHT_CHUNKS chunks of PREFLIGHT_ROWS rows from the first val batches (CPU)."""
        out = []
        for batch in val_loader:
            B = batch_size(batch)
            for s in range(0, B, PREFLIGHT_ROWS):
                out.append(slice_batch(batch, slice(s, s + PREFLIGHT_ROWS)))
                if len(out) >= PREFLIGHT_CHUNKS:
                    return out
        return out

    def admit(self, code: str, description: str, update_mode: str, added_at: Optional[int] = None,
              preflight: bool = True, exclude: Optional[int] = None, gate: bool = True) -> Program:
        fn = self.compile(code, exclude, gate)
        if preflight:
            self.preflight(fn)
        p = Program(self._next_uid, code, description, fn, update_mode, added_at)
        self._next_uid += 1
        return p

    def add(self, programs: List[Program]):
        self.programs.extend(programs)

    def remove(self, positions: List[int]):
        drop = set(positions)
        self.programs = [p for i, p in enumerate(self.programs) if i not in drop]

    def replace(self, pos: int, program: Program):
        self.programs[pos] = program

    def snapshot(self) -> List[Program]:
        return list(self.programs)

    def restore(self, snap: List[Program]):
        self.programs = list(snap)

    # ---- files ----------------------------------------------------------------------------------------------------
    def load(self, path: str) -> "Bank":
        """A bank JSON ({"heuristics": [{index, description, code, ...}]}). Loaded programs pass the setting's
        checks but no pre-flight (they ran before)."""
        data = json.load(open(path))
        for i, h in enumerate(data["heuristics"]):
            try:
                fn = self._compile_loaded(h["code"])
            except ValueError as e:
                raise RuntimeError(f"{path}: program {h.get('index', i)} does not load: {e}")
            self.programs.append(Program(self._next_uid, h["code"], h.get("description", ""), fn,
                                         h.get("update_mode", "loaded"), h.get("added_at")))
            self._next_uid += 1
        return self

    def _compile_loaded(self, code: str) -> Callable:
        programs, self.programs = self.programs, []
        try:
            return self.compile(code)
        finally:
            self.programs = programs

    def save(self, path: str, metadata: Optional[Dict] = None):
        data = {"metadata": metadata or {},
                "heuristics": [{"index": i, "description": p.description, "update_mode": p.update_mode,
                                "added_at": p.added_at, "code": p.code} for i, p in enumerate(self.programs)]}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def sha(self) -> str:
        """The bank's identity (its programs' code, in order); best_model.pt / last_model.pt record it as bank_sha."""
        return codes_sha([p.code for p in self.programs])

    # ---- probabilities --------------------------------------------------------------------------------------------
    def program_probs(self, p: Program, batch: Dict):
        try:
            return self.setting.program_probs(p.fn, batch, self.tau_h)
        except torch.cuda.OutOfMemoryError:
            # a program that runs out of memory at training batch size would do so on every batch
            torch.cuda.empty_cache()
            warnings.warn(f"program {p.uid} ran out of GPU memory and is disabled (uniform output from now on)")
            p.fn = _raise
            return self.setting.program_probs(p.fn, batch, self.tau_h)

    def compute_probs(self, batch: Dict, override: Optional[Dict[int, Callable]] = None,
                      programs: Optional[List[Program]] = None):
        """(V_full [B, M, A], V_K [B, M, K], ok [M] bool). override: {position: fn} evaluates a candidate in place of
        the program at that position; programs: a different list of programs (e.g. add candidates)."""
        progs = self.programs if programs is None else programs
        Vf, Vk, ok = [], [], []
        for i, p in enumerate(progs):
            if override and i in override:
                vf, vk, good = self.setting.program_probs(override[i], batch, self.tau_h)
            else:
                vf, vk, good = self.program_probs(p, batch)
            Vf.append(vf)
            Vk.append(vf if vk is None else vk)
            ok.append(good)
        return torch.stack(Vf, 1), torch.stack(Vk, 1), torch.tensor(ok)


# ═══════════════════════════════════════════════════════════════════════════════════════════════ LLM helpers

def _render_list(prompts, key: str) -> List[str]:
    val = prompts.render(key)
    return val if isinstance(val, list) else [val]


def implement(llm, prompts, bank: Bank, strategy: str, update_mode: str, added_at=None,
              system_key: str = "implement_system", user_key: str = "implement_user", extra: Optional[Dict] = None,
              source: str = "implement") -> Optional[Program]:
    """Strategy text -> admitted program, with up to IMPLEMENT_RETRIES attempts; the reason of a rejected attempt is
    fed back to the LLM. None if every attempt fails."""
    system = prompts.render(system_key, **(extra or {}))
    user = prompts.render(user_key, strategy=strategy, template=prompts.render("template"), **(extra or {}))
    for attempt in range(IMPLEMENT_RETRIES):
        try:
            code = strip_code_fences(llm.call(system, user, source=source))
        except LLMCallError as e:
            print(f"    [implement] attempt {attempt + 1}: {e}")
            continue
        try:
            return bank.admit(code, strategy, update_mode, added_at)
        except ValueError as e:
            print(f"    [implement] attempt {attempt + 1}: rejected ({e})")
            user_retry = prompts.render("retry_reject", reason=str(e))
            user = prompts.render(user_key, strategy=strategy, template=prompts.render("template"),
                                  **(extra or {})) + "\n\n" + user_retry
    return None


def describe(llm, prompts, code: str) -> str:
    try:
        return llm.call(prompts.render("describe_system"), prompts.render("describe_user", code=code),
                        source="describe").strip()
    except LLMCallError:
        return ""


def propose_bank(bank: Bank, n: int, llm, prompts) -> None:
    """Initial bank: n strategy ideas (one per seed direction, each told the previous ideas), implemented in
    parallel; an idea that cannot be implemented gets the fallback program."""
    seeds = _render_list(prompts, "seeds")
    ideas: List[str] = []
    for i in range(n):
        existing = "".join(f"  {j + 1}. {d}\n" for j, d in enumerate(ideas))
        try:
            idea = llm.call(prompts.render("idea_system"),
                            prompts.render("idea_user", direction=seeds[i % len(seeds)], existing=existing),
                            source="idea").strip()
        except LLMCallError as e:
            idea = seeds[i % len(seeds)]
            print(f"  idea {i}: LLM call failed ({e}); using the seed direction")
        print(f"  idea {i}: {idea}")
        ideas.append(idea)
    results = llm.map(lambda idea: implement(llm, prompts, bank, idea, "initial"), ideas)   # against the empty bank
    fallback = prompts.render("fallback")
    for i, r in enumerate(results):
        if isinstance(r, Program) and bank.budget is not None and \
                bank.budget.duplicate_of(r.code, [p.code for p in bank.programs]) is not None:
            print(f"  program {i}: behaviourally identical to an earlier program, re-implementing")
            r = implement(llm, prompts, bank, ideas[i], "initial")    # gated against the programs added so far
        if isinstance(r, Program):
            bank.add([r])
            print(f"  program {i}: OK")
        else:
            bank.add([bank.admit(fallback, "nearest neighbour (fallback)", "fallback", preflight=False, gate=False)])
            print(f"  program {i}: fallback ({r if isinstance(r, Exception) else 'implementation failed'})")


# ═══════════════════════════════════════════════════════════════════════════════════════════════ TextGrad

class TextGrad:
    """Text-gradient refinement of the bank's programs.

    During training, accumulate() keeps per program the rows with the highest failure score
    w_m * [argmax V_m != teacher top-1] * KL(teacher || V_m); every `backward_every_n` batches backward() turns the
    top cases of each program into code and description gradients (a case is used once). update() rewrites each
    program that carries enough weighted failure: phase 1 up to `rounds` code rewrites, phase 2 a rewritten
    description re-implemented; a rewrite is accepted only if the bank-level KL on `eval_batches` val
    batches (router fixed) drops by more than `min_improvement` (relative), and every accept moves the baseline.
    """

    def __init__(self, bank: Bank, llm, prompts, cfg, log: Callable[[Dict], None]):
        self.bank, self.llm, self.prompts, self.cfg, self.log = bank, llm, prompts, cfg, log
        self.state: Dict[int, Dict] = {}           # uid -> {"code": Variable, "desc": Variable, "opt", "dopt"}
        self.reset()

    def reset(self):
        self.buf: Dict[int, List] = {}             # uid -> [(score, kl, row_id, case_code, case_desc)]
        self.used: Dict[int, set] = {}             # uid -> row ids already turned into gradients
        self.wkl: Dict[int, float] = {}
        for st in self.state.values():
            st["opt"].zero_grad()
            st["dopt"].zero_grad()

    # ---- textgrad variables ---------------------------------------------------------------------------------------
    def _vars(self, p: Program) -> Dict:
        st = self.state.get(p.uid)
        if st is None:
            tg = import_textgrad()
            engine = self.llm.textgrad_engine()
            code = tg.Variable(p.code, requires_grad=True, role_description=self.prompts.render("textgrad.code_role"))
            desc = tg.Variable(p.description, requires_grad=True,
                               role_description=self.prompts.render("textgrad.desc_role"))
            st = {"code": code, "desc": desc,
                  "opt": make_optimizer(engine, code, MOMENTUM_WINDOW,
                                         _render_list(self.prompts, "textgrad.code_constraints"),
                                         self.prompts.render("optimizer_system")),
                  "dopt": make_optimizer(engine, desc, MOMENTUM_WINDOW,
                                          _render_list(self.prompts, "textgrad.desc_constraints"))}
            self.state[p.uid] = st
        return st

    def _add_grad(self, var, text: Optional[str], role: str):
        if text:
            var.gradients.add(import_textgrad().Variable(text, requires_grad=False, role_description=role))

    def _gradients(self, items: List[Tuple[Program, float, str, str]]):
        """items: (program, kl, case_code, case_desc) -> one code and one description gradient each."""
        def one(item):
            p, kl, case_code, case_desc = item
            st = self._vars(p)
            out = []
            for key, desc_key, case in (("backward_code", False, case_code), ("backward_desc", True, case_desc)):
                vars_ = dict(code=st["code"].value, kl=f"{kl:.4f}", case=case)
                if desc_key:
                    vars_["description"] = st["desc"].value
                try:
                    out.append(self.llm.call(self.prompts.render(f"{key}_system"),
                                             self.prompts.render(f"{key}_user", **vars_), source=key))
                except LLMCallError:
                    out.append(None)
            return out
        for item, res in zip(items, self.llm.map(one, items)):
            if isinstance(res, Exception):
                continue
            st = self._vars(item[0])
            self._add_grad(st["code"], res[0], "text gradient: diagnosis of the code on a failure case")
            self._add_grad(st["desc"], res[1], "text gradient: what the description fails to capture")

    # ---- training-time hooks --------------------------------------------------------------------------------------
    def accumulate(self, w: torch.Tensor, V_full: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor,
                   batch: Dict, row_ids: torch.Tensor):
        cap = max(4 * self.cfg.buffer_size, MAX_GRADS_PER_BATCH)
        t1 = masked_argmax(teacher, mask)
        setting = self.bank.setting
        for m, p in enumerate(self.bank.programs):
            V_m = V_full[:, m]
            kl_m = kl_rows(teacher, V_m, mask)
            score = (w[:, m] * (masked_argmax(V_m, mask) != t1).float() * kl_m).detach()
            self.wkl[p.uid] = self.wkl.get(p.uid, 0.0) + float(score.sum())
            buf = self.buf.setdefault(p.uid, [])
            used = self.used.setdefault(p.uid, set())
            vals, order = score.sort(descending=True)
            vals, order = vals[:cap].cpu().tolist(), order[:cap].cpu().tolist()
            for s, b in zip(vals, order):
                if s < GRAD_KL_THRESHOLD:
                    break
                if len(buf) >= cap and s <= min(x[0] for x in buf):
                    break
                rid = int(row_ids[b])
                if rid in used:
                    continue
                row = row_of(batch, b)
                Vb, tb, mb = V_m[b].detach().cpu(), teacher[b].detach().cpu(), mask[b].cpu()
                buf.append((s, float(kl_m[b]), rid, setting.failure_case(row, Vb, tb, mb, "code"),
                            setting.failure_case(row, Vb, tb, mb, "desc")))
                if len(buf) > cap:
                    buf.sort(key=lambda x: x[0], reverse=True)
                    del buf[cap:]

    def backward(self):
        items = []
        for p in self.bank.programs:
            buf = self.buf.get(p.uid, [])
            if not buf:
                continue
            buf.sort(key=lambda x: x[0], reverse=True)
            for s, kl, rid, cc, cd in buf[:MAX_GRADS_PER_BATCH]:
                items.append((p, kl, cc, cd))
                self.used[p.uid].add(rid)
            del buf[:MAX_GRADS_PER_BATCH]
        if items:
            self._gradients(items)

    # ---- update ---------------------------------------------------------------------------------------------------
    def _eval(self, predict, val_batches, override: Optional[Dict[int, Callable]] = None, idx: Optional[int] = None):
        """Bank-level mean KL on val_batches (router fixed), optionally with the program at position idx replaced;
        with idx, also the top failure cases of the replacement (KL x routing weight of idx)."""
        setting = self.bank.setting
        tot, n, cases = 0.0, 0, []
        for batch in val_batches:
            teacher, mask = setting.target(batch)
            with torch.no_grad():
                V_full, V_K, ok = self.bank.compute_probs(batch, override)
                if idx is not None and not bool(ok[idx]):
                    return float("inf"), []
                y_hat, w = predict(batch, V_full, V_K)
                kl = kl_rows(teacher, y_hat, mask)
            tot += float(kl.sum())
            n += kl.numel()
            if idx is not None:
                score = (kl * w[:, idx]).cpu()
                top_s, top_b = score.topk(min(20, score.numel()))
                for s, b in zip(top_s.tolist(), top_b.tolist()):
                    if s < 1e-6:
                        break
                    row = row_of(batch, b)
                    Vb, tb, mb = V_full[b, idx].cpu(), teacher[b].cpu(), mask[b].cpu()
                    cases.append((s, float(kl[b]), setting.failure_case(row, Vb, tb, mb, "code"),
                                  setting.failure_case(row, Vb, tb, mb, "desc")))
        return tot / max(n, 1), sorted(cases, key=lambda x: x[0], reverse=True)[:20]

    def _try(self, idx: int, code: str, description: str, mode: str, baseline: float, predict, val_batches):
        try:
            cand = self.bank.admit(code, description, mode, self.bank.programs[idx].added_at, exclude=idx)
        except ValueError as e:
            return None, float("inf"), [], str(e)
        kl, cases = self._eval(predict, val_batches, {idx: cand.fn}, idx)
        return cand, kl, cases, None

    def update(self, epoch: int, predict, val_batches: List[Dict]) -> int:
        """predict(batch, V_full, V_K) -> (y_hat, w) with the current router; val_batches: the evaluation batches (a
        uniform sample of the val split). Returns the number of accepted rewrites."""
        if not self.wkl:
            return 0
        dev = predict.device
        val_batches = [to_device(b, dev) for b in val_batches]
        baseline, _ = self._eval(predict, val_batches)
        mean_wkl = sum(self.wkl.values()) / max(len(self.wkl), 1)
        print(f"\n[TextGrad] epoch {epoch}: bank baseline KL {baseline:.4f}")
        accepted = 0
        for idx in range(len(self.bank.programs)):
            p = self.bank.programs[idx]
            wkl = self.wkl.get(p.uid, 0.0)
            if wkl < MIN_WEIGHTED_KL and wkl < KL_GOOD_FRACTION * mean_wkl:
                continue
            st = self._vars(p)
            new, new_kl = self._update_one(idx, st, baseline, predict, val_batches)
            rec = {"type": "textgrad", "epoch": epoch, "program": idx, "baseline_kl": baseline,
                   "accepted": new is not None, "kl": new_kl}
            if new is not None:
                self.state[new.uid] = self.state.pop(p.uid)
                self.bank.replace(idx, new)
                baseline = new_kl
                accepted += 1
                print(f"  H{idx}: accepted ({new.update_mode}), bank KL -> {baseline:.4f}")
            else:
                st["code"].set_value(p.code)
                st["desc"].set_value(p.description)
            self.log(rec)
        self.reset()
        return accepted

    def _update_one(self, idx, st, baseline, predict, val_batches):
        p = self.bank.programs[idx]
        accept = lambda kl: kl < baseline * (1 - self.cfg.min_improvement)   # noqa: E731
        budget, gate_retries = self.bank.budget, 0
        for r in range(self.cfg.rounds):
            if not st["code"].gradients:
                break
            old = st["code"].value
            if not optimizer_step(st["opt"], f"H{idx} r{r + 1}"):
                st["code"].set_value(old)
                break
            code = strip_code_fences(st["code"].value)
            st["code"].set_value(code)
            cand, kl, cases, err = self._try(idx, code, p.description, "textgrad", baseline, predict, val_batches)
            print(f"  H{idx} r{r + 1}: " + (f"rejected ({err})" if err else f"bank KL {kl:.4f} vs {baseline:.4f}"))
            if cand is not None and accept(kl):
                d = describe(self.llm, self.prompts, code)
                if d:
                    cand.description = d
                    st["desc"].set_value(d)
                    st["desc"].gradients = set()
                return cand, kl
            st["code"].set_value(old)
            if err and err.startswith(BUDGET_REJECT) and budget is not None and gate_retries < budget.tg_retries:
                gate_retries += 1     # keep the gradients and add the gate's reason: the next round fixes the violation
                self._add_grad(st["code"], self.prompts.render("budget.tg_reject", reason=err[len(BUDGET_REJECT):]),
                               "text gradient: program-budget violation")
                continue
            st["code"].gradients = set()
            if cases:
                self._gradients([(p, kl_c, cc, cd) for _, kl_c, cc, cd in cases[:RE_BACKWARD]])
        # phase 2: a rewritten description, implemented from scratch
        if not st["desc"].gradients:
            return None, None
        old_desc = st["desc"].value
        if not optimizer_step(st["dopt"], f"desc H{idx}"):
            return None, None
        new_desc = st["desc"].value
        for _ in range(REIMPLEMENT_ATTEMPTS):
            cand = implement(self.llm, self.prompts, self.bank, new_desc, "new_strategy", p.added_at)
            if cand is None:
                continue
            kl, _ = self._eval(predict, val_batches, {idx: cand.fn}, idx)
            if kl == float("inf"):
                continue
            if accept(kl):
                return cand, kl
            break
        st["desc"].set_value(old_desc)
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════════════════════ drop

def evaluate(bank: Bank, predict, batches, keep: Optional[List[int]] = None, cache: Optional[List] = None):
    """(mean KL, top-1 agreement) of the router mixture over batches; keep restricts the bank to those positions;
    cache: precomputed (V_full, V_K) per batch."""
    setting = bank.setting
    kl_sum, hit, n = 0.0, 0, 0
    for i, batch in enumerate(batches):
        teacher, mask = setting.target(batch)
        with torch.no_grad():
            if cache is not None:
                V_full, V_K = cache[i]
            else:
                V_full, V_K, _ = bank.compute_probs(batch)
            if keep is not None:
                V_full, V_K = V_full[:, keep], V_K[:, keep]
            y_hat, _ = predict(batch, V_full, V_K)
            kl = kl_rows(teacher, y_hat, mask)
            valid = mask.any(-1)
            hit += int((masked_argmax(y_hat, mask) == masked_argmax(teacher, mask))[valid].sum())
            n_valid = int(valid.sum())
        kl_sum += float(kl.sum())
        n += n_valid
    return kl_sum / max(n, 1), hit / max(n, 1)


def drop(bank: Bank, epoch: int, predict, eval_batches: List[Dict], val_loader, cfg, min_programs: int) -> Dict:
    """Leave-one-out drop (router fixed). On eval_batches (a uniform sample of the val split), repeatedly remove the program
    whose removal costs the least KL, as long as the cumulative changes stay within kl_tol / top1_tol and programs
    added in the last `grace_epochs` epochs are kept, up to `max_per_event` programs; then verify on the full val
    split and revert everything if the full-val KL rises by more than 2 kl_tol or top-1 falls by more than
    2 top1_tol."""
    dev = predict.device
    batches = [to_device(b, dev) for b in eval_batches]
    if not batches or len(bank) <= min_programs:
        return {"dropped": [], "reason": "nothing to do"}
    full_kl, full_t1 = evaluate(bank, predict, (to_device(b, dev) for b in val_loader))
    with torch.no_grad():
        cache = [bank.compute_probs(b)[:2] for b in batches]
    keep = list(range(len(bank)))
    base_kl, base_t1 = evaluate(bank, predict, batches, keep, cache)
    cur_kl, cur_t1, dropped, grace = base_kl, base_t1, [], cfg.grace_epochs
    for _ in range(cfg.max_per_event):
        if len(keep) <= min_programs:
            break
        best = None
        for m in keep:
            added = bank.programs[m].added_at
            if added is not None and epoch - added < grace:
                continue
            sub = [i for i in keep if i != m]
            kl, t1 = evaluate(bank, predict, batches, sub, cache)
            if kl - base_kl > cfg.kl_tol or base_t1 - t1 > cfg.top1_tol:
                continue
            if best is None or kl < best[1]:
                best = (m, kl, t1)
        if best is None:
            break
        keep.remove(best[0])
        dropped.append(best[0])
        cur_kl, cur_t1 = best[1], best[2]
        print(f"  [drop] H{best[0]}: subset KL {cur_kl:.4f} (base {base_kl:.4f}), top-1 {cur_t1:.4f} (base {base_t1:.4f})")
    if not dropped:
        return {"dropped": [], "reason": "no safe drop", "base_kl": base_kl, "base_top1": base_t1}
    snap = bank.snapshot()
    bank.remove(dropped)
    post_kl, post_t1 = evaluate(bank, predict, (to_device(b, dev) for b in val_loader))
    if post_kl > full_kl + 2 * cfg.kl_tol or post_t1 < full_t1 - 2 * cfg.top1_tol:
        bank.restore(snap)
        return {"dropped": [], "reason": "full-val regression, reverted", "candidates": dropped,
                "pre_kl": full_kl, "post_kl": post_kl, "pre_top1": full_t1, "post_top1": post_t1}
    return {"dropped": sorted(dropped), "pre_kl": full_kl, "post_kl": post_kl, "pre_top1": full_t1,
            "post_top1": post_t1, "reason": "accepted"}


# ═══════════════════════════════════════════════════════════════════════════════════════════════ add round

def _hits(bank: Bank, batch: Dict, programs: List[Program], t1: torch.Tensor) -> torch.Tensor:
    """[B, M] whether each program's argmax (programs whose output fell back count as misses) is the teacher's."""
    _, mask = bank.setting.target(batch)
    V_full, _, ok = bank.compute_probs(batch, programs=programs)
    hit = masked_argmax(V_full, mask.unsqueeze(1)) == t1.unsqueeze(1)
    return hit & ok.to(hit.device).unsqueeze(0)


def _parse_modes(text: str) -> List[Dict]:
    t = text.strip()
    if "```json" in t:
        t = t.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in t:
        t = t.split("```", 1)[1].split("```", 1)[0]
    try:
        modes = json.loads(t.strip())
    except Exception:  # noqa: BLE001
        s, e = t.find("["), t.rfind("]")
        modes = json.loads(t[s:e + 1]) if 0 <= s < e else None
    if not isinstance(modes, list):
        raise ValueError("cluster answer is not a JSON list")
    return [m for m in modes if isinstance(m, dict) and m.get("strategy")]


def select(bank: Bank, candidates: List[Program], val_loader, device, min_gain: float):
    """Greedy forward selection on the val split by oracle top-1 coverage (a row is covered when some program's
    argmax is the teacher's top-1): keep the candidate with the largest coverage gain while it is >= min_gain."""
    setting = bank.setting
    covered, hits = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = to_device(batch, device)
            teacher, mask = setting.target(batch)
            t1 = masked_argmax(teacher, mask)
            covered.append(_hits(bank, batch, bank.programs, t1).any(-1))
            hits.append(_hits(bank, batch, candidates, t1))
    covered, hits = torch.cat(covered), torch.cat(hits)
    baseline = float(covered.float().mean())
    picks, remaining = [], list(range(len(candidates)))
    while remaining:
        gains = {k: float((hits[:, k] & ~covered).float().mean()) for k in remaining}
        k = max(remaining, key=lambda j: (gains[j], -j))
        if gains[k] < min_gain:
            break
        picks.append({"candidate": k, "gain": gains[k]})
        covered |= hits[:, k]
        remaining.remove(k)
    return [p["candidate"] for p in picks], {"baseline": baseline, "picks": picks,
                                             "final_coverage": float(covered.float().mean())}


def add_round(bank: Bank, epoch: int, data, llm, prompts, cfg, tg_cfg, room: int, device, out_dir: str,
              log: Callable[[Dict], None], seed: int) -> List[Program]:
    """One add event. Returns the selected candidates (admitted, not yet in the bank); [] if nothing qualifies."""
    setting = bank.setting
    os.makedirs(out_dir, exist_ok=True)
    n_new = min(cfg.n_candidates, room)
    # 1. faults of the train split: rows where no program's argmax is the teacher's top-1
    fault_ids, fault_inst = [], []
    with torch.no_grad():
        for batch in data.scan(seed):
            batch = to_device(batch, device)
            teacher, mask = setting.target(batch)
            miss = ~_hits(bank, batch, bank.programs, masked_argmax(teacher, mask)).any(-1)
            fault_ids.append(setting.row_ids(batch)[miss].cpu())
            fault_inst.append(setting.instance_ids(batch)[miss].cpu())
            if sum(x.numel() for x in fault_ids) >= MAX_FAULTS:
                break
    fault_ids, fault_inst = torch.cat(fault_ids)[:MAX_FAULTS], torch.cat(fault_inst)[:MAX_FAULTS]
    print(f"  [add] {fault_ids.numel()} faults")
    log({"type": "add_faults", "epoch": epoch, "n_faults": int(fault_ids.numel())})
    if fault_ids.numel() < cfg.min_faults:
        return []
    # 2. fault split by instance: train rows drive the rewrites, held-out instances accept them
    inst = fault_inst.unique()
    g = torch.Generator().manual_seed(seed)
    inst = inst[torch.randperm(inst.numel(), generator=g)]
    val_inst = inst[:max(1, int(inst.numel() * FAULT_VAL_RATIO))]
    is_val = torch.isin(fault_inst, val_inst)
    ftrain, fval = fault_ids[~is_val], fault_ids[is_val]
    fval_batches = []
    for b in data.subset(fval, PRETRAIN_BATCH, True, seed):     # shuffled: LEHD batches are bucketed by sub-problem size
        if len(fval_batches) >= FAULT_EVAL_BATCHES:
            break
        fval_batches.append(to_device(b, device))
    # 3. cluster sampled faults into failure modes
    rows, V_banks, teachers, masks, scores = [], [], [], [], []
    sample = ftrain[torch.randperm(ftrain.numel(), generator=g)[:CLUSTER_CASES]]
    with torch.no_grad():
        for batch in data.subset(sample, 256, False, seed):
            batch = to_device(batch, device)
            teacher, mask = setting.target(batch)
            V_full, _, _ = bank.compute_probs(batch)
            min_kl = torch.stack([kl_rows(teacher, V_full[:, m], mask) for m in range(V_full.size(1))], -1).min(-1).values
            for b in range(teacher.size(0)):
                rows.append(row_of(batch, b))
                V_banks.append(V_full[b].cpu())
                teachers.append(teacher[b].cpu())
                masks.append(mask[b].cpu())
                scores.append(float(min_kl[b]))
    order = sorted(range(len(rows)), key=lambda i: -scores[i])
    cases = [setting.fault_case(rows[i], V_banks[i], teachers[i], masks[i]) for i in order]
    frozen = "".join(f"  {i + 1}. {p.description}\n" for i, p in enumerate(bank.programs))
    cases_text = "\n\n".join(f"Case {k + 1} [min program KL={scores[i]:.4f}]:\n{c}" for k, (i, c) in
                             enumerate(zip(order, cases)))
    summary = setting.fault_summary([rows[i] for i in order], [V_banks[i] for i in order],
                                    [teachers[i] for i in order], [masks[i] for i in order])
    modes = []
    for attempt in range(2):
        try:
            modes = _parse_modes(llm.call(prompts.render("cluster_system", n_modes=str(n_new)),
                                          prompts.render("cluster_user", n_modes=str(n_new), frozen=frozen,
                                                         summary=summary, cases=cases_text), source="cluster"))
            break
        except (LLMCallError, ValueError) as e:
            print(f"  [add] cluster attempt {attempt + 1} failed: {e}")
    modes = modes[:n_new]
    log({"type": "add_modes", "epoch": epoch, "modes": [m.get("mode_name") or m.get("name") for m in modes]})
    if not modes:
        return []
    # 4. one candidate per mode: re-implemented with feedback while its held-out fault hit rate is below the target
    #    SANITY_MIN_HIT; the best attempt is returned even if it stays below (step 6 keeps only candidates that add
    #    val oracle coverage)
    case_of = {k + 1: cases[k] for k in range(len(cases))}

    def build(mode):
        demos = "\n\n".join(case_of[i] for i in (mode.get("case_indices") or [])[:DEMOS_PER_MODE] if i in case_of)
        extra = {"frozen": frozen, "demos": demos}
        best, best_hit, note = None, -1.0, ""
        for _ in range(SANITY_ATTEMPTS):
            cand = implement(llm, prompts, bank, mode["strategy"] + note, "add", epoch, "add_implement_system",
                             "add_implement_user", extra, source="add_implement")
            if cand is None:
                break
            hit = _fault_hit(bank, cand, fval_batches)
            if hit > best_hit:
                best, best_hit = cand, hit
            if hit >= SANITY_MIN_HIT:
                break
            note = "\n\n" + prompts.render("retry_lowhit", best_hit=f"{hit:.3f}")
        if best is not None and best_hit < SANITY_MIN_HIT:
            name = mode.get("mode_name") or mode.get("name") or "?"
            print(f"  [add] mode {name!r}: best held-out fault hit {best_hit:.3f}, below the target {SANITY_MIN_HIT}; "
                  f"kept as a candidate (the selection by val coverage gain decides)")
            log({"type": "add_low_hit", "epoch": epoch, "mode": name, "hit": best_hit, "target": SANITY_MIN_HIT})
        return best
    cands = [c for c in llm.map(build, modes) if isinstance(c, Program)]
    if not cands:
        return []
    # 5. TextGrad pretrain of the candidates on the train faults; best epoch by union coverage of the held-out faults
    best_cov, best_set = _union_hit(bank, cands, fval_batches), list(cands)
    print(f"  [add] {len(cands)} candidates, held-out fault coverage {best_cov:.3f}")
    tg = TextGrad(_Pool(bank, cands), llm, prompts, tg_cfg, log)
    for e in range(1, cfg.pretrain_epochs + 1):
        for bi, batch in enumerate(data.subset(ftrain, PRETRAIN_BATCH, True, seed + e)):
            batch = to_device(batch, device)
            teacher, mask = setting.target(batch)
            with torch.no_grad():
                V_full, _, _ = bank.compute_probs(batch, programs=tg.bank.programs)
                kl = torch.stack([kl_rows(teacher, V_full[:, k], mask) for k in range(V_full.size(1))], -1)
                hit = (masked_argmax(V_full, mask.unsqueeze(1)) == masked_argmax(teacher, mask).unsqueeze(1)).float()
                w = torch.nn.functional.one_hot((hit * 1e6 - kl).argmax(-1), V_full.size(1)).float()
            tg.accumulate(w, V_full, teacher, mask, batch, setting.row_ids(batch))
            if (bi + 1) % PRETRAIN_BACKWARD_EVERY == 0:
                tg.backward()
                _rewrite(tg, fval_batches)
        cov = _union_hit(bank, tg.bank.programs, fval_batches)
        print(f"  [add] pretrain epoch {e}: held-out fault coverage {cov:.3f}")
        log({"type": "add_pretrain", "epoch": epoch, "pretrain_epoch": e, "coverage": cov})
        if cov > best_cov:
            best_cov, best_set = cov, list(tg.bank.programs)
    Bank.save(_Pool(bank, best_set), os.path.join(out_dir, "candidates.json"), {"epoch": epoch, "coverage": best_cov})
    # 6. selection on the val split
    picked, diag = select(bank, best_set, data.val, device, cfg.min_gain)
    picked = picked[:room]
    log({"type": "add_selection", "epoch": epoch, "n_candidates": len(best_set), "picked": picked, **diag})
    print(f"  [add] selected {len(picked)}/{len(best_set)} (val coverage {diag['baseline']:.4f} -> {diag['final_coverage']:.4f})")
    return [best_set[k] for k in picked]


class _Pool(Bank):
    """A bank view over a list of candidate programs (same setting, temperature and checks)."""

    def __init__(self, bank: Bank, programs: List[Program]):
        self.__dict__.update(bank.__dict__)
        self.programs = list(programs)
        self._parent = bank

    def admit(self, *args, **kwargs):
        return self._parent.admit(*args, **kwargs)


def _fault_hit(bank: Bank, cand: Program, batches) -> float:
    hit, n = 0, 0
    with torch.no_grad():
        for batch in batches:
            teacher, mask = bank.setting.target(batch)
            h = _hits(bank, batch, [cand], masked_argmax(teacher, mask))
            hit += int(h.sum())
            n += h.size(0)
    return hit / max(n, 1)


def _union_hit(bank: Bank, programs: List[Program], batches) -> float:
    hit, n = 0, 0
    with torch.no_grad():
        for batch in batches:
            teacher, mask = bank.setting.target(batch)
            h = _hits(bank, batch, programs, masked_argmax(teacher, mask)).any(-1)
            hit += int(h.sum())
            n += h.numel()
    return hit / max(n, 1)


def _rewrite(tg: TextGrad, fval_batches):
    """One TGD step for every candidate with gradients; accepted if its held-out fault coverage does not drop."""
    pool = tg.bank
    for k, p in enumerate(pool.programs):
        st = tg.state.get(p.uid)
        if st is None or not st["code"].gradients:
            continue
        cur = _fault_hit(pool, p, fval_batches)
        old = st["code"].value
        ok = optimizer_step(st["opt"], f"candidate {k}")
        accepted, new = False, cur
        if ok:
            code = strip_code_fences(st["code"].value)
            try:
                cand = pool.admit(code, p.description, "add_pretrained", p.added_at)
                new = _fault_hit(pool, cand, fval_batches)
                if new >= cur:
                    tg.state[cand.uid] = tg.state.pop(p.uid)
                    st["code"].set_value(code)
                    pool.replace(k, cand)
                    accepted = True
            except ValueError:
                pass
        if not accepted:
            st["code"].set_value(old)
        st["code"].gradients = set()
        st["desc"].gradients = set()
        tg.log({"type": "add_rewrite", "candidate": k, "coverage": cur, "new_coverage": new, "accepted": accepted})
