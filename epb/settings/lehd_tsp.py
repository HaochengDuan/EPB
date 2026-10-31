"""LEHD teacher on TSP: programs and router see the LEHD sub-problem (unvisited nodes + first node + current node).

Data: the per-step file of epb.teachers.lehd_tsp ({meta, instance_locs [I,N,2], tour [I,N],
per_step {step: {unselected_locs [I,K,2], first_loc, current_loc, teacher_probs [I,K], next_node_local,
first_node_idx, current_node_idx, unselected_idx [I,K], ...}}}, K = N - step), memory-mapped once. The split is by
instance; every batch holds one step, so K is uniform inside a batch, and carries the full-N fields (locs, mask,
teacher_probs over N, ...) plus the sub-graph fields the programs and the router read.

Programs: heuristic(unselected_locs [B,K,2], first_loc [B,2], current_loc [B,2]) -> [B,K] scores.
V_K = softmax(scores / tau_h) over the K candidates; V_full = the same logits scattered to their node ids, softmax
over the unvisited nodes (0 on visited ones).

Router: LEHD-style light embeddings of [first, unselected..., current] + pre-LN attention blocks; query from
[h_first, h_current] and the mean of h_unsel; dynamic keys K_m = V_K,m @ h_unsel.
"""
import math
import os
import queue
import subprocess
import sys
import threading
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Data, Setting, eval_option, register, state_convention

_GENERATOR = "lehd_tsp"               # the generator module, python -m epb.teachers.lehd_tsp
_ROW_SHIFT = 32                       # row id = step << 32 | instance index (instance index of the per-step file)
_STEP_FIELDS = ("unselected_locs", "first_loc", "current_loc", "teacher_probs", "next_node_local",
                "first_node_idx", "current_node_idx", "unselected_idx")


def _cfg(cfg, path: str, default=None):
    """cfg value at a dotted path (attribute or dict access); default if missing or None."""
    node = cfg
    for key in path.split("."):
        if node is None:
            return default
        node = node.get(key) if isinstance(node, dict) else getattr(node, key, None)
    return default if node is None else node


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask = mask.bool()
    has_valid = mask.any(dim=dim, keepdim=True)
    safe = logits.masked_fill(~mask, float("-inf"))
    safe = torch.where(has_valid, safe, torch.zeros_like(safe))
    probs = F.softmax(safe, dim=dim)
    probs = torch.where(mask, probs, torch.zeros_like(probs))
    probs = torch.where(has_valid, probs, torch.zeros_like(probs))
    return probs


# ═══════════════════════════════════════════════════════════════════════════════════════════════ data

def check_step_file(d: Dict, path: str = "") -> None:
    """Raise ValueError if `d` (a loaded per-step file) does not have the format of epb.teachers.lehd_tsp."""
    where = f" ({path})" if path else ""
    for key in ("meta", "instance_locs", "tour", "per_step"):
        if key not in d:
            raise ValueError(f"per-step file{where}: missing {key!r}")
    N = int(d["meta"]["N"])
    I = int(d["meta"]["n_instances"])
    if tuple(d["instance_locs"].shape) != (I, N, 2):
        raise ValueError(f"per-step file{where}: instance_locs {tuple(d['instance_locs'].shape)}, expected {(I, N, 2)}")
    if tuple(d["tour"].shape) != (I, N):
        raise ValueError(f"per-step file{where}: tour {tuple(d['tour'].shape)}, expected {(I, N)}")
    steps = sorted(d["per_step"])
    if not steps or steps != list(range(steps[0], N)):
        raise ValueError(f"per-step file{where}: steps {steps[:3]}..{steps[-3:]}, expected start..{N - 1}")
    for step in steps:
        buf, K = d["per_step"][step], N - step
        for key in _STEP_FIELDS:
            if key not in buf:
                raise ValueError(f"per-step file{where}: step {step} lacks {key!r}")
            if buf[key].size(0) != I or (key in ("unselected_locs", "teacher_probs", "unselected_idx")
                                         and buf[key].size(1) != K):
                raise ValueError(f"per-step file{where}: step {step} {key} {tuple(buf[key].shape)} (K={K}, I={I})")


def warm_page_cache(path: str, chunk: int = 64 << 20) -> None:
    """Read the file once sequentially so that the random row reads of the batches hit the page cache (on a disk,
    random reads from a cold memory map are much slower than from the cache). Uses no process memory beyond one
    chunk."""
    t0 = time.perf_counter()
    buf = memoryview(bytearray(chunk))
    n = 0
    with open(path, "rb", buffering=0) as fh:
        while True:
            k = fh.readinto(buf)
            if not k:
                break
            n += k
    print(f"[data] page cache warmed: {n / 1e9:.1f} GB in {time.perf_counter() - t0:.0f} s")


class StepFile:
    """The per-step file, memory-mapped: loaded once, rows are read on demand."""

    def __init__(self, path: str, warm_cache: bool = True):
        print(f"[data] {path} (mmap)")
        if warm_cache:
            warm_page_cache(path)
        d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        check_step_file(d, path)
        self.meta = d["meta"]
        self.N = int(self.meta["N"])
        self.n_instances = int(self.meta["n_instances"])
        self.instance_locs = d["instance_locs"]
        self.per_step = d["per_step"]
        self.steps = sorted(self.per_step)
        self.step_max = self.N - 1
        print(f"[data] N={self.N}, instances={self.n_instances}, steps={self.steps[0]}..{self.steps[-1]}")


def split_train_val(n_instances: int, val_ratio: float, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_instances, generator=g)
    n_val = int(n_instances * val_ratio)
    return perm[n_val:], perm[:n_val]


def build_batch(f: StepFile, step: int, inst: torch.Tensor) -> Dict[str, torch.Tensor]:
    """One same-step batch of instances `inst` (indices into the file): full-N fields + sub-graph fields.

    Every row has K = N - step candidates, so unselected_idx is a clean scatter index (no padding)."""
    buf = f.per_step[step]
    B = inst.size(0)
    N = f.N
    K = buf["unselected_locs"].size(1)

    unselected_idx = buf["unselected_idx"][inst].long()               # [B, K]
    unselected_locs = buf["unselected_locs"][inst].float()             # [B, K, 2]
    first_loc       = buf["first_loc"][inst].float()                  # [B, 2]
    current_loc     = buf["current_loc"][inst].float()                # [B, 2]
    teacher_probs_K = buf["teacher_probs"][inst].float()              # [B, K]
    first_node      = buf["first_node_idx"][inst].long()              # [B]
    current_node    = buf["current_node_idx"][inst].long()            # [B]
    next_node_local = buf["next_node_local"][inst].long()             # [B]  (idx in K)

    locs = f.instance_locs[inst].float()                               # [B, N, 2]

    # mask = True at unselected positions (= unvisited / feasible)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask.scatter_(1, unselected_idx, True)
    visited = (~mask).float()

    # teacher_probs K -> N (zeros at visited positions)
    teacher_probs = torch.zeros(B, N, dtype=torch.float32)
    teacher_probs.scatter_(1, unselected_idx, teacher_probs_K)

    step_t = torch.full((B,), float(step), dtype=torch.float32)
    step_max_t = torch.full((B,), float(f.step_max), dtype=torch.float32)
    step_norm = step_t / step_max_t.clamp_min(1.0)

    inst = inst.long()
    return {
        # full-N fields
        "locs":          locs,
        "mask":          mask,
        "visited":       visited,
        "current_node":  current_node,
        "first_node":    first_node,
        "step_i":        step_t,
        "step_max":      step_max_t,
        "teacher_probs": teacher_probs,
        # sub-graph fields used by the programs and the router
        "unselected_idx":  unselected_idx,
        "unselected_locs": unselected_locs,
        "first_loc":       first_loc,
        "current_loc":     current_loc,
        "step_norm":       step_norm,
        "next_node_local": next_node_local,   # [B] teacher's argmax in K
        # bookkeeping
        "_step":           torch.tensor(step, dtype=torch.long),
        "_K":              torch.tensor(K, dtype=torch.long),
        "_row":            (torch.full_like(inst, step) << _ROW_SHIFT) | inst,
        "_inst":           inst.clone(),
    }


class StepBucketLoader:
    """Same-step batches over `groups` = [(step, instance indices)], the old StepBucketBatchLoader (one group per
    step over the split's instances) and FaultSubsetLoader (given rows). With shuffle, the instances of each group
    and then the batch order are permuted by a generator seeded with epoch + seed_offset; the epoch counter moves
    after every full pass, or is set by set_epoch."""

    def __init__(self, f: StepFile, groups: List[Tuple[int, torch.Tensor]], batch_size: int, shuffle: bool,
                 seed_offset: int = 0, prefetch: int = 0):
        self.f = f
        self.groups = [(int(s), idx.long()) for s, idx in groups if idx.numel() > 0]
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed_offset = int(seed_offset)
        self.prefetch = int(prefetch)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """epoch counts from 1 like the original trainer, whose first pass used the generator seed seed_offset + 0
        (every epoch number gives its own order, 0 included)."""
        self._epoch = int(epoch) - 1

    def _build_batches(self) -> List[Tuple[int, torch.Tensor]]:
        all_batches: List[Tuple[int, torch.Tensor]] = []
        gen = torch.Generator()
        gen.manual_seed(self._epoch + self.seed_offset)
        for step, idx in self.groups:
            if self.shuffle:
                idx = idx[torch.randperm(idx.numel(), generator=gen)]
            for i in range(0, idx.numel(), self.batch_size):
                all_batches.append((step, idx[i:i + self.batch_size]))
        if self.shuffle and all_batches:
            order = torch.randperm(len(all_batches), generator=gen).tolist()
            all_batches = [all_batches[i] for i in order]
        return all_batches

    def __iter__(self):
        batches = self._build_batches()
        it = (build_batch(self.f, step, idx) for step, idx in batches)
        yield from (_prefetched(it, self.prefetch) if self.prefetch > 0 else it)
        self._epoch += 1

    def __len__(self) -> int:
        return sum((idx.numel() + self.batch_size - 1) // self.batch_size for _, idx in self.groups)


def _prefetched(it, depth: int):
    """Build the batches of `it` in a background thread (same order), `depth` ahead."""
    q: "queue.Queue" = queue.Queue(maxsize=depth)
    stop = threading.Event()
    end = object()

    def work():
        try:
            for x in it:
                if stop.is_set():
                    return
                q.put(x)
            q.put(end)
        except BaseException as e:  # noqa: BLE001  (re-raised in the consumer)
            q.put(e)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    try:
        while True:
            x = q.get()
            if x is end:
                return
            if isinstance(x, BaseException):
                raise x
            yield x
    finally:
        stop.set()
        while t.is_alive():
            try:
                q.get_nowait()
            except queue.Empty:
                t.join(0.01)


# ═══════════════════════════════════════════════════════════════════════════════════════════════ router

class _RouterDecoderLayer(nn.Module):
    """One self-attention + FFN block with residual + pre-LayerNorm. LEHD's teacher-side decoder has no LayerNorm;
    the student router keeps the pre-LN of the original student architecture."""

    def __init__(self, embed_dim: int, num_heads: int, ff_hidden: int, qkv_dim: int):
        super().__init__()
        self.h = num_heads
        self.qkv_dim = qkv_dim
        self.Wq = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.combine = nn.Linear(num_heads * qkv_dim, embed_dim)
        self.W1 = nn.Linear(embed_dim, ff_hidden)
        self.W2 = nn.Linear(ff_hidden, embed_dim)
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.norm_ffn  = nn.LayerNorm(embed_dim)

    def _multi_head(self, x):
        B, L, _ = x.shape
        q = self.Wq(x).view(B, L, self.h, self.qkv_dim).transpose(1, 2)
        k = self.Wk(x).view(B, L, self.h, self.qkv_dim).transpose(1, 2)
        v = self.Wv(x).view(B, L, self.h, self.qkv_dim).transpose(1, 2)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.qkv_dim)
        attn = F.softmax(score, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, -1)
        return self.combine(out)

    def forward(self, x):
        x = x + self._multi_head(self.norm_attn(x))
        x = x + self.W2(F.relu(self.W1(self.norm_ffn(x))))
        return x


class LEHDStyleRouterStudent(nn.Module):
    """LEHD-style router: 3 light embeddings + light encoder + shallow heavy decoder on the sub-graph
    [first_token, unselected_tokens..., current_token], producing `h` for both the query and the dynamic keys.

      h_first   = h[:, 0]            [B, D]
      h_unsel   = h[:, 1:-1]         [B, K, D]
      h_current = h[:, -1]           [B, D]
      Q = state_proj([h_first ‖ h_current]) + graph_proj(h_unsel.mean(1))
      K_dyn = V_K.detach() @ h_unsel  [B, M, D]
      scores = Q · K_dyn^T / τ
      w     = softmax(scores)
      y_hat = Σ_m w_m · V_full_m     [B, N]

    `codebook_keys` [M, D] is a buffer kept only so that checkpoints have every key of the original class (whose
    fixed-routing ablation used it); it takes no part in the routing. set_num_programs(M) resizes it (forward does so
    when the bank size changed), and loading a state dict resizes it to the stored shape.
    """

    def __init__(
        self,
        num_heuristics: int,
        embed_dim: int = 128,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        qkv_dim: int = 16,
        ff_hidden: int = 512,
        routing_temperature: Optional[float] = None,
        num_encoder_layers: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heuristics = num_heuristics
        self.routing_temperature = routing_temperature or (embed_dim ** 0.5)
        self.routing_mode = "dynamic"

        # Light encoder: 3 separate 2→D embeddings (LEHD-style anchors)
        self.embed_first      = nn.Linear(2, embed_dim)
        self.embed_unselected = nn.Linear(2, embed_dim)
        self.embed_current    = nn.Linear(2, embed_dim)

        # Encoder attention layers (cross-node mixing before the decoder stack)
        self.encoder_layers = nn.ModuleList([
            _RouterDecoderLayer(embed_dim, num_heads, ff_hidden, qkv_dim)
            for _ in range(num_encoder_layers)
        ])

        # Shallow heavy decoder
        self.decoder_layers = nn.ModuleList([
            _RouterDecoderLayer(embed_dim, num_heads, ff_hidden, qkv_dim)
            for _ in range(num_decoder_layers)
        ])

        # Q construction: two-path (state + graph)
        self.state_proj = nn.Linear(2 * embed_dim, embed_dim, bias=False)
        self.graph_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # checkpoint compatibility only (drawn like the original parameter, so the RNG stream is unchanged)
        self.register_buffer("codebook_keys", torch.randn(num_heuristics, embed_dim) * 0.02)
        self._register_load_state_dict_pre_hook(self._resize_codebook_keys, with_module=True)

    def set_num_programs(self, num_programs: int) -> None:
        old = self.codebook_keys
        M = int(num_programs)
        new = torch.zeros(M, old.size(1), dtype=old.dtype, device=old.device)
        n = min(M, old.size(0))
        new[:n] = old[:n]
        self.codebook_keys = new
        self.num_heuristics = M

    @staticmethod
    def _resize_codebook_keys(module, state_dict, prefix, *args):
        key = prefix + "codebook_keys"
        if key in state_dict and state_dict[key].shape != module.codebook_keys.shape:
            module.set_num_programs(state_dict[key].size(0))

    def encode_subgraph(self, unselected_locs, first_loc, current_loc):
        f = self.embed_first(first_loc).unsqueeze(1)         # [B, 1, D]
        u = self.embed_unselected(unselected_locs)           # [B, K, D]
        c = self.embed_current(current_loc).unsqueeze(1)     # [B, 1, D]
        x = torch.cat([f, u, c], dim=1)                      # [B, K+2, D]
        for layer in self.encoder_layers:
            x = layer(x)
        for layer in self.decoder_layers:
            x = layer(x)
        return x[:, 0], x[:, 1:-1], x[:, -1]                 # h_first, h_unsel, h_current

    def compute_query(self, h_first, h_unsel, h_current):
        state_ctx = self.state_proj(torch.cat([h_first, h_current], dim=-1))   # [B, D]
        graph_ctx = self.graph_proj(h_unsel.mean(dim=1))                        # [B, D]
        return state_ctx + graph_ctx                                            # [B, D]

    def forward(
        self,
        batch: Dict,                # carries unselected_locs, first_loc, current_loc
        V_full: torch.Tensor,       # [B, M, N]  scattered, used in the mixture
        V_K: torch.Tensor,          # [B, M, K]  pre-scatter, used to form K_dyn
        return_scores: bool = False,
    ):
        if V_full.size(1) != self.codebook_keys.size(0):     # bank resized: keep the checkpoint shape consistent
            self.set_num_programs(V_full.size(1))
        h_first, h_unsel, h_current = self.encode_subgraph(
            batch["unselected_locs"], batch["first_loc"], batch["current_loc"])
        q = self.compute_query(h_first, h_unsel, h_current)                     # [B, D]

        K_dyn = torch.bmm(V_K.detach(), h_unsel)                                 # [B, M, D]
        scores = torch.bmm(K_dyn, q.unsqueeze(-1)).squeeze(-1) / self.routing_temperature

        w = F.softmax(scores, dim=-1)                                            # [B, M]
        y_hat = (w.unsqueeze(-1) * V_full).sum(dim=1)                            # [B, N]
        if return_scores:
            return y_hat, w, scores
        return y_hat, w


# ═══════════════════════════════════════════════════════════════════════════════════════════════ failure text

def _topk_info(probs, locs, cur_pos, mask, k=2):
    # Geometry only: candidates have no stable id across steps (the program only sees unselected_locs).
    probs_m = probs.masked_fill(~mask, float("-inf"))
    result = []
    for idx in probs_m.argsort(descending=True):
        if not mask[idx]:
            continue
        pos  = (round(float(locs[idx, 0]), 3), round(float(locs[idx, 1]), 3))
        dist = round(float(((locs[idx, 0] - cur_pos[0]) ** 2 + (locs[idx, 1] - cur_pos[1]) ** 2) ** 0.5), 3)
        result.append({"pos": pos, "dist_to_cur": dist})
        if len(result) >= k:
            break
    return result


def _nearest_unvisited(locs, mask, cur_pos, n=10):
    result = []
    for i in range(locs.size(0)):
        if not mask[i]:
            continue
        pos  = (round(float(locs[i, 0]), 3), round(float(locs[i, 1]), 3))
        dist = round(float(((locs[i, 0] - cur_pos[0]) ** 2 + (locs[i, 1] - cur_pos[1]) ** 2) ** 0.5), 3)
        result.append({"pos": pos, "dist_to_cur": dist})
    result.sort(key=lambda x: x["dist_to_cur"])
    return result[:n]


def sample_info(row: Dict, V_m: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> Dict:
    """Geometry of one failure case (the old WeightedLLMHeuristicBank._make_sample_info, on one row)."""
    locs = row["locs"]
    cur_idx = int(row["current_node"])
    first_idx = int(row["first_node"])
    mask = mask.bool()
    cur_pos = (round(float(locs[cur_idx, 0]), 3), round(float(locs[cur_idx, 1]), 3))
    return {
        "step":              float(row["step_i"]),
        "step_max":          float(row["step_max"]),
        "n_unvisited":       int(mask.sum().item()),
        "cur_pos":           cur_pos,
        "first_pos":         (round(float(locs[first_idx, 0]), 3), round(float(locs[first_idx, 1]), 3)),
        "nearest_unvisited": _nearest_unvisited(locs, mask, cur_pos),
        "heuristic_top2":    _topk_info(V_m, locs, cur_pos, mask),
        "teacher_top2":      _topk_info(teacher, locs, cur_pos, mask),
    }


def fault_features(row: Dict) -> Dict:
    """Geometry-only features of one bank fault (no node ids, no step index): the old _featurize_fault_case."""
    cur_pos    = row["current_loc"]                            # [2]
    first_pos  = row["first_loc"]                              # [2]
    unsel      = row["unselected_locs"]                        # [K, 2]
    K          = unsel.size(0)
    teacher_local = int(row["next_node_local"])
    teacher_pos = unsel[teacher_local]

    d_to_cur   = (unsel - cur_pos.unsqueeze(0)).pow(2).sum(-1).sqrt()
    nearest_d  = float(d_to_cur.min().item())
    teacher_d  = float(d_to_cur[teacher_local].item())
    farness_ratio = teacher_d / max(nearest_d, 1e-9)

    fvec = cur_pos - first_pos
    tvec = teacher_pos - cur_pos
    fnorm = float(fvec.norm().item())
    tnorm = float(tvec.norm().item())
    if fnorm > 1e-6 and tnorm > 1e-6:
        forward_dot = float((fvec @ tvec).item() / (fnorm * tnorm))
    else:
        forward_dot = 0.0

    teacher_to_first = float((teacher_pos - first_pos).norm().item())

    return {
        "K":              K,
        "cur_pos":        (round(float(cur_pos[0]), 3), round(float(cur_pos[1]), 3)),
        "first_pos":      (round(float(first_pos[0]), 3), round(float(first_pos[1]), 3)),
        "teacher_pos":    (round(float(teacher_pos[0]), 3), round(float(teacher_pos[1]), 3)),
        "nearest_d":      round(nearest_d, 3),
        "teacher_d":      round(teacher_d, 3),
        "farness_ratio":  round(farness_ratio, 2),
        "forward_dot":    round(forward_dot, 2),
        "teacher_to_first_d": round(teacher_to_first, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════════════════════ program checks

def _edge_cases(g: torch.Generator) -> List[Tuple[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Degenerate but valid sub-problems (B = 4): (name, (unselected_locs, first_loc, current_loc))."""
    B = 4
    rnd = lambda *s: torch.rand(*s, generator=g)               # noqa: E731
    c = torch.full((B, 2), 0.5)
    rays = torch.tensor([[0.2, 0.0], [-0.2, 0.0], [0.0, 0.2], [0.0, -0.2], [0.4, 0.0], [0.0, 0.4]])
    corners = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.5], [1.0, 0.5]])
    first, cur = rnd(B, 2), rnd(B, 2)
    cases = [
        ("coincident candidates", (torch.full((B, 5, 2), 0.3), first, cur)),
        ("a candidate on the current node", (torch.cat([cur.unsqueeze(1), rnd(B, 4, 2)], 1), first, cur)),
        ("a candidate on the first node", (torch.cat([first.unsqueeze(1), rnd(B, 4, 2)], 1), first, cur)),
        ("current == first", (rnd(B, 6, 2), cur.clone(), cur)),
        ("all points coincide", (c.unsqueeze(1).repeat(1, 3, 1), c.clone(), c.clone())),
        ("all points coincide, K=1", (c.unsqueeze(1).clone(), c.clone(), c.clone())),
        ("K=1, candidate on the current node", (cur.unsqueeze(1).clone(), first, cur)),
        ("K=2", (rnd(B, 2, 2), first, cur)),
        ("axis-aligned rays from the current node", (c.unsqueeze(1) + rays.unsqueeze(0), rnd(B, 2), c.clone())),
        ("axis-aligned rays, first on a ray", (c.unsqueeze(1) + rays.unsqueeze(0), c + rays[0], c.clone())),
        ("coordinates 0 and 1", (corners.unsqueeze(0).repeat(B, 1, 1), torch.ones(B, 2), torch.zeros(B, 2))),
        ("coordinates 0 and 1, current == first", (corners.unsqueeze(0).repeat(B, 1, 1), torch.zeros(B, 2),
                                                   torch.zeros(B, 2))),
    ]
    return cases


def _run(fn, args, name: str, require_finite: bool) -> None:
    B, K = args[0].shape[:2]
    try:
        with torch.no_grad():
            out = fn(*(a.clone() for a in args))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"raises on {name} (B={B}, K={K}): {type(e).__name__}: {e}")
    if not isinstance(out, torch.Tensor):
        raise ValueError(f"returns {type(out).__name__} on {name}, expected a torch.Tensor [B, K]")
    if tuple(out.shape) != (B, K):
        raise ValueError(f"returns shape {tuple(out.shape)} on {name}, expected {(B, K)}")
    if require_finite and not torch.isfinite(out.float()).all():
        raise ValueError(f"non-finite output on {name} (B={B}, K={K})")


# ═══════════════════════════════════════════════════════════════════════════════════════════════ setting

@register
class LehdTsp(Setting):
    name = "lehd_tsp"
    problem = "tsp"
    # the training of the released student
    defaults = {
        "data": {"val_ratio": 0.1},
        "generate": {"problem_size": 100, "num_instances": 100000, "seed": 42, "batch_size": 1024,
                     "start_step": 2, "first_node": 0, "save_dtype": "float16", "device": "cuda",
                     "nco_root": None, "pomo_file": None},
        "bank": {"num_programs": 8, "min_programs": 4, "max_programs": 20, "tau_h": 0.03},
        "router": {"embed_dim": 128, "num_heads": 8, "num_encoder_layers": 1, "num_decoder_layers": 3,
                   "qkv_dim": 16, "ff_hidden": 512, "routing_temperature": 1.0},
        "train": {"epochs": 56, "batch_size": 1024, "lr": 1e-4, "scheduler": "cosine", "amp": False},
        "textgrad": {"start_epoch": 3, "every_n_epochs": 5, "backward_every_n": 200, "buffer_size": 10,
                     "rounds": 5, "min_improvement": 0.005, "eval_batches": 20},
        "drop": {"grace_epochs": 20},
        "add": {"n_candidates": 6, "min_gain": 0.001, "pretrain_epochs": 8, "finetune_epochs": 9,
                "cooldown_epochs": 9},
        "prompts": {"teacher": "LEHD"},
    }

    def __init__(self):
        self.problem_size: Optional[int] = None      # N, set by build_data (from the file's meta)

    # ---- data ---------------------------------------------------------------------------------------------------
    def build_data(self, cfg, split: Optional[Dict] = None) -> Data:
        """The file is memory-mapped once (after a sequential read that fills the page cache); train / val / scan /
        subset all read rows from it. cfg.data.num_workers > 0 builds that many batches ahead in a background thread
        (same order)."""
        f = StepFile(_cfg(cfg, "data.path"))
        batch_size = int(_cfg(cfg, "train.batch_size", self.defaults["train"]["batch_size"]))
        seed = int(_cfg(cfg, "run.seed", 42))
        prefetch = int(_cfg(cfg, "data.num_workers", 0))
        if split is None:
            n = f.n_instances
            val_ratio = float(_cfg(cfg, "data.val_ratio", self.defaults["data"]["val_ratio"]))
            train_idx, val_idx = split_train_val(n, val_ratio, seed)
            split = {"by": "instance", "n_instances": n, "val_ratio": val_ratio, "seed": seed,
                     "train": train_idx.tolist(), "val": val_idx.tolist()}
        else:
            train_idx = torch.tensor(split["train"], dtype=torch.long)
            val_idx = torch.tensor(split["val"], dtype=torch.long)
            seed = int(split.get("seed", seed))
            both = torch.cat([train_idx, val_idx])
            if both.numel() and (int(both.min()) < 0 or int(both.max()) >= f.n_instances):
                raise ValueError(f"split refers to instances outside the file (0..{f.n_instances - 1})")
            if both.unique().numel() != both.numel():
                raise ValueError("split: train and val instances overlap or repeat")
        print(f"[data] split: train_instances={train_idx.numel()}  val_instances={val_idx.numel()}")

        is_train = torch.zeros(f.n_instances, dtype=torch.bool)
        is_train[train_idx] = True
        train = StepBucketLoader(f, [(s, train_idx) for s in f.steps], batch_size, True, seed, prefetch)
        val = StepBucketLoader(f, [(s, val_idx) for s in f.steps], batch_size, False, seed, prefetch)

        def scan(scan_seed: int):
            return StepBucketLoader(f, [(s, train_idx) for s in f.steps], batch_size, True, scan_seed, prefetch)

        def subset(row_ids: torch.Tensor, bs: int, shuffle: bool, sub_seed: int):
            row_ids = torch.as_tensor(row_ids, dtype=torch.long).view(-1)
            steps, inst = (row_ids >> _ROW_SHIFT), row_ids & ((1 << _ROW_SHIFT) - 1)
            if inst.numel() and (int(inst.max()) >= f.n_instances or not bool(is_train[inst].all())):
                raise ValueError("subset: row ids outside the train split")
            groups: Dict[int, List[int]] = {}
            for s, i in zip(steps.tolist(), inst.tolist()):
                if s not in f.per_step:
                    raise ValueError(f"subset: row id of unknown step {s}")
                groups.setdefault(s, []).append(i)
            return StepBucketLoader(f, [(s, torch.tensor(v, dtype=torch.long)) for s, v in groups.items()],
                                    bs, shuffle, sub_seed, prefetch)

        self.problem_size = f.N
        return Data(train=train, val=val, problem_size=f.N, scan=scan, subset=subset,
                    set_epoch=train.set_epoch, split=split)

    def row_ids(self, batch: Dict) -> torch.Tensor:
        return batch["_row"]

    def instance_ids(self, batch: Dict) -> torch.Tensor:
        return batch["_inst"]

    def target(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        return batch["teacher_probs"], batch["mask"].bool()

    # ---- programs -----------------------------------------------------------------------------------------------
    def program_args(self, batch: Dict) -> tuple:
        return batch["unselected_locs"].float(), batch["first_loc"].float(), batch["current_loc"].float()

    def program_probs(self, fn: Callable, batch: Dict, tau_h: float):
        """The old WeightedLLMHeuristicBank.compute_probs for one program: an exception, a wrong shape or a
        non-finite output gives zeros (uniform over the candidates)."""
        mask = batch["mask"].bool()
        device = mask.device
        B, N = mask.shape
        unselected_idx = batch["unselected_idx"].long()                          # [B, K]
        K = unselected_idx.size(1)
        ok = True
        try:
            heu_K = fn(*self.program_args(batch))
            heu_K = heu_K.float()
            if heu_K.device != device:
                heu_K = heu_K.to(device)
            if heu_K.shape != (B, K):
                raise ValueError(f"shape {tuple(heu_K.shape)}, expected ({B},{K})")
            if not torch.isfinite(heu_K).all():
                bad_rows = int((~torch.isfinite(heu_K)).any(dim=-1).sum().item())
                raise ValueError(f"non-finite output (bad_rows={bad_rows}/{B}, K={K})")
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"program runtime error: {e}. Using zeros.")
            heu_K = torch.zeros(B, K, device=device)
            ok = False

        heu_K_sharp = heu_K / tau_h
        V_K = F.softmax(heu_K_sharp, dim=-1)                                     # [B, K]
        logits_full = torch.full((B, N), float("-inf"), device=device)
        logits_full.scatter_(1, unselected_idx, heu_K_sharp)
        V_full = masked_softmax(logits_full.masked_fill(~mask, float("-inf")), mask)
        return V_full, V_K, ok

    def check_program(self, fn: Callable) -> None:
        """The old signature smoke test (random B=4, K=17 and K=1: tensor of shape [B, K], finite), then degenerate
        geometry (coincident points, current == first, axis-aligned rays, coordinates 0 and 1, K = 1 and 2): must
        not raise and must return [B, K] (non-finite values there are left to the runtime fallback)."""
        g = torch.Generator().manual_seed(0)
        B_test, K_test = 4, 17
        unselected = torch.rand(B_test, K_test, 2, generator=g)
        first = torch.rand(B_test, 2, generator=g)
        current = torch.rand(B_test, 2, generator=g)
        _run(fn, (unselected, first, current), "a random state", True)
        _run(fn, (torch.rand(B_test, 1, 2, generator=g), first, current), "a random state with K=1", True)
        for name, args in _edge_cases(g):
            _run(fn, args, name, False)

    # ---- router -------------------------------------------------------------------------------------------------
    def build_router(self, cfg, num_programs: int) -> nn.Module:
        r = lambda k: _cfg(cfg, f"router.{k}", self.defaults["router"][k])     # noqa: E731
        return LEHDStyleRouterStudent(
            num_heuristics=num_programs,
            embed_dim=int(r("embed_dim")),
            num_decoder_layers=int(r("num_decoder_layers")),
            num_heads=int(r("num_heads")),
            qkv_dim=int(r("qkv_dim")),
            ff_hidden=int(r("ff_hidden")),
            routing_temperature=float(r("routing_temperature")),
            num_encoder_layers=int(r("num_encoder_layers")),
        )

    def router_forward(self, model: nn.Module, batch: Dict, V_full: torch.Tensor, V_K: torch.Tensor):
        return model(batch, V_full, V_K, return_scores=True)

    def ckpt_args(self, cfg, num_programs: int) -> Dict:
        """The keys epb.eval reads (and the original trainer's tools; its RRC evaluator takes the architecture on the
        command line, defaults = these values)."""
        r = lambda k: _cfg(cfg, f"router.{k}", self.defaults["router"][k])     # noqa: E731
        return {
            "num_heuristics": int(num_programs),
            "problem_size": int(getattr(self, "problem_size", None)
                                or _cfg(cfg, "generate.problem_size", self.defaults["generate"]["problem_size"])),
            "heuristic_temperature": float(_cfg(cfg, "bank.tau_h", self.defaults["bank"]["tau_h"])),
            "embed_dim": int(r("embed_dim")),
            "num_heads": int(r("num_heads")),
            "router_encoder_layers": int(r("num_encoder_layers")),
            "router_decoder_layers": int(r("num_decoder_layers")),
            "router_qkv_dim": int(r("qkv_dim")),
            "router_ff_hidden": int(r("ff_hidden")),
            "routing_temperature": float(r("routing_temperature")),
            "routing_mode": "dynamic",
            "router_arch": "lehd",
            "input": _cfg(cfg, "data.path"),
            "val_ratio": float(_cfg(cfg, "data.val_ratio", self.defaults["data"]["val_ratio"])),
            "seed": int(_cfg(cfg, "run.seed", 42)),
        }

    # ---- prompts ------------------------------------------------------------------------------------------------
    def prompt_vars(self, cfg, problem_size: int) -> Dict[str, str]:
        """teacher (prompts.teacher, "LEHD"); formulation ("<teacher>-style", the sub-problem formulation);
        problem_size; k_min / k_max (range of K); tau_h."""
        teacher = str(_cfg(cfg, "prompts.teacher", self.defaults["prompts"]["teacher"]))
        N = int(problem_size)
        return {
            "teacher": teacher,
            "formulation": f"{teacher}-style",
            "problem_size": str(N),
            "k_min": "1",
            "k_max": str(N - 1),
            "tau_h": f"{float(_cfg(cfg, 'bank.tau_h', self.defaults['bank']['tau_h'])):g}",
        }

    def failure_case(self, row: Dict, V_m: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor,
                     kind: str) -> str:
        """The case lines of the old gradient user prompts (6-space indented, as they appeared there)."""
        info = sample_info(row, V_m, teacher, mask)
        step = info["step"]
        step_max = max(info["step_max"], 1)
        h2 = [f"pos={x['pos']} d={x['dist_to_cur']}" for x in info["heuristic_top2"]]
        t2 = [f"pos={x['pos']} d={x['dist_to_cur']}" for x in info["teacher_top2"]]
        lines = [f"Step: {int(step)}/{int(step_max)} ({100 * int(step) // int(step_max)}% complete)"]
        if kind == "code":
            nu = [f"pos={x['pos']} d={x['dist_to_cur']}" for x in info["nearest_unvisited"][:5]]
            lines += [
                f"Unvisited candidates remaining (K): {info['n_unvisited']}",
                f"Current position: {info['cur_pos']}",
                f"Tour-start position: {info['first_pos']}",
                f"5 nearest unvisited (geometry only): {nu}",
                f"Heuristic top-2: {h2}",
                f"Teacher   top-2: {t2}",
            ]
        elif kind == "desc":
            lines += [
                f"Current pos: {info['cur_pos']}",
                f"Heuristic top-2 (geometry only): {h2}",
                f"Teacher   top-2 (geometry only): {t2}",
            ]
        else:
            raise ValueError(f"unknown failure-case kind {kind!r}")
        return "\n".join("      " + ln for ln in lines)

    def fault_case(self, row: Dict, V_bank: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> str:
        """The old cluster-prompt case line (geometry of the teacher's pick only)."""
        f = fault_features(row)
        return (f"  K={f['K']}, cur={f['cur_pos']}, first={f['first_pos']}, "
                f"teacher_pick={f['teacher_pos']}, "
                f"farness_ratio={f['farness_ratio']}, forward_dot={f['forward_dot']}, "
                f"teacher_to_first_d={f['teacher_to_first_d']}")

    # ---- data generation ----------------------------------------------------------------------------------------
    def generate(self, cfg) -> None:
        """Run the generator python -m epb.teachers.lehd_tsp in a process of its own (greedy LEHD roll-outs from
        node 0, steps >= start_step stored) on fresh uniform instances (or on the instances of a POMO decision-step file, cfg.generate.pomo_file), then
        check the written file."""
        g = lambda k: _cfg(cfg, f"generate.{k}", self.defaults["generate"].get(k))   # noqa: E731
        ckpt, out = g("teacher_ckpt"), _cfg(cfg, "data.path")
        if not ckpt:
            raise ValueError("generate.teacher_ckpt (a LEHD-TSP checkpoint in the NCO_code format) is required")
        if not out:
            raise ValueError("data.path (the output file) is required")
        from ..teachers import command
        args = ["--ckpt", ckpt, "--problem-size", g("problem_size"),
                "--first-node", g("first_node"), "--start-step", g("start_step"),
                "--batch-size", g("batch_size"), "--device", g("device"),
                "--save-dtype", g("save_dtype"), "--seed", g("seed"), "--output", out]
        if g("pomo_file"):
            args += ["--instance-source", "pomo", "--pomo-file", g("pomo_file")]
            if g("num_instances"):
                args += ["--n-instances", g("num_instances")]
        else:
            args += ["--instance-source", "fresh", "--n-instances", g("num_instances")]
        if g("nco_root"):
            args += ["--nco-root", nco_dir(g("nco_root"))]
        cmd, env = command(_GENERATOR, args)
        print("[generate] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=env)
        d = torch.load(out, map_location="cpu", weights_only=False, mmap=True)
        check_step_file(d, out)
        if int(d["meta"]["N"]) != int(g("problem_size")):
            raise ValueError(f"{out}: N={d['meta']['N']}, expected {g('problem_size')}")
        print(f"[generate] {out}: N={d['meta']['N']}, instances={d['meta']['n_instances']}, "
              f"steps {min(d['per_step'])}..{max(d['per_step'])}")

    # ---- evaluation (epb.eval) ----------------------------------------------------------------------------------
    search_default = "rrc"
    budget_default = {100: 500, 500: 500}

    def eval_args(self, parser) -> None:
        parser.add_argument("--rrc-seed", type=int, default=RRC_SEED,
                            help="seed of the RRC segment stream, reseeded at every batch (default %(default)s)")

    def eval_policy(self, method: str, a):
        """teacher: the LEHD model of --teacher-ckpt (code from --nco-root). epb / expert / frozen: the router of
        --ckpt over the bank of --bank, decoded with --k-mode / --v-mode (default soft, the trained forward), --tau-h
        (default: the checkpoint's heuristic_temperature, then the setting's) and --drop. nam / moet / tree / agent:
        the package of --pkg (tree: --depth)."""
        if method == "teacher":
            return TeacherPolicy(load_lehd(a.teacher_ckpt, a.nco_root, a.device))
        if method in ("epb", "expert", "frozen"):
            return StudentPolicy.load(self, a)
        if method in ("nam", "moet"):
            return ScorerPolicy.load(self, method, a)
        if method == "tree":
            return TreeSurrogatePolicy.load(self, a)
        if method == "agent":
            return load_agent(a)
        raise ValueError(f"unknown method {method!r}")

    def eval_greedy(self, policy, inst: Dict[str, torch.Tensor], a) -> torch.Tensor:
        """Greedy from node 0, in the batches of eval_batches."""
        out = []
        for lo, hi, locs in eval_batches(inst, a):
            tour, _ = policy.rollout(locs)
            out.append(tour_length(locs, tour).cpu())
        return torch.cat(out)

    def eval_search(self, policy, inst: Dict[str, torch.Tensor], a) -> torch.Tensor:
        """RRC for a.budget iterations from the greedy tour of each batch; the segment stream is reseeded with
        a.rrc_seed at every batch, so every method sees the same destroy sequence."""
        if a.search != "rrc":
            raise ValueError(f"{self.name}: search {a.search!r} is not supported (rrc)")
        out = []
        for lo, hi, locs in eval_batches(inst, a):
            t0 = time.time()
            tour, _ = policy.rollout(locs)
            tour = rrc(policy, locs, tour, a.budget, a.rrc_seed)
            out.append(tour_length(locs, tour).cpu())
            print(f"[eval] rrc-{a.budget} batch {lo}:{hi}  mean {out[-1].mean().item():.4f}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
        return torch.cat(out)

    def eval_top1(self, teacher, policy, inst: Dict[str, torch.Tensor], a) -> Dict[str, torch.Tensor]:
        """Along the teacher's greedy tour (from node 0) the policy picks from every teacher state. Primary states:
        steps 2 .. N-2 (step 1 is symmetric under the tour direction, step N-1 is forced)."""
        hits, hits_all = [], []
        for lo, hi, locs in eval_batches(inst, a):
            tour, _ = teacher.rollout(locs)
            _, picks = policy.rollout(locs, forced=tour)
            hit = picks == tour[:, 1:]                                   # [B, N-1], column t-1 = step t
            N = locs.size(1)
            steps = torch.arange(1, N, device=hit.device)
            primary = (N - steps > 1) & (steps != 1)
            hits.append(hit[:, primary].sum(1).double().cpu())
            hits_all.append(hit.sum(1).double().cpu())
            n_primary, n_all = int(primary.sum()), N - 1
        hits, hits_all = torch.cat(hits), torch.cat(hits_all)
        return {"hits": hits, "states": torch.full_like(hits, float(n_primary)),
                "hits_all": hits_all, "states_all": torch.full_like(hits_all, float(n_all))}


# ═══════════════════════════════════════════════════════════════════════════════════════════════ evaluation
#
# Greedy decoding from node 0 and random re-construct (RRC, the LEHD paper's search): starting from the greedy tour,
# every iteration draws one segment (start position, length in [4, N], direction) shared by all instances of the
# batch, re-decodes the interior of the segment with both endpoints fixed and keeps the new sub-path where it is
# shorter. The draws come from one CPU generator per batch, so a batch is part of the protocol: the main set (N=100)
# is evaluated in batches of 50 instances, the N=500 set in batches of 32, on a grid of the set's instance index (a
# slice evaluates the batches it covers, a slice that cuts a batch gives other results for it).
#
# The teacher re-decodes a segment in LEHD's repair mode (selected = [last, first], greedy fill). A router-bank
# student has no repair mode: it decodes from the first endpoint with the last one masked until it is the only
# unvisited node.

RRC_SEED = 1334
RRC_SUB_MIN = 4                       # shortest segment (two endpoints + two re-decoded nodes)
EVAL_BATCH = {100: 50, 500: 32}       # N -> instances per batch; other N: EVAL_BATCH_OTHER
EVAL_BATCH_OTHER = 50
TEACHER_CHUNK = 64                    # rows per teacher forward pass (greedy and repair)
LEHD_PARAMS = {"mode": "test", "embedding_dim": 128, "sqrt_embedding_dim": 128 ** 0.5, "decoder_layer_num": 6,
               "qkv_dim": 16, "head_num": 8, "ff_hidden_dim": 512}
ROUTER_ARGS = {"embed_dim": "embed_dim", "num_heads": "num_heads", "router_encoder_layers": "num_encoder_layers",
               "router_decoder_layers": "num_decoder_layers", "router_qkv_dim": "qkv_dim",
               "router_ff_hidden": "ff_hidden", "routing_temperature": "routing_temperature"}   # ckpt args -> router


def eval_batches(inst: Dict[str, torch.Tensor], a):
    """(lo, hi, locs on a.device) per batch of a.batch instances (default EVAL_BATCH[N]); the batch grid is the one
    of the full set (a.slice "lo:hi" gives the offset)."""
    locs_all = inst["locs"]
    n, N = locs_all.shape[:2]
    bs = int(a.batch or EVAL_BATCH.get(N, EVAL_BATCH_OTHER))
    off = int(a.slice.split(":")[0]) if getattr(a, "slice", None) else 0
    cuts = [0] + [i for i in range(1, n) if (off + i) % bs == 0] + [n]
    if getattr(a, "slice", None) and (off % bs or (off + n) % bs):
        print(f"[eval] note: the slice cuts a batch of {bs} (RRC is batch-coupled; those instances get other "
              f"search results than in the full run)", flush=True)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        yield lo, hi, locs_all[lo:hi].to(a.device)


def tour_length(locs: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    """Closed-tour length [B]; tour [B, N] node ids."""
    g = torch.gather(locs, 1, tour.unsqueeze(-1).expand(-1, -1, 2))
    return torch.norm(g - torch.roll(g, shifts=-1, dims=1), dim=-1).sum(-1)


def path_length(locs: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
    """Open-path length [B]; path [B, L] node ids."""
    g = torch.gather(locs, 1, path.unsqueeze(-1).expand(-1, -1, 2))
    return torch.norm(g[:, 1:] - g[:, :-1], dim=-1).sum(-1)


@torch.no_grad()
def rrc(policy, locs: torch.Tensor, tours: torch.Tensor, budget: int, seed: int, sub_min: int = RRC_SUB_MIN,
        log_every: Optional[int] = None) -> torch.Tensor:
    """RRC on one batch: tours [B, N] -> improved tours. policy.repair(sub_locs [B, L, 2], N) -> [B, L] local ids of
    a path from local node 0 to local node L-1. The segment is cut from the doubled (possibly flipped) tour and the
    tour is kept in the representation it was spliced in (policy.keep_flip: also when nothing is accepted)."""
    B, N, _ = locs.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    keep_flip = getattr(policy, "keep_flip", False)
    log_every = log_every or max(1, budget // 4)
    n_acc = 0
    for it in range(budget):
        start = int(torch.randint(0, N, (1,), generator=g).item())
        length = int(torch.randint(sub_min, N + 1, (1,), generator=g).item())
        flip = bool(torch.rand(1, generator=g).item() < 0.5)
        tours_it = torch.flip(tours, dims=[1]) if flip else tours
        double = torch.cat([tours_it, tours_it], dim=1)                   # [B, 2N]
        seg = double[:, start:start + length]                             # [B, L] node ids, endpoints kept
        sub_locs = locs.gather(1, seg.unsqueeze(-1).expand(B, length, 2))
        old_len = path_length(locs, seg)
        new_seg = seg.gather(1, policy.repair(sub_locs, N))
        accept = path_length(locs, new_seg) < old_len
        if accept.any():
            spliced = torch.cat([double[:, :start], new_seg, double[:, start + length:]], dim=1)[:, start:start + N]
            tours = torch.where(accept.unsqueeze(1), spliced, tours_it)
            n_acc += int(accept.sum().item())
        elif keep_flip:
            tours = tours_it
        if (it + 1) % log_every == 0:
            print(f"  [rrc] iter {it + 1}/{budget}  mean {tour_length(locs, tours).mean().item():.4f}  "
                  f"accepted {n_acc}", flush=True)
    return tours


# ---- teacher ------------------------------------------------------------------------------------------------------

def nco_dir(nco_root: Optional[str]) -> str:
    """The directory holding the LEHD package: <clone>/single_objective, or nco_root itself (env NCO_ROOT if unset)."""
    root = nco_root or os.environ.get("NCO_ROOT")
    if not root:
        raise ValueError("the LEHD teacher needs --nco-root (a clone of CIAM-Group/NCO_code)")
    root = os.path.abspath(os.path.expanduser(str(root)))
    sub = os.path.join(root, "single_objective")
    return sub if os.path.isdir(os.path.join(sub, "LEHD")) else root


def load_lehd(ckpt: Optional[str], nco_root: Optional[str], device) -> nn.Module:
    """NCO_code LEHD TSPModel (test mode) with the weights of a checkpoint in the NCO_code format."""
    if not ckpt:
        raise ValueError("--teacher-ckpt (a LEHD-TSP checkpoint in the NCO_code format) is required")
    d = nco_dir(nco_root)
    if d not in sys.path:
        sys.path.insert(0, d)
    from LEHD.TSP.TSPModel import TSPModel
    model = TSPModel(**LEHD_PARAMS).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model_state_dict"])
    return model.eval()


@torch.no_grad()
def lehd_step(model: nn.Module, encoded: torch.Tensor, selected: torch.Tensor, return_probs: bool = False
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """One LEHD decoder step on [first, unselected..., last] with selected = [first, ..., last] (node ids into
    encoded [B, N, D]): (argmax position in the unselected nodes [B], unselected node ids [B, K] ascending); with
    return_probs, the next-node probabilities over the unselected nodes [B, K] in place of the argmax."""
    dec = model.decoder
    B, N, D = encoded.shape
    T = selected.size(1)
    K = N - T
    ids = torch.arange(N, device=encoded.device).unsqueeze(0).repeat(B, 1)
    ids[torch.arange(B, device=encoded.device).unsqueeze(1).expand(B, T), selected] = -2
    unsel_idx = ids[ids >= 0].view(B, K)
    unsel = encoded.gather(1, unsel_idx.unsqueeze(-1).expand(B, K, D))
    first = encoded.gather(1, selected[:, 0:1].unsqueeze(-1).expand(B, 1, D)).squeeze(1)
    last = encoded.gather(1, selected[:, -1:].unsqueeze(-1).expand(B, 1, D)).squeeze(1)
    out = torch.cat([dec.embedding_first_node(first).unsqueeze(1), unsel, dec.embedding_last_node(last).unsqueeze(1)],
                    dim=1)
    for layer in dec.layers:
        out = layer(out)
    out = dec.Linear_final(out).squeeze(-1)
    out[:, 0] = float("-inf")
    out[:, -1] = float("-inf")
    probs = F.softmax(out, dim=-1)[:, 1:-1]
    return (probs if return_probs else probs.argmax(dim=-1)), unsel_idx


class TeacherPolicy:
    """The LEHD teacher. rollout: greedy from node 0 (or teacher-forced along `forced`); repair: LEHD's repair
    decoding of a segment (selected = [last, first], greedy fill)."""

    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def rollout(self, locs: torch.Tensor, forced: Optional[torch.Tensor] = None):
        """(tour [B, N], picks [B, N-1]): picks[:, t-1] is the argmax at step t; the state follows the picks, or
        forced[:, t] when forced (a tour from node 0) is given."""
        B, N, _ = locs.shape
        tours, picks = [], []
        for s in range(0, B, TEACHER_CHUNK):
            encoded = self.model.encoder(locs[s:s + TEACHER_CHUNK])
            selected = torch.zeros(encoded.size(0), 1, dtype=torch.long, device=locs.device)
            p = []
            for step in range(1, N):
                nxt_local, unsel_idx = lehd_step(self.model, encoded, selected)
                pick = unsel_idx.gather(1, nxt_local.unsqueeze(-1))
                p.append(pick)
                nxt = pick if forced is None else forced[s:s + TEACHER_CHUNK, step:step + 1]
                selected = torch.cat([selected, nxt], dim=1)
            tours.append(selected)
            picks.append(torch.cat(p, dim=1))
        return torch.cat(tours), torch.cat(picks)

    @torch.no_grad()
    def repair(self, sub_locs: torch.Tensor, N: int) -> torch.Tensor:
        """Paths [B, L] (local ids) from local node 0 to local node L-1; N: size of the instance (sets the chunk)."""
        B, L, _ = sub_locs.shape
        chunk = max(1, min(TEACHER_CHUNK, int(2.0e8 / max(1, N * N))))
        paths = []
        for s in range(0, B, chunk):
            encoded = self.model.encoder(sub_locs[s:s + chunk])
            b = encoded.size(0)
            selected = torch.stack([torch.full((b,), L - 1, dtype=torch.long, device=sub_locs.device),
                                    torch.zeros(b, dtype=torch.long, device=sub_locs.device)], dim=1)
            for step in range(2, L):
                nxt_local, unsel_idx = lehd_step(self.model, encoded, selected)
                selected = torch.cat([selected, unsel_idx.gather(1, nxt_local.unsqueeze(-1))], dim=1)
            paths.append(torch.cat([selected[:, 1:], selected[:, 0:1]], dim=1))      # [first, ..., last]
        return torch.cat(paths)


# ---- per-step policies -------------------------------------------------------------------------------------------

class StepPolicy:
    """A policy that picks one node per state of a constructive TSP rollout over the full graph: begin(locs, first,
    last) -> ctx once per rollout, choose(ctx, current, mask, step) -> [B] node ids (mask: the nodes it may pick),
    commit(ctx, node) after every step. rollout: greedy from node 0 (or teacher-forced along `forced`); repair: a
    segment from local node 0 to local node L-1, which is masked until it is the only node left, in chunks of
    repair_chunk(N) rows (None: the whole batch). keep_flip: the RRC loop keeps the flipped tour also when no instance
    of the batch accepts (the tree package's loop)."""
    keep_flip = False

    def repair_chunk(self, N: int) -> Optional[int]:
        return None

    def begin(self, locs: torch.Tensor, first: torch.Tensor, last: Optional[torch.Tensor]) -> Dict:
        return {"locs": locs, "first": first, "ar": torch.arange(locs.size(0), device=locs.device)}

    def choose(self, ctx: Dict, current: torch.Tensor, mask: torch.Tensor, step: int) -> torch.Tensor:
        raise NotImplementedError

    def commit(self, ctx: Dict, node: torch.Tensor) -> None:
        pass

    @torch.no_grad()
    def rollout(self, locs: torch.Tensor, forced: Optional[torch.Tensor] = None,
                first: Optional[torch.Tensor] = None, last: Optional[torch.Tensor] = None):
        """(tour [B, N], picks [B, N-1]): picks[:, t-1] is the pick at step t; the state follows the picks, or
        forced[:, t] when forced is given. Starts at `first` (default node 0); `last` is masked until the last step."""
        B, N, _ = locs.shape
        ar = torch.arange(B, device=locs.device)
        if first is None:
            first = torch.zeros(B, dtype=torch.long, device=locs.device)
        visited = torch.zeros(B, N, dtype=torch.bool, device=locs.device)
        visited[ar, first] = True
        ctx = self.begin(locs, first, last)
        current = first.clone()
        tour, picks = [first.clone()], []
        for step in range(1, N):
            mask = ~visited
            if last is not None and step < N - 1:
                mask[ar, last] = False
            pick = self.choose(ctx, current, mask, step)
            picks.append(pick)
            current = pick if forced is None else forced[:, step]
            self.commit(ctx, current)
            visited[ar, current] = True
            tour.append(current.clone())
        return torch.stack(tour, dim=1), torch.stack(picks, dim=1)

    @torch.no_grad()
    def repair(self, sub_locs: torch.Tensor, N: int) -> torch.Tensor:
        """Paths [B, L] (local ids) from local node 0 to local node L-1; N: size of the instance (sets the chunk)."""
        B, L, _ = sub_locs.shape
        chunk = self.repair_chunk(N) or B
        first = torch.zeros(B, dtype=torch.long, device=sub_locs.device)
        last = torch.full((B,), L - 1, dtype=torch.long, device=sub_locs.device)
        return torch.cat([self.rollout(sub_locs[s:s + chunk], first=first[s:s + chunk], last=last[s:s + chunk])[0]
                          for s in range(0, B, chunk)])


def sub_problem(ctx: Dict, current: torch.Tensor, mask: torch.Tensor) -> Dict:
    """The LEHD sub-problem batch of a state (every row has the same number of candidates)."""
    locs, ar = ctx["locs"], ctx["ar"]
    unsel_idx = mask.nonzero(as_tuple=False)[:, 1].view(locs.size(0), -1)
    return {"mask": mask, "unselected_idx": unsel_idx,
            "unselected_locs": locs.gather(1, unsel_idx.unsqueeze(-1).expand(-1, -1, 2)),
            "first_loc": locs[ar, ctx["first"]], "current_loc": locs[ar, current]}


def _student_chunk(N: int) -> int:
    return min(64, max(8, max(1, int(1.0e8 / max(1, N ** 3))) * 4))


def _import_from(directory: str, module: str):
    """Import `module` with `directory` first on sys.path (the baseline packages import their siblings by name)."""
    import importlib
    d = os.path.abspath(os.path.expanduser(directory))
    if not os.path.isdir(d):
        raise ValueError(f"{directory}: not a directory")
    if d not in sys.path:
        sys.path.insert(0, d)
    return importlib.import_module(module)


def _trained_size_dir(setting: Setting) -> str:
    return f"{setting.name}{setting.defaults['generate']['problem_size']}"


# ---- router-bank student --------------------------------------------------------------------------------------------

def route(setting: Setting, model: nn.Module, batch: Dict, V_full: torch.Tensor, V_K: torch.Tensor,
          k_mode: str = "soft", v_mode: str = "soft", drop: Optional[List[int]] = None) -> torch.Tensor:
    """y_hat [B, N] with the eval-time decoding options. k_mode onehot: K_m = h_unsel at the argmax of V_K,m (soft:
    V_K,m @ h_unsel, the trained keys); v_mode onehot: program m votes for its argmax node (soft: V_full,m); drop:
    slots removed from the routing softmax. soft / soft without drop is the training forward."""
    if k_mode == "soft" and v_mode == "soft" and not drop:
        return setting.router_forward(model, batch, V_full, V_K)[0]
    h_first, h_unsel, h_current = model.encode_subgraph(batch["unselected_locs"], batch["first_loc"],
                                                        batch["current_loc"])
    q = model.compute_query(h_first, h_unsel, h_current)
    argmax_k = V_K.argmax(dim=-1)                                                 # [B, M]
    if k_mode == "onehot":
        K_dyn = h_unsel.gather(1, argmax_k.unsqueeze(-1).expand(-1, -1, h_unsel.size(-1)))
    else:
        K_dyn = torch.bmm(V_K, h_unsel)
    scores = torch.bmm(K_dyn, q.unsqueeze(-1)).squeeze(-1) / model.routing_temperature
    if drop:
        scores[:, drop] = float("-inf")
    w = F.softmax(scores, dim=-1)
    if v_mode == "onehot":
        V_use = F.one_hot(batch["unselected_idx"].gather(1, argmax_k), num_classes=V_full.size(-1)).float()
    else:
        V_use = V_full
    return (w.unsqueeze(-1) * V_use).sum(dim=1)


def _guard_fallbacks() -> None:
    """Every program fallback is shown; a program that runs out of memory aborts the run (it would be disabled, which
    changes the method being evaluated)."""
    warnings.filterwarnings("always", message=".*runtime error.*")
    warnings.filterwarnings("error", message=".*out of (GPU )?memory.*")


class StudentPolicy(StepPolicy):
    """Router + bank (epb / expert / frozen): the argmax of y_hat over the sub-problem of every state."""

    def __init__(self, setting: Setting, model: nn.Module, bank, k_mode: str, v_mode: str,
                 drop: Optional[List[int]] = None):
        self.setting, self.model, self.bank = setting, model, bank
        self.k_mode, self.v_mode, self.drop = k_mode, v_mode, list(drop or [])

    @classmethod
    def load(cls, setting: Setting, a) -> "StudentPolicy":
        from ..bank import Bank
        if not a.ckpt or not a.bank:
            raise ValueError("--ckpt (a directory with best_model.pt and bank.json, or a .pt with --bank) is required")
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        args = ck.get("args") or {}
        args = args if isinstance(args, dict) else vars(args)
        # flag / decode.json > the checkpoint's heuristic_temperature > the setting's bank.tau_h
        tau_h = float(eval_option(a, "tau_h", args, "heuristic_temperature", setting.defaults["bank"]["tau_h"]))
        state_convention(a, args)        # the evaluator's sub-problems equal the generator's rows under both
        k_mode, v_mode = a.k_mode or "soft", a.v_mode or "soft"
        _guard_fallbacks()
        bank = Bank(setting, tau_h).load(a.bank)
        cfg = {"router": {dst: args[src] for src, dst in ROUTER_ARGS.items() if args.get(src) is not None}}
        model = setting.build_router(cfg, len(bank))
        model.load_state_dict(ck.get("model_state_dict", ck.get("student_state_dict", ck)))
        model = model.to(a.device).eval()
        print(f"[eval] router epoch {ck.get('epoch')}  M={len(bank)}  tau_h={tau_h:g}  k={k_mode}  v={v_mode}"
              + (f"  drop={a.drop}" if a.drop else ""), flush=True)
        return cls(setting, model, bank, k_mode, v_mode, a.drop)

    def repair_chunk(self, N: int) -> int:
        return _student_chunk(N)

    @torch.no_grad()
    def choose(self, ctx: Dict, current: torch.Tensor, mask: torch.Tensor, step: int) -> torch.Tensor:
        batch = sub_problem(ctx, current, mask)
        V_full, V_K, _ = self.bank.compute_probs(batch)
        y_hat = route(self.setting, self.model, batch, V_full, V_K, self.k_mode, self.v_mode, self.drop)
        return y_hat.masked_fill(~mask, float("-inf")).argmax(dim=-1)


# ---- baselines ----------------------------------------------------------------------------------------------------

class ScorerPolicy(StepPolicy):
    """NAM / MoET: the model scores the candidates of the sub-problem like a program (heuristic(unselected_locs,
    first_loc, current_loc) -> [B, K]) and enters as a one-program bank at temperature 1 with routing weight 1, so the
    policy is the model's softmax over the unvisited nodes (the router-bank engine, as for EPB)."""

    def __init__(self, setting: Setting, scorer: Callable):
        self.setting, self.scorer = setting, scorer

    @classmethod
    def load(cls, setting: Setting, method: str, a) -> "ScorerPolicy":
        """<pkg>/<method>_lehd.py, checkpoint --ckpt or <pkg>/lehd_tsp100/best.pt."""
        if not a.pkg:
            raise ValueError(f"--pkg (the {method} package directory) is required")
        mod = _import_from(a.pkg, f"{method}_lehd")
        ckpt = a.ckpt or os.path.join(a.pkg, _trained_size_dir(setting), "best.pt")
        _guard_fallbacks()
        if method == "nam":
            model, ck = mod.load_nam(ckpt, a.device)
            scorer = mod.NAMScorerTSP(model)
            print(f"[eval] nam epoch {ck.get('epoch')}  params {model.n_params()}", flush=True)
        else:
            model, ck = mod.load_moet(ckpt, a.device)
            scorer = mod.MoETScorerTSP(model)
            print(f"[eval] moet round {ck.get('round')}  K={model.K}", flush=True)
        return cls(setting, scorer)

    def repair_chunk(self, N: int) -> int:
        return _student_chunk(N)

    @torch.no_grad()
    def choose(self, ctx: Dict, current: torch.Tensor, mask: torch.Tensor, step: int) -> torch.Tensor:
        V_full, _, _ = self.setting.program_probs(self.scorer, sub_problem(ctx, current, mask), 1.0)
        return V_full.masked_fill(~mask, float("-inf")).argmax(dim=-1)


class TreeSurrogatePolicy(StepPolicy):
    """Decision-tree surrogate (tree package: features.lehd_tsp_features on the full graph with the unvisited nodes
    as candidates, gpu_tree.TreePolicy.score, masked argmax). Repairs run on the whole batch, and the RRC loop keeps
    the flipped tour when nothing is accepted, as the package's own loop."""
    keep_flip = True

    def __init__(self, pol, FT):
        self.pol, self.FT = pol, FT

    @classmethod
    def load(cls, setting: Setting, a) -> "TreeSurrogatePolicy":
        """<pkg>/trees/lehd_tsp100.pkl, --depth d4 | d6 | d8 | d12 (default) | staged."""
        import pickle
        if not a.pkg:
            raise ValueError("--pkg (the tree package directory) is required")
        FT = _import_from(a.pkg, "features")
        GT = _import_from(a.pkg, "gpu_tree")
        depth = getattr(a, "depth", None) or "d12"
        with open(os.path.join(a.pkg, "trees", f"{_trained_size_dir(setting)}.pkl"), "rb") as fh:
            pl = pickle.load(fh)
        if depth == "staged":
            pol = GT.TreePolicy("staged", [pl["trees"][f"staged{i}"] for i in range(3)], a.device,
                                stage_feat_idx=pl["stage_feature_index"], stage_thresholds=pl["stage_thresholds"])
        elif depth in pl["trees"]:
            pol = GT.TreePolicy(depth, [pl["trees"][depth]], a.device)
        else:
            raise ValueError(f"--depth {depth!r}: one of {sorted(pl['trees'])} or staged")
        print(f"[eval] tree {depth}", flush=True)
        return cls(pol, FT)

    def begin(self, locs, first, last) -> Dict:
        ctx = super().begin(locs, first, last)
        ctx["D"] = self.FT.pairwise_dist(locs)
        return ctx

    @torch.no_grad()
    def choose(self, ctx: Dict, current: torch.Tensor, mask: torch.Tensor, step: int) -> torch.Tensor:
        feats = self.FT.lehd_tsp_features(ctx["locs"], ctx["D"], current, ctx["first"], mask)
        return self.pol.score(feats).masked_fill(~mask, float("-inf")).argmax(dim=-1)


class FablePolicy(StepPolicy):
    """White-box agent policy of agents/fable51 (policies.tsp_policy.tsp_probs with the frozen parameters of
    eval/common_eval.load_frozen): needs the running mean of the visited coordinates and the tour's end point (the
    first node, or the fixed last node of a repaired segment). Repairs run on the whole batch."""

    def __init__(self, params, tsp_probs, pairwise_dist):
        self.p, self.tsp_probs, self.pairwise_dist = params, tsp_probs, pairwise_dist

    @classmethod
    def load(cls, a) -> "FablePolicy":
        root = os.path.abspath(a.pkg)
        TP = _import_from(root, "policies.tsp_policy")
        CE = _import_from(os.path.join(root, "eval"), "common_eval")
        params, _, _ = CE.load_frozen("tsp")
        return cls(params, TP.tsp_probs, TP.pairwise_dist)

    def begin(self, locs, first, last) -> Dict:
        ctx = super().begin(locs, first, last)
        ar = ctx["ar"]
        ctx["end"] = locs[ar, first] if last is None else locs[ar, last]
        ctx["D"] = self.pairwise_dist(locs)
        ctx["vis_sum"] = locs[ar, first].clone()
        return ctx

    @torch.no_grad()
    def choose(self, ctx: Dict, current: torch.Tensor, mask: torch.Tensor, step: int) -> torch.Tensor:
        locs = ctx["locs"]
        probs = self.tsp_probs(locs, mask, locs[ctx["ar"], current], ctx["end"], self.p, ctx["D"],
                               ctx["vis_sum"] / step)
        return probs.argmax(-1)

    def commit(self, ctx: Dict, node: torch.Tensor) -> None:
        ctx["vis_sum"] = ctx["vis_sum"] + ctx["locs"][ctx["ar"], node]


class GptPolicy(StepPolicy):
    """White-box agent policy of agents/gpt6 (whitebox_routing.tsp.lehd_action_distribution over the LEHD
    sub-problem, probabilities checked by whitebox_routing.evaluate._check_probs). Repairs in chunks of 64 rows."""

    def __init__(self, dist, check):
        self.dist, self.check = dist, check

    @classmethod
    def load(cls, a) -> "GptPolicy":
        TS = _import_from(a.pkg, "whitebox_routing.tsp")
        EV = _import_from(a.pkg, "whitebox_routing.evaluate")
        return cls(TS.lehd_action_distribution, EV._check_probs)

    def repair_chunk(self, N: int) -> int:
        return 64

    @torch.no_grad()
    def choose(self, ctx: Dict, current: torch.Tensor, mask: torch.Tensor, step: int) -> torch.Tensor:
        locs, ar = ctx["locs"], ctx["ar"]
        ids = mask.nonzero(as_tuple=False)[:, 1].reshape(locs.size(0), -1)
        cand = locs.gather(1, ids[..., None].expand(-1, -1, 2))
        probs = self.dist(cand, locs[ar, ctx["first"]], locs[ar, current])
        self.check(probs, torch.ones_like(ids, dtype=torch.bool))
        return ids.gather(1, probs.argmax(-1)[:, None]).squeeze(1)


def load_agent(a) -> StepPolicy:
    """--pkg agents/fable51 (policies/, eval/) or agents/gpt6 (whitebox_routing/)."""
    if not a.pkg:
        raise ValueError("--pkg (an agent package directory) is required")
    if os.path.isdir(os.path.join(a.pkg, "whitebox_routing")):
        return GptPolicy.load(a)
    if os.path.isfile(os.path.join(a.pkg, "policies", "tsp_policy.py")):
        return FablePolicy.load(a)
    raise ValueError(f"{a.pkg}: no agent policy (policies/tsp_policy.py or whitebox_routing/)")
