"""POMO teacher on the 2D Euclidean TSP (TSP-50, TSP-100).

Data: the single-file decision-step dataset of epb.teachers.pomo_tsp (teacher_probs, mask, locs, current_node,
first_node, visited, step_i [, teacher_logits, step_max]), memory-mapped with torch.load(mmap=True); a batch gathers
its rows by index (the split reads the coordinates of every row once, and the prefetch thread reads the whole file
into the page cache). The train/val split is by instance: an instance is a set of rows with identical `locs` (sha1 of
the coordinates), and for the same seed and val_ratio the split draws the same instances as the instance-grouped
split of the original trainer.

Programs: heuristic(locs [B,N,2], current_node [B], mask [B,N] bool, visited [B,N], step_norm [B], first_node [B])
-> logits [B,N]; V_m = masked_softmax(nan_to_num(logits / tau_h)) over the unvisited nodes. A program that raises
(or returns a wrong shape) gives zeros, i.e. a uniform V_m.

Router: the attention router of the paper (dynamic keys K_m = V_m^T h, w = softmax(q K^T / tau_r), y = sum_m w_m V_m),
with the parameter names of the released students; `codebook_keys` [M, D] is kept as a buffer so checkpoints stay
loadable by the paper's evaluators.
"""
import hashlib
import multiprocessing
import os
import subprocess
import sys
import threading
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset

from rl4co.models.nn.env_embeddings.context import TSPContext
from rl4co.models.nn.graph.attnnet import GraphAttentionNetwork

from . import Data, Setting, eval_option, register, state_convention

try:
    from scipy.spatial import ConvexHull  # type: ignore
    _HAS_SCIPY = True
except Exception:  # noqa: BLE001
    _HAS_SCIPY = False

GENERATOR = "pomo_tsp"              # the generator module, python -m epb.teachers.pomo_tsp

ROUTING_TEMPERATURE = 1.0       # tau_r of w = softmax(q K^T / tau_r) (all reported runs)
NORMALIZATION = "batch"         # encoder normalisation
FEEDFORWARD_HIDDEN = 512        # encoder feed-forward width
GEN_BATCH_SIZE = 256            # instances per teacher rollout batch of the generator
GEN_POMO_SIZE = 1               # rollouts per instance (start nodes 0 .. pomo_size-1)
CHECK_SIZES = (5, 10, 20, 50, 100)   # N of the synthetic CPU checks of a new program
CHECK_BATCH = (1, 3)                 # B of the synthetic CPU checks
CHECK_THREADS = 4
PREFETCH = True                 # read the data file once in a background thread so it sits in the page cache
PREFETCH_CHUNK = 16 << 20
# DataLoader workers are forked (they inherit the mapping); textgrad switches the default start method to spawn.
_MP_CONTEXT = "fork" if "fork" in multiprocessing.get_all_start_methods() else None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(cfg, path: str, default=None):
    """cfg value at a dotted path (attribute or dict access); default if absent or None."""
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        try:
            cur = cur[key] if isinstance(cur, dict) else getattr(cur, key)
        except (AttributeError, KeyError):
            return default
    return default if cur is None else cur


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask = mask.bool()
    has_valid = mask.any(dim=dim, keepdim=True)
    safe = logits.masked_fill(~mask, float("-inf"))
    safe = torch.where(has_valid, safe, torch.zeros_like(safe))
    probs = F.softmax(safe, dim=dim)
    probs = torch.where(mask, probs, torch.zeros_like(probs))
    probs = torch.where(has_valid, probs, torch.zeros_like(probs))
    return probs


def build_td(batch: Dict, device: torch.device) -> TensorDict:
    B = batch["locs"].size(0)
    N = batch["locs"].size(1)
    step_i = batch.get("step_i", torch.zeros(B, device=device)).long().view(B)
    first_node = (
        batch["first_node"].long().view(B)
        if "first_node" in batch
        else torch.zeros(B, dtype=torch.long, device=device)
    )
    mask = batch["mask"].bool() if "mask" in batch else torch.ones(B, N, dtype=torch.bool, device=device)
    visited = batch.get("visited", (~mask).float())
    return TensorDict(
        {
            "locs":         batch["locs"],
            "current_node": batch["current_node"].long().view(B),
            "first_node":   first_node,
            "i":            step_i,
            "action_mask":  mask,
            "visited":      visited,
        },
        batch_size=[B],
        device=device,
    )


def _sha1(row: torch.Tensor) -> str:
    return hashlib.sha1(row.contiguous().numpy().tobytes()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Decision-step data (mmap)
# ─────────────────────────────────────────────────────────────────────────────

class _Store:
    """The decision-step file, memory-mapped. rows(idx) gathers one batch in the format of the original trainer's
    dataset + default_collate (float32 / int64), plus `_row` (global row index) and `_inst` (instance id).

    Random rows of a file that is not in the page cache cost a disk seek each (a cold 1024-row batch of the TSP-100
    file on a spinning disk takes tens of seconds), so prefetch() reads the file once, sequentially, in a daemon
    thread: the pages land in the kernel's page cache (reclaimable, not part of the process heap) and later batches
    are served from memory."""

    def __init__(self, path: str):
        self.path = path
        self._t: Optional[Dict[str, torch.Tensor]] = None
        self.inst: Optional[torch.Tensor] = None
        t = self.tensors()
        self.num_rows = int(t["mask"].size(0))
        self.problem_size = int(t["mask"].size(1))
        self.step_max = float(t["step_i"].max().item()) if "step_i" in t else 1.0

    def tensors(self) -> Dict[str, torch.Tensor]:
        if self._t is None:
            data = torch.load(self.path, map_location="cpu", mmap=True, weights_only=True)
            if not (isinstance(data, dict) and {"mask", "locs", "current_node"}.issubset(data.keys())):
                raise ValueError(f"Unsupported dataset format in {self.path}")
            if "teacher_probs" not in data and "probs" not in data and "teacher_logits" not in data \
                    and "logits" not in data:
                raise ValueError(f"Expected teacher_probs or teacher_logits in {self.path}")
            alias = {"probs": "teacher_probs", "logits": "teacher_logits", "i": "step_i"}
            self._t = {alias.get(k, k): v for k, v in data.items()}
        return self._t

    def __getstate__(self):                 # a spawned worker re-opens the mapping instead of pickling it
        state = dict(self.__dict__)
        state["_t"] = None
        return state

    def prefetch(self) -> threading.Thread:
        def run():
            try:
                with open(self.path, "rb", buffering=0) as f:
                    if hasattr(os, "posix_fadvise"):
                        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
                    buf = memoryview(bytearray(PREFETCH_CHUNK))
                    while f.readinto(buf):
                        pass
            except OSError as e:
                warnings.warn(f"prefetch of {self.path} stopped: {e}")
        th = threading.Thread(target=run, name="epb-prefetch", daemon=True)
        th.start()
        return th

    def rows(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        t = self.tensors()
        idx = idx.long().view(-1)
        srt, order = idx.sort()
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel())

        def take(name):                      # sorted reads of the mapped file, then the requested order
            return t[name].index_select(0, srt)[inv]

        B = idx.numel()
        mask = take("mask").float()
        if "teacher_probs" in t:
            teacher_probs = take("teacher_probs").float()
        else:
            warnings.warn("No teacher_probs found; falling back to softmax of logits.", stacklevel=2)
            teacher_probs = masked_softmax(take("teacher_logits").float(), mask.bool())
        return {
            "teacher_probs": teacher_probs,
            "mask":          mask,
            "locs":          take("locs").float(),
            "current_node":  take("current_node").long().view(B),
            "first_node":    take("first_node").long().view(B) if "first_node" in t
                             else torch.zeros(B, dtype=torch.long),
            "visited":       take("visited").float() if "visited" in t else torch.zeros_like(mask),
            "step_i":        take("step_i").float().view(B) if "step_i" in t else torch.zeros(B),
            "step_max":      torch.full((B,), self.step_max, dtype=torch.float32),
            "_row":          idx.clone(),
            "_inst":         self.inst[idx] if self.inst is not None else torch.full((B,), -1, dtype=torch.long),
        }


def _group_rows(store: _Store) -> Tuple[torch.Tensor, List[str]]:
    """(instance id of every row [S], sorted sha1 keys of the instances). Rows with identical locs form one instance.

    Fast path: the generator writes, per rollout batch of L columns (instance x start node), steps 1..N-1 as blocks
    of L rows, so a column is rows o + t*L + j. The layout is read from step_i, checked on first_node (constant along
    a column) and on the locs of the first, middle and last step of every column; only one row per column is hashed.
    Any mismatch falls back to hashing every row (chunked)."""
    t = store.tensors()
    S = store.num_rows
    cols = _column_layout(t, S)
    if cols is None:
        warnings.warn(f"{store.path}: unrecognised row layout, hashing every row")
        locs = t["locs"]
        keys_row = []
        for lo in range(0, S, 65536):
            block = locs[lo:lo + 65536].float().reshape(-1, store.problem_size * 2)
            keys_row.extend(_sha1(r) for r in block)
        keys = sorted(set(keys_row))
        pos = {k: i for i, k in enumerate(keys)}
        return torch.tensor([pos[k] for k in keys_row], dtype=torch.long), keys
    col_of_row, reps = cols
    locs_rep = t["locs"].index_select(0, reps).float().reshape(reps.numel(), -1)
    col_keys = [_sha1(r) for r in locs_rep]
    keys = sorted(set(col_keys))
    pos = {k: i for i, k in enumerate(keys)}
    inst_of_col = torch.tensor([pos[k] for k in col_keys], dtype=torch.long)
    return inst_of_col[col_of_row], keys


def _column_layout(t: Dict[str, torch.Tensor], S: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(column of every row [S], representative (first-step) row of every column) or None if the file does not have
    the generator's batch-major layout."""
    if "step_i" not in t or "first_node" not in t:
        return None
    s = t["step_i"]
    if S == 0:
        return None
    change = torch.nonzero(s[1:] != s[:-1]).view(-1) + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.long), change])
    lens = torch.diff(torch.cat([starts, torch.tensor([S])]))
    vals = s[starts]
    T = int(vals.max().item() - vals.min().item()) + 1
    if T < 1 or starts.numel() % T != 0:
        return None
    nb = starts.numel() // T
    vals, lens, starts = vals.view(nb, T), lens.view(nb, T), starts.view(nb, T)
    steps = torch.arange(T, dtype=vals.dtype) + vals.min()
    if not bool((vals == steps).all()) or not bool((lens == lens[:, :1]).all()):
        return None
    col_of_row, reps = [], []
    n_cols = 0
    locs, first = t["locs"], t["first_node"]
    for b in range(nb):
        o, L = int(starts[b, 0]), int(lens[b, 0])
        fn = first[o:o + T * L].view(T, L)
        if not bool((fn == fn[:1]).all()):
            return None
        rep = torch.arange(o, o + L)
        ref = locs[o:o + L]
        for step in sorted({T // 2, T - 1}):
            if step > 0 and not torch.equal(locs[o + step * L:o + (step + 1) * L], ref):
                return None
        col_of_row.append((torch.arange(L) + n_cols).repeat(T))
        reps.append(rep)
        n_cols += L
    return torch.cat(col_of_row), torch.cat(reps)


def _split(inst: torch.Tensor, keys: List[str], val_ratio: float, seed: int) -> Tuple[torch.Tensor, int]:
    """Boolean val flag per instance, drawn exactly as the original trainer's instance-grouped split draws its val
    groups."""
    G = len(keys)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(G, generator=generator)
    val_count = min(G - 1, max(1, int(G * val_ratio)))
    is_val = torch.zeros(G, dtype=torch.bool)
    is_val[perm[:val_count]] = True
    return is_val, val_count


def _digest(keys: List[str], is_val: torch.Tensor) -> str:
    val_keys = [k for k, v in zip(keys, is_val.tolist()) if v]
    return hashlib.sha1("\n".join(val_keys).encode()).hexdigest()


class _Batches(Dataset):
    """Batch i = rows order[i*bs:(i+1)*bs] (two tensors, so a spawned worker unpickles it cheaply)."""

    def __init__(self, store: _Store, order: torch.Tensor, batch_size: int):
        self.store, self.order, self.batch_size = store, order, batch_size

    def __len__(self):
        return (self.order.numel() + self.batch_size - 1) // self.batch_size

    def __getitem__(self, i):
        return self.store.rows(self.order[i * self.batch_size:(i + 1) * self.batch_size])


class _Loader:
    """Batches of `rows` (int64 global row ids), in order or shuffled by a generator seeded with (seed, epoch)."""

    def __init__(self, store: _Store, rows: torch.Tensor, batch_size: int, shuffle: bool, seed: int,
                 num_workers: int = 0):
        self.store, self.rows, self.batch_size = store, rows.long().view(-1), int(batch_size)
        self.shuffle, self.seed, self.num_workers = shuffle, int(seed), int(num_workers)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return (self.rows.numel() + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rows = self.rows
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
            rows = rows[torch.randperm(rows.numel(), generator=g)]
        batches = _Batches(self.store, rows, self.batch_size)
        if self.num_workers <= 0 or len(batches) <= 1:
            for i in range(len(batches)):
                yield batches[i]
            return
        yield from DataLoader(batches, batch_size=None, shuffle=False, num_workers=self.num_workers,
                              multiprocessing_context=_MP_CONTEXT)


# ─────────────────────────────────────────────────────────────────────────────
# Router (Q, K)
# ─────────────────────────────────────────────────────────────────────────────

class HeuristicAttentionStudent(nn.Module):
    """
    Attention router over a bank of heuristic programs (dynamic routing).

      Q: TSPContext(h, td) + graph_context     [B, D]
      K: V.detach() @ h                        [B, M, D]   (dynamic key, no parameters)
      V: heuristic probs [B, M, N]             (external, stop gradient)
      w = softmax(Q·Kᵀ / τ_routing)            [B, M]
      ŷ = Σ_m w_m · V_m                        [B, N]

    `codebook_keys` [M, D] is the key table of the old fixed-routing ablation. It is unused here and kept as a
    persistent buffer so that checkpoints keep the released state_dict layout; set_num_programs resizes it with the
    bank, and a load_state_dict pre-hook resizes it to the incoming shape.
    """

    def __init__(
        self,
        num_heuristics: int,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = NORMALIZATION,
        feedforward_hidden: int = FEEDFORWARD_HIDDEN,
        routing_temperature: float = ROUTING_TEMPERATURE,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heuristics = num_heuristics
        self.routing_temperature = routing_temperature or (embed_dim ** 0.5)
        self.routing_mode = "dynamic"

        # Node features: [x, y, visited] → embed_dim
        self.init_embed = nn.Linear(3, embed_dim)
        self.encoder = GraphAttentionNetwork(
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_layers=num_encoder_layers,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
        )
        self.context_embedding = TSPContext(embed_dim=embed_dim)
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=False)
        # Same draw as the old parameter, so a router built from the same seed starts from the same weights.
        self.register_buffer("codebook_keys", torch.randn(num_heuristics, embed_dim) * 0.02)
        self._register_load_state_dict_pre_hook(self._resize_codebook_hook)

    def set_num_programs(self, num_programs: int, keep: Optional[List[int]] = None) -> None:
        """Resize codebook_keys to [num_programs, D]: rows `keep` (default: the first rows) are kept, new rows are
        drawn from a local generator (the global RNG is not touched)."""
        old = self.codebook_keys
        M = int(num_programs)
        if keep is None:
            keep = list(range(min(M, old.size(0))))
        kept = old[torch.as_tensor(keep, dtype=torch.long, device=old.device)]
        g = torch.Generator().manual_seed(M)
        new = (torch.randn(M - kept.size(0), old.size(1), generator=g) * 0.02).to(old)
        self.codebook_keys = torch.cat([kept, new], dim=0)
        self.num_heuristics = M

    def _resize_codebook_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        incoming = state_dict.get(prefix + "codebook_keys")
        if incoming is not None and tuple(incoming.shape) != tuple(self.codebook_keys.shape):
            self.codebook_keys = torch.zeros(incoming.shape, dtype=self.codebook_keys.dtype,
                                             device=self.codebook_keys.device)
            self.num_heuristics = int(incoming.shape[0])

    def encode(self, td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        locs = td["locs"]                          # [B, N, 2]
        visited = td["visited"].unsqueeze(-1)       # [B, N, 1]
        node_features = torch.cat([locs, visited], dim=-1)  # [B, N, 3]
        init_h = self.init_embed(node_features)
        h = self.encoder(init_h)
        graph_context = self.project_fixed_context(h.mean(dim=1))
        return h, graph_context

    def compute_query(self, h, graph_context, td) -> torch.Tensor:
        B = h.size(0)
        batch_idx    = torch.arange(B, device=h.device)
        first_node   = td["first_node"].long().view(B)
        current_node = td["current_node"].long().view(B)
        step_i       = td["i"].long().view(B)

        first_emb   = h[batch_idx, first_node]
        current_emb = h[batch_idx, current_node]
        normal_ctx  = self.context_embedding.project_context(
            torch.cat([first_emb, current_emb], dim=-1)
        )
        placeholder_ctx = self.context_embedding.project_context(
            self.context_embedding.W_placeholder.unsqueeze(0).expand(B, -1)
        )
        step_context = torch.where((step_i < 1).unsqueeze(-1), placeholder_ctx, normal_ctx)
        return step_context + graph_context

    def forward(self, td: TensorDict, V: torch.Tensor, return_scores: bool = False):
        h, graph_context = self.encode(td)
        q = self.compute_query(h, graph_context, td)
        # K_m = Σ_n V_m[n] · h[n]: V_m-weighted node embedding
        # = "where V_m proposes to go" in encoder embedding space
        K_dynamic = torch.bmm(V.detach(), h)                    # [B, M, D]
        scores = torch.bmm(
            K_dynamic, q.unsqueeze(-1)                          # [B, M, D]·[B, D, 1]
        ).squeeze(-1) / self.routing_temperature                 # [B, M]
        w = F.softmax(scores, dim=-1)                  # [B, M]
        y_hat = (w.unsqueeze(-1) * V).sum(dim=1)       # [B, N]
        if return_scores:
            return y_hat, w, scores
        return y_hat, w


# ─────────────────────────────────────────────────────────────────────────────
# Failure / fault case text
# ─────────────────────────────────────────────────────────────────────────────

def _topk_info(probs, locs, cur_pos, mask, k=2):
    probs_m = probs.masked_fill(~mask, float("-inf"))
    result = []
    for idx in probs_m.argsort(descending=True):
        if not mask[idx]:
            continue
        pos  = (round(float(locs[idx, 0]), 3), round(float(locs[idx, 1]), 3))
        dist = round(float(((locs[idx,0]-cur_pos[0])**2 + (locs[idx,1]-cur_pos[1])**2)**0.5), 3)
        result.append({"node": int(idx), "pos": pos, "dist_to_cur": dist})
        if len(result) >= k:
            break
    return result


def _nearest_unvisited(locs, mask, cur_pos, n=10):
    result = []
    for i in range(locs.size(0)):
        if not mask[i]:
            continue
        pos  = (round(float(locs[i,0]),3), round(float(locs[i,1]),3))
        dist = round(float(((locs[i,0]-cur_pos[0])**2+(locs[i,1]-cur_pos[1])**2)**0.5),3)
        result.append({"node": int(i), "pos": pos, "dist_to_cur": dist})
    result.sort(key=lambda x: x["dist_to_cur"])
    return result[:n]


def _hull_membership(coords: torch.Tensor) -> Optional[set]:
    """Indices of the nodes on the convex hull (None without scipy)."""
    if not _HAS_SCIPY:
        return None
    try:
        hull = ConvexHull(coords.detach().cpu().numpy())
        return set(int(i) for i in hull.vertices.tolist())
    except Exception:  # noqa: BLE001
        return None


def _step_norm(row: Dict) -> float:
    if "step_i" not in row:
        return 0.0
    return float(row["step_i"].float() / row["step_max"].float().clamp_min(1.0))


def _fault_features(row: Dict, V_bank: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> Dict:
    """The per-case features of the fault-cluster prompt (one fault: no program has the teacher's top-1)."""
    locs = row["locs"].float()
    mask = mask.bool()
    cur_idx = int(row["current_node"])
    first_idx = int(row["first_node"]) if "first_node" in row else 0
    frozen_top1 = [int(i) for i in V_bank.masked_fill(~mask.unsqueeze(0), float("-inf")).argmax(-1)] or [0]

    teacher_top1 = int(teacher.masked_fill(~mask, float("-inf")).argmax().item())
    cur_loc = locs[cur_idx]
    teacher_loc = locs[teacher_top1]
    first_loc = locs[first_idx]

    dists_to_cur = torch.norm(locs - cur_loc.unsqueeze(0), dim=-1)
    dists_to_cur_unv = dists_to_cur.masked_fill(~mask, float("inf"))
    nearest_d = float(dists_to_cur_unv.min().item())
    teacher_d = float(dists_to_cur[teacher_top1].item())
    ratio = teacher_d / max(nearest_d, 1e-9)

    forward_vec = cur_loc - first_loc
    target_vec = teacher_loc - cur_loc
    fnorm = float(torch.norm(forward_vec).item())
    tnorm = float(torch.norm(target_vec).item())
    if fnorm > 1e-6 and tnorm > 1e-6:
        forward_dot = float((forward_vec @ target_vec).item() / (fnorm * tnorm))
    else:
        forward_dot = 0.0

    hull = _hull_membership(locs)
    frozen_pick = frozen_top1[0]
    return {
        "step_norm": _step_norm(row),
        "n_unvisited": int(mask.sum().item()),
        "cur_pos": (round(float(cur_loc[0]), 3), round(float(cur_loc[1]), 3)),
        "teacher_top1_pos": (round(float(teacher_loc[0]), 3), round(float(teacher_loc[1]), 3)),
        "dist_cur_to_teacher": round(teacher_d, 3),
        "dist_cur_to_nearest_unv": round(nearest_d, 3),
        "farness_ratio": round(ratio, 2),
        "forward_dot": round(forward_dot, 2),
        "teacher_on_hull": (teacher_top1 in hull) if hull is not None else None,
        "dist_teacher_to_first": round(float(torch.norm(teacher_loc - first_loc).item()), 3),
        "frozen_consensus": len(set(frozen_top1)) == 1,
        "frozen_top1_pos": (round(float(locs[frozen_pick, 0]), 3), round(float(locs[frozen_pick, 1]), 3)),
    }


def _numeric_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    t = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(t.mean().item()),
        "p10": float(torch.quantile(t, 0.10).item()),
        "p25": float(torch.quantile(t, 0.25).item()),
        "p50": float(torch.quantile(t, 0.50).item()),
        "p75": float(torch.quantile(t, 0.75).item()),
        "p90": float(torch.quantile(t, 0.90).item()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Program checks
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_state(N: int, B: int, k: int, g: torch.Generator, grid: bool = False) -> tuple:
    """A decision step with k visited cities (k = 1: first decision, k = N-1: one unvisited city left)."""
    locs = torch.rand(B, N, 2, generator=g)
    if grid:                                   # coincident cities and tied distances
        locs = (locs * 4).round() / 4
    order = torch.argsort(torch.rand(B, N, generator=g), dim=1)
    visited = torch.zeros(B, N)
    visited.scatter_(1, order[:, :k], 1.0)
    mask = visited == 0
    first = order[:, 0].clone()
    cur = order[:, k - 1].clone()
    step_norm = torch.full((B,), k / max(N - 1, 1), dtype=torch.float32)
    return locs, cur, mask, visited, step_norm, first


def _check_call(fn, args: tuple, label: str, finite: bool = False) -> None:
    B, N = args[2].shape
    try:
        out = fn(*args)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"{label}: raised {type(e).__name__}: {e}")
    if not torch.is_tensor(out):
        raise ValueError(f"{label}: returned {type(out).__name__}, expected a tensor [B, N]")
    if tuple(out.shape) != (B, N):
        raise ValueError(f"{label}: output shape {tuple(out.shape)}, expected {(B, N)}")
    if finite and not bool(torch.isfinite(out).all()):
        raise ValueError(f"{label}: non-finite output")


# ─────────────────────────────────────────────────────────────────────────────
# Setting
# ─────────────────────────────────────────────────────────────────────────────

@register
class PomoTSP(Setting):
    name = "pomo_tsp"
    problem = "tsp"
    # The first stage of the released TSP-100 student: from-scratch M=15 in [10, 30], bs 1024,
    # lr 3e-4 cosine, 4 encoder layers, tau_h 0.1, TextGrad from epoch 8 every 5 epochs.
    defaults = {
        "data": {"val_ratio": 0.1, "num_workers": 4},
        "generate": {"num_instances": 100000, "seed": 42, "problem_size": None, "pomo_size": 1, "batch_size": 256},
        "bank": {"num_programs": 15, "min_programs": 10, "max_programs": 30, "tau_h": 0.1},
        "router": {"embed_dim": 128, "num_layers": 4, "num_heads": 8},
        "train": {"epochs": 80, "batch_size": 1024, "lr": 3e-4, "scheduler": "cosine", "amp": False},
        "textgrad": {"start_epoch": 8, "every_n_epochs": 5, "backward_every_n": 400, "buffer_size": 10,
                     "rounds": 4, "min_improvement": 0.005, "eval_batches": 20},
        "drop": {"grace_epochs": 15},
        "add": {"n_candidates": 6, "min_gain": 0.001, "pretrain_epochs": 5, "finetune_epochs": 5,
                "cooldown_epochs": 9},
    }

    def __init__(self):
        self._problem_size: Optional[int] = None

    # ---- data ---------------------------------------------------------------------------------------------------
    def build_data(self, cfg, split: Optional[Dict] = None) -> Data:
        path = _cfg(cfg, "data.path")
        if not path:
            raise ValueError("data.path is not set")
        store = _Store(path)
        if PREFETCH:
            store.prefetch()
        self._problem_size = store.problem_size
        if split:
            if split.get("method") != "instance-sha1":
                raise ValueError(f"split method {split.get('method')!r} is not this setting's")
            seed, val_ratio = int(split["seed"]), float(split["val_ratio"])
        else:
            seed, val_ratio = int(_cfg(cfg, "run.seed", 42)), float(_cfg(cfg, "data.val_ratio", 0.1))
        inst, keys = _group_rows(store)
        is_val_inst, val_count = _split(inst, keys, val_ratio, seed)
        digest = _digest(keys, is_val_inst)
        if split and (int(split["num_rows"]) != store.num_rows or int(split["num_instances"]) != len(keys)
                      or split["val_digest"] != digest):
            raise ValueError(f"{path} does not match the saved split (rows {store.num_rows} vs {split['num_rows']}, "
                             f"instances {len(keys)} vs {split['num_instances']}, val digest)")
        store.inst = inst
        is_val = is_val_inst[inst]
        train_rows = torch.nonzero(~is_val).view(-1)
        val_rows = torch.nonzero(is_val).view(-1)
        if train_rows.numel() == 0 or val_rows.numel() == 0:
            raise ValueError(f"Invalid split: train={train_rows.numel()} val={val_rows.numel()}")
        is_train = ~is_val
        bs = int(_cfg(cfg, "train.batch_size", 1024))
        workers = int(_cfg(cfg, "data.num_workers", 0))
        train = _Loader(store, train_rows, bs, True, seed, workers)
        val = _Loader(store, val_rows, bs, False, seed, workers)

        def scan(scan_seed: int):
            return _Loader(store, train_rows, bs, True, scan_seed, workers)

        def subset(row_ids: torch.Tensor, batch_size: int, shuffle: bool = False, sub_seed: int = 0):
            row_ids = torch.as_tensor(row_ids, dtype=torch.long).view(-1)
            if row_ids.numel() and not bool(is_train[row_ids].all()):
                raise ValueError("Data.subset: rows outside the train split")
            return _Loader(store, row_ids, batch_size, shuffle, sub_seed, 0)

        print(f"Split: train={train_rows.numel()}  val={val_rows.numel()}  "
              f"(instances {len(keys) - val_count} / {val_count}, N={store.problem_size})")
        return Data(
            train=train, val=val, problem_size=store.problem_size, scan=scan, subset=subset,
            set_epoch=train.set_epoch,
            split={"method": "instance-sha1", "seed": seed, "val_ratio": val_ratio, "num_rows": store.num_rows,
                   "num_instances": len(keys), "num_val_instances": val_count, "val_digest": digest},
        )

    def row_ids(self, batch: Dict) -> torch.Tensor:
        return batch["_row"].long()

    def instance_ids(self, batch: Dict) -> torch.Tensor:
        return batch["_inst"].long()

    def target(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        return batch["teacher_probs"], batch["mask"].bool()

    # ---- programs -----------------------------------------------------------------------------------------------
    def program_args(self, batch: Dict) -> tuple:
        mask = batch["mask"].bool()
        B, N = mask.shape
        device = mask.device
        locs = batch["locs"]
        cur = batch["current_node"].long().view(B)
        vis = batch["visited"] if "visited" in batch else torch.zeros(B, N, device=device)
        step = (
            batch["step_i"].float() / batch["step_max"].float().clamp_min(1.0)
            if "step_i" in batch else torch.zeros(B, device=device)
        )
        first = (
            batch["first_node"].long().view(B)
            if "first_node" in batch else torch.zeros(B, dtype=torch.long, device=device)
        )
        return locs, cur, mask, vis, step, first

    def program_probs(self, fn: Callable, batch: Dict, tau_h: float
                      ) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        """V_m = masked_softmax(nan_to_num(logits / tau_h)); a program that raises or returns a wrong shape gives
        zeros (uniform V_m) and ok=False. Non-finite logits are sanitised element-wise (NaN = least preferred), which
        is the identity on finite output and keeps ok=True."""
        args = self.program_args(batch)
        mask = args[2]
        B, N = mask.shape
        device = mask.device
        ok = True
        try:
            logits = fn(*args)
            logits = logits.float()
            if logits.device != device:
                logits = logits.to(device)
            if tuple(logits.shape) != (B, N):
                raise ValueError(f"output shape {tuple(logits.shape)}, expected {(B, N)}")
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Heuristic runtime error: {e}. Using zeros.")
            logits = torch.zeros(B, N, device=device)
            ok = False
        logits_sharp = logits / tau_h
        logits_sharp = torch.nan_to_num(logits_sharp, nan=-1e9, posinf=1e9, neginf=-1e9)
        probs = masked_softmax(logits_sharp.masked_fill(~mask, float("-inf")), mask)
        return probs, None, ok

    def check_program(self, fn: Callable) -> None:
        """The dummy call of the original trainer's program check (B=2, N=5, all cities unvisited: right shape,
        finite), then synthetic decision steps on the CPU for every N in CHECK_SIZES and B in CHECK_BATCH: the first
        decision, a mid-tour step, the last decision (a single unvisited city), and a mid-tour step with coincident
        cities."""
        n_threads = torch.get_num_threads()
        torch.set_num_threads(min(CHECK_THREADS, n_threads))
        try:
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                g = torch.Generator().manual_seed(0)
                dummy = (torch.rand(2, 5, 2, generator=g), torch.zeros(2, dtype=torch.long),
                         torch.ones(2, 5, dtype=torch.bool), torch.zeros(2, 5), torch.zeros(2),
                         torch.zeros(2, dtype=torch.long))
                _check_call(fn, dummy, "dummy batch (B=2, N=5, step 0)", finite=True)
                for N in CHECK_SIZES:
                    for B in CHECK_BATCH:
                        for case, k, grid in (("first step", 1, False), ("mid step", max(2, N // 2), False),
                                              ("last step", N - 1, False), ("coincident cities", max(2, N // 2), True)):
                            _check_call(fn, _synthetic_state(N, B, k, g, grid), f"{case} (B={B}, N={N})")
        finally:
            torch.set_num_threads(n_threads)

    # ---- router -------------------------------------------------------------------------------------------------
    def build_router(self, cfg, num_programs: int) -> nn.Module:
        return HeuristicAttentionStudent(
            num_heuristics=num_programs,
            embed_dim=int(_cfg(cfg, "router.embed_dim", 128)),
            num_encoder_layers=int(_cfg(cfg, "router.num_layers", 4)),
            num_heads=int(_cfg(cfg, "router.num_heads", 8)),
        )

    def router_forward(self, model: nn.Module, batch: Dict, V_full: torch.Tensor, V_K: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if model.codebook_keys.size(0) != V_full.size(1):
            model.set_num_programs(V_full.size(1))
        td = build_td(batch, V_full.device)
        return model(td, V_full, return_scores=True)

    def ckpt_args(self, cfg, num_programs: int) -> Dict:
        args = {
            "num_heuristics": int(num_programs),
            "heuristic_temperature": float(_cfg(cfg, "bank.tau_h", 0.1)),
            "embed_dim": int(_cfg(cfg, "router.embed_dim", 128)),
            "num_encoder_layers": int(_cfg(cfg, "router.num_layers", 4)),
            "num_heads": int(_cfg(cfg, "router.num_heads", 8)),
            "routing_temperature": ROUTING_TEMPERATURE,
            "routing_mode": "dynamic",
        }
        if self._problem_size is not None:
            args["problem_size"] = self._problem_size
        return args

    # ---- prompts ------------------------------------------------------------------------------------------------
    def prompt_vars(self, cfg, problem_size: int) -> Dict[str, str]:
        return {
            "teacher": str(_cfg(cfg, "prompts.teacher", "POMO")),
            "problem_size": str(int(problem_size)),
            "tau_h": f"{float(_cfg(cfg, 'bank.tau_h', 0.1)):g}",
        }

    def failure_case(self, row: Dict, V_m: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor, kind: str) -> str:
        """The failure-case lines of the old backward prompts (code: step, unvisited count, positions, 5 nearest
        unvisited, top-2 of program and teacher; desc: step, position, top-2s), indented as they were inside the old
        prompt."""
        locs = row["locs"].float()
        mask = mask.bool()
        cur_idx = int(row["current_node"])
        first_idx = int(row["first_node"]) if "first_node" in row else 0
        step = float(row["step_i"]) if "step_i" in row else 0.0
        step_max = max(float(row["step_max"]) if "step_max" in row else 1.0, 1)
        cur_pos = (round(float(locs[cur_idx, 0]), 3), round(float(locs[cur_idx, 1]), 3))
        h3 = [f"node{x['node']} pos={x['pos']} d={x['dist_to_cur']}"
              for x in _topk_info(V_m, locs, cur_pos, mask)]
        t3 = [f"node{x['node']} pos={x['pos']} d={x['dist_to_cur']}"
              for x in _topk_info(teacher, locs, cur_pos, mask)]
        step_line = f"      Step: {int(step)}/{int(step_max)} ({100*int(step)//int(step_max)}% complete)"
        if kind == "desc":
            return "\n".join([
                step_line,
                f"      Current pos: {cur_pos}",
                f"      Heuristic top-2: {h3}",
                f"      Teacher   top-2: {t3}",
            ])
        if kind != "code":
            raise ValueError(f"failure_case kind {kind!r}")
        first_pos = (round(float(locs[first_idx, 0]), 3), round(float(locs[first_idx, 1]), 3))
        nu = [f"node{x['node']} d={x['dist_to_cur']}" for x in _nearest_unvisited(locs, mask, cur_pos)[:5]]
        return "\n".join([
            step_line,
            f"      Unvisited nodes remaining: {int(mask.sum().item())}",
            f"      Current position: {cur_pos}",
            f"      Tour-start position: {first_pos}",
            f"      5 nearest unvisited: {nu}",
            f"      Heuristic top-2: {h3}",
            f"      Teacher   top-2: {t3}",
        ])

    def fault_case(self, row: Dict, V_bank: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> str:
        """One line of compact features of the fault-cluster prompt; `frozen_pick` is the argmax of program 0 of the
        current bank and `consensus` means every program of the current bank picked the same city (the prompt keeps
        the original trainer's labels)."""
        f = _fault_features(row, V_bank, teacher, mask)
        hull_str = "on_hull" if f["teacher_on_hull"] is True else ("interior" if f["teacher_on_hull"] is False else "?")
        cons_str = "consensus" if f["frozen_consensus"] else "split"
        return (
            f"  step={f['step_norm']:.2f} n_unv={f['n_unvisited']:>2d} "
            f"cur={f['cur_pos']} teacher={f['teacher_top1_pos']} "
            f"d_t={f['dist_cur_to_teacher']} d_near={f['dist_cur_to_nearest_unv']} "
            f"ratio={f['farness_ratio']} fwd_dot={f['forward_dot']:+.2f} "
            f"teacher={hull_str} d_to_start={f['dist_teacher_to_first']} "
            f"frozen_pick={f['frozen_top1_pos']} ({cons_str})"
        )

    def fault_summary(self, rows: List[Dict], V_banks: List[torch.Tensor], teachers: List[torch.Tensor],
                      masks: List[torch.Tensor]) -> str:
        """The aggregate-statistics block of the fault-cluster prompt over the given faults ("frozen heuristics" in its
        text are the programs of the current bank)."""
        feats = [_fault_features(r, v, t, m) for r, v, t, m in zip(rows, V_banks, teachers, masks)]
        stats: Dict[str, Dict] = {}
        for fld in ("step_norm", "n_unvisited", "dist_cur_to_teacher", "dist_cur_to_nearest_unv",
                    "farness_ratio", "forward_dot", "dist_teacher_to_first"):
            stats[fld] = _numeric_summary([float(f[fld]) for f in feats])
        hull_known = [f for f in feats if f["teacher_on_hull"] is not None]
        hull_frac = sum(1 for f in hull_known if f["teacher_on_hull"]) / max(len(hull_known), 1)
        cons_frac = sum(1 for f in feats if f["frozen_consensus"]) / max(len(feats), 1)
        lines = [
            f"=== Aggregate stats over {len(feats)} portfolio-fault cases ===",
            "(use these to decide which axes are most discriminative for bucketing)",
            "",
        ]
        for key, label in (("step_norm", "step_norm"), ("n_unvisited", "n_unvisited"),
                           ("farness_ratio", "ratio (d_t/d_near)"), ("forward_dot", "fwd_dot"),
                           ("dist_cur_to_teacher", "d_cur_to_teacher"),
                           ("dist_cur_to_nearest_unv", "d_cur_to_nearest_unv"),
                           ("dist_teacher_to_first", "d_teacher_to_first")):
            s = stats[key]
            lines.append(
                f"  {label:<23s}  mean={s['mean']:>6.2f}  "
                f"p10={s['p10']:>6.2f}  p25={s['p25']:>6.2f}  "
                f"p50={s['p50']:>6.2f}  p75={s['p75']:>6.2f}  "
                f"p90={s['p90']:>6.2f}"
            )
        lines.append("")
        lines.append(f"  teacher_on_hull True fraction: {hull_frac:.1%}  (known for {len(hull_known)})")
        lines.append(f"  frozen_consensus True fraction: {cons_frac:.1%}  "
                     f"(all frozen heuristics agreed on the same wrong pick)")
        return "\n".join(lines)

    # ---- data generation ----------------------------------------------------------------------------------------
    def generate(self, cfg) -> None:
        """Run the generator python -m epb.teachers.pomo_tsp (greedy POMO rollouts, every decision step) in a process
        of its own and check the written file. generate.problem_size (default 100), generate.pomo_size / batch_size (default
        GEN_POMO_SIZE / GEN_BATCH_SIZE) are read when the config has them."""
        ckpt = _cfg(cfg, "generate.teacher_ckpt")
        if not ckpt or str(ckpt).startswith("<"):
            raise ValueError("generate.teacher_ckpt is not set (a POMO-TSP checkpoint in the yd-kwon/POMO format)")
        out = _cfg(cfg, "data.path")
        if not out:
            raise ValueError("data.path is not set")
        N = int(_cfg(cfg, "generate.problem_size", 100))
        n_inst = int(_cfg(cfg, "generate.num_instances", 100000))
        pomo = int(_cfg(cfg, "generate.pomo_size", GEN_POMO_SIZE))
        from ..teachers import command
        cmd, env = command(GENERATOR, [
            "--checkpoint", ckpt, "--problem-size", N, "--num-instances", n_inst,
            "--batch-size", int(_cfg(cfg, "generate.batch_size", GEN_BATCH_SIZE)),
            "--pomo-size", pomo, "--seed", int(_cfg(cfg, "generate.seed", 42)),
            "--output", os.path.abspath(out)])
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=env)
        self._check_file(out, N, n_inst * pomo * (N - 1))

    @staticmethod
    def _check_file(path: str, N: int, n_rows: int) -> None:
        t = _Store(path).tensors()
        S = int(t["mask"].size(0))
        shapes = {"teacher_probs": (S, N), "mask": (S, N), "locs": (S, N, 2), "current_node": (S,),
                  "first_node": (S,), "visited": (S, N), "step_i": (S,)}
        for k, shape in shapes.items():
            if k not in t or tuple(t[k].shape) != shape:
                raise ValueError(f"{path}: {k} has shape {tuple(t[k].shape) if k in t else None}, expected {shape}")
        if S != n_rows:
            raise ValueError(f"{path}: {S} rows, expected {n_rows}")
        s = t["step_i"]
        if float(s.min()) != 1.0 or float(s.max()) != float(N - 1):
            raise ValueError(f"{path}: step_i spans [{float(s.min())}, {float(s.max())}], expected [1, {N - 1}]")
        probe = torch.linspace(0, S - 1, min(S, 4096)).long()
        m = t["mask"].index_select(0, probe)
        if not torch.equal(m.sum(-1), float(N) - s.index_select(0, probe)):
            raise ValueError(f"{path}: feasible-node counts do not match step_i")
        if not torch.allclose(t["teacher_probs"].index_select(0, probe).sum(-1), torch.ones(probe.numel()),
                              atol=1e-4):
            raise ValueError(f"{path}: teacher_probs rows do not sum to 1")
        print(f"{path}: {S} decision steps, N={N}: format OK")

    # ---- evaluation (epb.eval) ----------------------------------------------------------------------------------
    search_default = "ms"

    def eval_args(self, parser) -> None:
        parser.add_argument("--enc-layers", type=int, help="router encoder depth if the checkpoint does not record it")
        parser.add_argument("--ms-chunk", type=int, help="instances per multi-start batch (default: the method's)")
        parser.add_argument("--starts-per-chunk", type=int, help="start nodes per multi-start sub-batch (default: the "
                                                                 "method's)")

    def eval_policy(self, method: str, a):
        """teacher (--teacher-ckpt), epb / expert / frozen (--ckpt, --bank), nam / moet / tree / agent (--pkg, the
        package directory under epb/baselines; the model of the instances' N is loaded at the first batch)."""
        device = torch.device(a.device)
        if method == "teacher":
            if not a.teacher_ckpt:
                raise ValueError("the teacher needs --teacher-ckpt (a POMO-TSP checkpoint in the yd-kwon/POMO format)")
            return PomoTeacher(a.teacher_ckpt, device)
        if method in ("epb", "expert", "frozen"):
            return self._router_policy(a, device)
        if not a.pkg or not os.path.isdir(a.pkg):
            raise ValueError(f"{method} needs --pkg, its package directory under epb/baselines")
        if method in ("nam", "moet"):
            return AdditivePolicy(method, a.pkg, device, state_convention(a))
        if method == "tree":
            return TreePolicy(a.pkg, a.depth or "d12", device)
        if method == "agent":
            if os.path.exists(os.path.join(a.pkg, "policies", "tsp_policy.py")):
                return FablePolicy(a.pkg, device)
            if os.path.exists(os.path.join(a.pkg, "whitebox_routing", "tsp.py")):
                return GptPolicy(a.pkg, device)
            raise ValueError(f"{a.pkg}: not an agent package (policies/tsp_policy.py or whitebox_routing/tsp.py)")
        raise ValueError(f"unknown method {method!r}")

    def _router_policy(self, a, device) -> "RouterPolicy":
        """--ckpt (best_model.pt) + --bank with the router of this setting and epb.bank.Bank. tau_h: flag / decode.json,
        else the checkpoint's heuristic_temperature, else the setting's bank.tau_h; k_mode / v_mode default to soft
        (the training forward); the state convention follows settings.state_convention."""
        from ..bank import Bank
        if not a.ckpt or not a.bank:
            raise ValueError("router methods need --ckpt and --bank (or --ckpt DIR with bank.json)")
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        ca = ck.get("args") or {}
        ca = ca if isinstance(ca, dict) else vars(ca)
        tau_h = eval_option(a, "tau_h", ca, "heuristic_temperature", self.defaults["bank"]["tau_h"])
        # a program that runs out of GPU memory would be replaced by a uniform one: abort instead
        warnings.filterwarnings("error", message=".*ran out of GPU memory.*")
        bank = Bank(self, float(tau_h)).load(a.bank)
        sd = ck["model_state_dict"]
        layers = ca.get("num_encoder_layers") or a.enc_layers or \
            1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers."))
        cfg = {"router": {"embed_dim": ca.get("embed_dim", 128), "num_layers": layers,
                          "num_heads": ca.get("num_heads", 8)}}
        model = self.build_router(cfg, len(bank))
        model.load_state_dict(sd)
        keep = torch.ones(len(bank), dtype=torch.bool)
        keep[list(a.drop or [])] = False
        return RouterPolicy(model.to(device).eval(), bank, a.k_mode or "soft", a.v_mode or "soft", keep, device,
                            state_convention(a, ca))

    @staticmethod
    def eval_layout(policy, a, N: int, flags: bool = True) -> Tuple[int, int, int]:
        """(instances per batch, instances per multi-start batch, starts per sub-batch): the flags, else the policy's."""
        b, c, s = policy.layout(N)
        if not flags:
            return b, c, s
        return int(a.batch or b), int(a.ms_chunk or c), int(a.starts_per_chunk or s)

    def eval_greedy(self, policy, inst: Dict[str, torch.Tensor], a) -> torch.Tensor:
        """Start node 0."""
        locs = inst["locs"]
        bs = self.eval_layout(policy, a, locs.size(1))[0]
        out = []
        for s in range(0, locs.size(0), bs):
            x = locs[s:s + bs].to(policy.device)
            tours, _ = construct(policy, x, torch.zeros(x.size(0), 1, dtype=torch.long, device=x.device))
            out.append(tour_costs(x, tours)[:, 0])
        return torch.cat(out)

    def eval_search(self, policy, inst: Dict[str, torch.Tensor], a) -> torch.Tensor:
        """ms: the best of the N start nodes, in multi-start batches of instances whose starts run in sub-batches."""
        if a.search != "ms":
            raise ValueError(f"pomo_tsp has no search {a.search!r} (ms only)")
        locs = inst["locs"]
        n, N, _ = locs.shape
        _, chunk, S = self.eval_layout(policy, a, N)
        out = []
        for s in range(0, n, chunk):
            x = locs[s:s + chunk].to(policy.device)
            best = torch.full((x.size(0),), float("inf"), device=x.device)
            for s0 in range(0, N, S):
                first = torch.arange(s0, min(s0 + S, N), device=x.device).expand(x.size(0), -1).contiguous()
                tours, _ = construct(policy, x, first)
                best = torch.minimum(best, tour_costs(x, tours).min(1).values)
            out.append(best)
        return torch.cat(out)

    def eval_top1(self, teacher, policy, inst: Dict[str, torch.Tensor], a) -> Dict[str, torch.Tensor]:
        """Along the teacher's greedy tour from node 0 (each in its own greedy batches). Primary states: steps
        2 .. N-2 (step 1 and the forced last step dropped); all states: steps 1 .. N-1."""
        locs = inst["locs"]
        n, N, _ = locs.shape
        dev = policy.device
        zeros = lambda b: torch.zeros(b, 1, dtype=torch.long, device=dev)   # noqa: E731
        tb = self.eval_layout(teacher, a, N, flags=teacher is policy)[0]
        T = torch.cat([construct(teacher, locs[s:s + tb].to(dev), zeros(locs[s:s + tb].size(0)))[0][:, 0].cpu()
                       for s in range(0, n, tb)])
        step = torch.arange(1, N)
        prim = (step >= 2) & (N - step > 1)
        bs = self.eval_layout(policy, a, N)[0]
        hits, hits_all = [], []
        for s in range(0, n, bs):
            x, f = locs[s:s + bs].to(dev), T[s:s + bs].to(dev)
            _, picks = construct(policy, x, zeros(x.size(0)), forced=f.unsqueeze(1))
            hit = (picks[:, 0] == f[:, 1:]).cpu()
            hits.append((hit & prim).sum(1))
            hits_all.append(hit.sum(1))
        return {"hits": torch.cat(hits).double(), "states": torch.full((n,), float(prim.sum()), dtype=torch.double),
                "hits_all": torch.cat(hits_all).double(),
                "states_all": torch.full((n,), float(N - 1), dtype=torch.double)}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (epb.eval)
# ─────────────────────────────────────────────────────────────────────────────
#
# Every method is a Stepper and runs through construct(). Float32 results depend on the batch layout, so each
# method's layout() is the one of the main table (router methods: decode.json).

POMO_PARAMS = dict(embedding_dim=128, sqrt_embedding_dim=128 ** 0.5, encoder_layer_num=6, qkv_dim=16, head_num=8,
                   logit_clipping=10, ff_hidden_dim=512, eval_type="argmax")
EVAL_BATCH = {50: 500, 100: 512}    # instances per batch by N (otherwise 500)
MS_CHUNK = {50: 20, 100: 8}         # instances per multi-start batch of the router-family methods by N
_MODULES: Dict[str, object] = {}


def _module(path: str):
    """A package file loaded by path under a private name (cached; sys.path is restored afterwards)."""
    import importlib.util
    path = os.path.abspath(path)
    if path not in _MODULES:
        name = "epb_eval_" + hashlib.sha1(path.encode()).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        saved = list(sys.path)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path[:] = saved
        _MODULES[path] = mod
    return _MODULES[path]


def _pomo():
    """The POMO model code, epb.teachers.pomo_tsp (TSPModel, _get_encoding)."""
    from ..teachers import pomo_tsp
    return pomo_tsp


def _model_file(pkg: str, N: int, name: str) -> str:
    path = os.path.join(pkg, f"pomo_tsp{N}", name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: no model of {pkg} for N={N}")
    return path


def tour_length(locs: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    """Closed-tour length [R]: locs [R, N, 2], tour [R, N]."""
    gathered = torch.gather(locs, 1, tour.unsqueeze(-1).expand(-1, -1, 2))
    shifted = torch.cat([gathered[:, 1:], gathered[:, :1]], dim=1)
    return torch.norm(gathered - shifted, dim=-1).sum(-1)


def repeat_rows(x: torch.Tensor, S: int) -> torch.Tensor:
    """[B, ...] -> [B * S, ...], every instance repeated for its S start nodes."""
    return x.unsqueeze(1).expand(x.size(0), S, *x.shape[1:]).reshape(x.size(0) * S, *x.shape[1:])


def tour_costs(locs: torch.Tensor, tours: torch.Tensor) -> torch.Tensor:
    """[B, S] lengths of tours [B, S, N] on locs [B, N, 2]."""
    B, S, N = tours.shape
    return tour_length(repeat_rows(locs, S), tours.reshape(B * S, N)).view(B, S)


@torch.no_grad()
def construct(policy, locs: torch.Tensor, first: torch.Tensor, forced: Optional[torch.Tensor] = None
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tours from the start nodes first [B, S]: greedy, or along forced [B, S, N] (teacher forcing).
    Returns (tours [B, S, N], the policy's picks at steps 1 .. N-1 [B, S, N-1])."""
    B, N, _ = locs.shape
    visited = torch.zeros(B, first.size(1), N, dtype=torch.bool, device=locs.device)
    visited.scatter_(2, first.unsqueeze(-1), True)
    policy.begin(locs, first)
    cur, tour, picks = first, [first], []
    for step in range(1, N):
        pick = policy.choose(visited, cur, step)
        cur = pick if forced is None else forced[:, :, step]
        visited.scatter_(2, cur.unsqueeze(-1), True)
        policy.advance(cur)
        tour.append(cur)
        picks.append(pick)
    return torch.stack(tour, 2), torch.stack(picks, 2)


def decision_state(locs, vis, cur, first, step: int, convention: str = "legacy") -> Dict[str, torch.Tensor]:
    """The decision state at decision t = `step` (1 .. N-1) as the bank programs and the router see it: rows of mask,
    visited, current / first node, step_i and step_max = N - 1. convention "train": step_i = t, the rows of
    the generator epb.teachers.pomo_tsp (t nodes visited); "legacy": step_i = t - 1, the evaluator of the
    released results (the router's query then uses W_placeholder at t = 1, a state training never contains, and the
    programs see step_norm = (t - 1) / (N - 1))."""
    if convention not in ("legacy", "train"):
        raise ValueError(f"convention must be legacy or train, got {convention!r}")
    R, N = vis.shape
    step_i = step if convention == "train" else step - 1
    return {"locs": locs, "mask": ~vis, "current_node": cur, "first_node": first, "visited": vis.float(),
            "step_i": torch.full((R,), float(step_i), device=locs.device),
            "step_max": torch.full((R,), float(N - 1), device=locs.device)}


class Stepper:
    """A policy for construct(): begin(locs [B, N, 2], first [B, S]) once per batch (S start nodes per instance, the
    POMO layout), choose(visited [B, S, N] bool, current [B, S], step) -> next node [B, S] at steps 1 .. N-1, and
    advance(node [B, S]) after every move. layout(N): (instances per batch, instances per multi-start batch, starts
    per sub-batch) of the main table."""
    device: torch.device

    def begin(self, locs, first):
        pass

    def advance(self, node):
        pass

    def layout(self, N: int) -> Tuple[int, int, int]:
        b = EVAL_BATCH.get(N, 500)
        return b, MS_CHUNK.get(N, max(1, b // N)), N


class PomoTeacher(Stepper):
    """The POMO teacher (yd-kwon/POMO TSPModel), argmax decoding; one encoder pass per batch."""

    def __init__(self, ckpt: str, device: torch.device):
        model = _pomo().TSPModel(**POMO_PARAMS).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
        self.model, self.device = model.eval(), device

    def layout(self, N):
        b = EVAL_BATCH.get(N, 500)
        return b, max(1, b // N), N

    def begin(self, locs, first):
        self.model.pre_forward(locs)
        self.model.decoder.set_q1(_pomo()._get_encoding(self.model.encoded_nodes, first))

    def decode(self, visited, cur) -> Tuple[torch.Tensor, torch.Tensor]:
        """(probabilities, masked logits) [B, S, N] of the next node at current nodes cur [B, S] (after begin())."""
        ninf_mask = torch.zeros(visited.shape, device=self.device).masked_fill_(visited, float("-inf"))
        return self.model.decoder(_pomo()._get_encoding(self.model.encoded_nodes, cur), ninf_mask=ninf_mask)

    def choose(self, visited, cur, step):
        return self.decode(visited, cur)[0].argmax(dim=2)


class _RowStepper(Stepper):
    """A stepper that sees the B * S rollouts as rows: pick(visited [R, N], current [R], step) -> [R]."""

    def begin(self, locs, first):
        self.locs, self.first = repeat_rows(locs, first.size(1)), first.reshape(-1)
        self.ar = torch.arange(self.first.numel(), device=locs.device)

    def choose(self, visited, cur, step):
        B, S, N = visited.shape
        return self.pick(visited.reshape(B * S, N), cur.reshape(B * S), step).view(B, S)

    def pick(self, vis, cur, step):
        raise NotImplementedError


class RouterPolicy(_RowStepper):
    """Router + bank with the decoding options of the paper. k_mode: soft keys K_m = V_m^T h (training) or onehot
    K_m = h[argmax V_m]; v_mode: soft mixes V_m, onehot mixes the programs' argmax one-hots; programs with keep False
    are excluded from routing; tau_r = ROUTING_TEMPERATURE; the state of decision_state() under `convention`. After
    every pick, `weights` holds the routing weights w [R, M] of that decision."""

    def __init__(self, model: HeuristicAttentionStudent, bank, k_mode: str, v_mode: str, keep: torch.Tensor,
                 device: torch.device, convention: str = "legacy"):
        if k_mode not in ("soft", "onehot") or v_mode not in ("soft", "onehot"):
            raise ValueError(f"k_mode / v_mode must be soft or onehot, got {k_mode} / {v_mode}")
        self.model, self.bank, self.k_mode, self.v_mode, self.device = model, bank, k_mode, v_mode, device
        self.convention = convention
        self.keep = keep.to(device).view(1, -1)
        self.weights: Optional[torch.Tensor] = None

    def pick(self, vis, cur, step):
        N = vis.size(1)
        batch = decision_state(self.locs, vis, cur, self.first, step, self.convention)
        V = self.bank.compute_probs(batch)[0]                                  # [R, M, N]
        td = build_td(batch, self.device)
        h, graph_context = self.model.encode(td)
        q = self.model.compute_query(h, graph_context, td)
        top = V.masked_fill(vis.unsqueeze(1), float("-inf")).argmax(dim=-1)   # [R, M]
        if self.k_mode == "onehot":
            K = h.gather(1, top.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        else:
            K = torch.bmm(V, h)
        scores = torch.bmm(K, q.unsqueeze(-1)).squeeze(-1) / self.model.routing_temperature
        w = F.softmax(scores.masked_fill(~self.keep, float("-inf")), dim=-1)
        self.weights = w
        V_mix = F.one_hot(top, num_classes=N).float() if self.v_mode == "onehot" else V
        y_hat = (w.unsqueeze(-1) * V_mix).sum(dim=1)
        return y_hat.masked_fill(vis, float("-inf")).argmax(dim=-1)


class AdditivePolicy(_RowStepper):
    """NAM / MoET (<pkg>/<method>_pomo.py, model <pkg>/pomo_tsp<N>/best.pt): <method>_logits_tsp on the inputs of the
    bank programs (the state of decision_state() under `convention`), argmax over the unvisited nodes."""

    def __init__(self, method: str, pkg: str, device: torch.device, convention: str = "legacy"):
        self.method, self.pkg, self.device, self.N = method, pkg, device, None
        self.convention = convention

    def begin(self, locs, first):
        N = locs.size(1)
        if self.N != N:
            mod = _module(os.path.join(self.pkg, f"{self.method}_pomo.py"))
            load = mod.load_nam if self.method == "nam" else mod.load_moet
            self.model, _ = load(_model_file(self.pkg, N, "best.pt"), self.device)
            self.logits, self.N = getattr(mod, f"{self.method}_logits_tsp"), N
        super().begin(locs, first)

    def pick(self, vis, cur, step):
        b = decision_state(self.locs, vis, cur, self.first, step, self.convention)
        R = vis.size(0)
        step_norm = b["step_i"].float() / b["step_max"].float().clamp_min(1.0)
        y = self.logits(self.model, b["locs"], b["current_node"].long().view(R), b["mask"].bool(), b["visited"],
                        step_norm, b["first_node"].long().view(R))
        return y.masked_fill(vis, float("-inf")).argmax(dim=-1)


class TreePolicy(_RowStepper):
    """Decision tree (<pkg>/trees/pomo_tsp<N>.pkl, scikit-learn; depth d4 / d6 / d8 / d12 / staged) on the features
    of <pkg>/features.py (current, first and previous node, unvisited set, step, static pairwise distances and
    convex hull), evaluated by <pkg>/gpu_tree.py; argmax of the leaf probability over the unvisited nodes."""

    def __init__(self, pkg: str, depth: str, device: torch.device):
        self.pkg, self.depth, self.device, self.N = pkg, depth, device, None
        self.FT, self.G = _module(os.path.join(pkg, "features.py")), _module(os.path.join(pkg, "gpu_tree.py"))

    def layout(self, N):
        return (250 if N >= 100 else 500), max(1, 4000 // N), N

    def begin(self, locs, first):
        N = locs.size(1)
        if self.N != N:
            import pickle
            path = os.path.join(self.pkg, "trees", f"pomo_tsp{N}.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path}: no tree for N={N}")
            with open(path, "rb") as f:
                pl = pickle.load(f)
            if self.depth == "staged":
                self.tree = self.G.TreePolicy("staged", [pl["trees"][f"staged{i}"] for i in range(3)], self.device,
                                              stage_feat_idx=pl["stage_feature_index"],
                                              stage_thresholds=pl["stage_thresholds"])
            else:
                self.tree = self.G.TreePolicy(self.depth, [pl["trees"][self.depth]], self.device)
            self.N = N
        super().begin(locs, first)
        S = first.size(1)
        self.D, self.hull = repeat_rows(self.FT.pairwise_dist(locs), S), repeat_rows(self.FT.static_hull(locs), S)
        self.prev = self.cur = self.first

    def pick(self, vis, cur, step):
        feas = ~vis
        feats = self.FT.tsp_features(self.locs, self.D, self.hull, cur, self.first, self.prev, feas, step)
        self.cur = cur
        return self.G.masked_argmax(self.tree.score(feats), feas)

    def advance(self, node):
        self.prev = self.cur


class FablePolicy(_RowStepper):
    """Agent policy of <pkg>/policies/tsp_policy.py with the frozen parameters of policies/frozen_params.json: tsp_probs
    with the static pairwise distances, the tour start as the closing node and the running centroid of the visited
    nodes; argmax."""

    def __init__(self, pkg: str, device: torch.device):
        import json
        self.P, self.device = _module(os.path.join(pkg, "policies", "tsp_policy.py")), device
        d = json.load(open(os.path.join(pkg, "policies", "frozen_params.json")))
        if "tsp_frozen_at" not in d.get("meta", {}):
            raise ValueError(f"{pkg}: the TSP parameters are not frozen")
        self.p = self.P.TSPParams(**d["tsp"])

    def layout(self, N):
        return 2048, max(1, 2048 // N), N

    def begin(self, locs, first):
        super().begin(locs, first)
        self.D = repeat_rows(self.P.pairwise_dist(locs), first.size(1))
        self.end_loc = self.locs[self.ar, self.first]
        self.vis_sum = self.locs[self.ar, self.first].clone()

    def pick(self, vis, cur, step):
        return self.P.tsp_probs(self.locs, ~vis, self.locs[self.ar, cur], self.end_loc, self.p, self.D,
                                self.vis_sum / step).argmax(-1)

    def advance(self, node):
        self.vis_sum = self.vis_sum + self.locs[self.ar, node.reshape(-1)]


class GptPolicy(_RowStepper):
    """Agent policy of <pkg>/whitebox_routing/tsp.py: pomo_action_distribution with progress (t - 1) / (N - 1), checked
    by whitebox_routing/evaluate.py's _check_probs; argmax. Multi-start runs one start node at a time."""

    def __init__(self, pkg: str, device: torch.device):
        self.T = _module(os.path.join(pkg, "whitebox_routing", "tsp.py"))
        self.E = _module(os.path.join(pkg, "whitebox_routing", "evaluate.py"))
        self.device = device

    def layout(self, N):
        return 1000, 1000, 1

    def pick(self, vis, cur, step):
        R, N = vis.shape
        feasible = ~vis
        progress = torch.full((R,), (step - 1) / max(N - 1, 1), device=vis.device)
        probs = self.T.pomo_action_distribution(self.locs, cur, feasible, vis.float(), progress, self.first)
        self.E._check_probs(probs, feasible)
        return probs.argmax(-1)
