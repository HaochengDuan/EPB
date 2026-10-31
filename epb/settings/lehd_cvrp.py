"""LEHD-CVRP: a LEHD teacher on CVRP; programs and router see the LEHD sub-problem (the K unvisited customers, the depot
and the current node).

Data: the K-bucketed decision-step shards written by epb.teachers.lehd_cvrp

    <data.path>/manifest.pt                      {"format": "lehd_cvrp_kbucket_v1", "N", "n_instances", "per_K", ...}
    <data.path>/bucket_K{K:03d}_shard_{S:04d}.pt  one row per physical decision step, K = number of unvisited customers

mmap'd; a batch holds rows of one K. The action space is [depot, customer 1..N] (A = N + 1, depot at index 0); the
teacher's customer probabilities are scattered to unselected_idx and its depot probability goes to index 0.

Programs:
    heuristic(unselected_locs [B,K,2], unselected_demand [B,K], current_loc [B,2], depot_loc [B,2], load [B],
              feasibility_mask [B,K]) -> [B, K+1] logits
    customers first (infeasible -inf), then a finite depot logit; the probability wrapper sets the depot to -inf
    when the vehicle is at the depot.

Every batch carries "_row" (row id, (K << 40) | row index within the K-bucket) and "_inst" (instance id) besides the
shard fields the setting uses.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import warnings
from collections import Counter
from collections.abc import Mapping
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Data, Setting, eval_option, register, state_convention

_FORMAT = "lehd_cvrp_kbucket_v1"
_ROW_SHIFT = 40                               # row id = (K << 40) | row index within the K-bucket
_ROW_MASK = (1 << _ROW_SHIFT) - 1
# Shard fields a batch carries (the shards also hold pomo_idx, teacher_logits_K, teacher_depot_logit, next_node_local;
# nothing here reads them, and not loading them saves disk reads).
_KEYS = ("depot_loc", "current_loc", "load", "unselected_idx", "unselected_locs", "unselected_demand",
         "feasibility_mask", "teacher_probs_K", "teacher_depot_prob", "instance_id")
_GENERATOR = "lehd_cvrp"                      # the generator module, python -m epb.teachers.lehd_cvrp
_REQUIRED = object()


def _cfg(cfg, path: str, default=_REQUIRED):
    """cfg value at a dotted path (attribute or mapping access); `default` when the key is absent."""
    node = cfg
    for part in path.split("."):
        try:
            node = node[part] if isinstance(node, Mapping) else getattr(node, part)
        except (KeyError, AttributeError):
            if default is _REQUIRED:
                raise KeyError(f"lehd_cvrp: config key {path!r} is missing") from None
            return default
    return node


# ─────────────────────────────────────────────────────────────────────────────
# Data: K-bucketed shards, instance-level split, homogeneous-K batches
# ─────────────────────────────────────────────────────────────────────────────

class _KBucketShards:
    """The mmap'd shards of one dataset, indexed by (K, row index within the K-bucket over its shards)."""

    def __init__(self, path: str, mmap: bool = True):
        manifest_path = os.path.join(path, "manifest.pt")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"manifest.pt not found in {path}")
        self.manifest = torch.load(manifest_path, weights_only=False)
        if self.manifest.get("format") != _FORMAT:
            raise ValueError(f"unexpected manifest format {self.manifest.get('format')!r} in {path}")
        self.N = int(self.manifest["N"])
        self.n_instances = int(self.manifest["n_instances"])
        self.shards: Dict[int, List[Dict[str, torch.Tensor]]] = {}
        self.offsets: Dict[int, List[int]] = {}               # cumulative row offsets of the shards of one K
        for K in sorted(int(k) for k in self.manifest["per_K"]):
            shards, offsets = [], [0]
            for fname in self.manifest["per_K"][K]["shards"]:
                p = os.path.join(path, fname)
                try:
                    s = torch.load(p, weights_only=True, mmap=mmap, map_location="cpu")
                except Exception:
                    s = torch.load(p, weights_only=False, map_location="cpu")
                shards.append(s)
                offsets.append(offsets[-1] + int(s["instance_id"].size(0)))
            self.shards[K] = shards
            self.offsets[K] = offsets

    def n_rows(self, K: int) -> int:
        return self.offsets[K][-1] if K in self.offsets else 0

    def instance_id(self, K: int) -> torch.Tensor:
        """[n_rows(K)] int64 instance id of every row of the K-bucket."""
        return torch.cat([s["instance_id"] for s in self.shards[K]], dim=0).long()

    def get_rows(self, K: int, raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Rows `raw` (int64, indices within the K-bucket) of the K-bucket, in the given order."""
        shards, offsets = self.shards[K], self.offsets[K]
        if len(shards) == 1:
            return {k: shards[0][k].index_select(0, raw) for k in _KEYS}
        si = torch.searchsorted(torch.tensor(offsets[1:]), raw, right=True)
        local = raw - torch.tensor(offsets[:-1])[si]
        out = {k: torch.empty((raw.numel(),) + tuple(shards[0][k].shape[1:]), dtype=shards[0][k].dtype)
               for k in _KEYS}
        for s in si.unique().tolist():
            pos = (si == s).nonzero(as_tuple=True)[0]
            loc = local.index_select(0, pos)
            for k in _KEYS:
                out[k].index_copy_(0, pos, shards[s][k].index_select(0, loc))
        return out

    def batch(self, K: int, raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = raw.long()
        rows = self.get_rows(K, raw)
        rows["_row"] = (K << _ROW_SHIFT) | raw
        rows["_inst"] = rows.pop("instance_id").long()
        return rows


class _PlanBatches(torch.utils.data.Dataset):
    """One planned batch per item (DataLoader workers prefetch whole batches)."""

    def __init__(self, shards: _KBucketShards, plan: List[Tuple[int, torch.Tensor]]):
        self.shards, self.plan = shards, plan

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, i):
        K, raw = self.plan[i]
        return self.shards.batch(K, raw)


class _KBatches:
    """Homogeneous-K batches over given rows {K: row indices within the K-bucket} (port of KBucketBatchLoader /
    FaultSubsetKLoader). Pass e (set_epoch(e), else the number of completed passes) takes its order from
    Generator(seed + e): K-buckets in ascending order, the rows of each K shuffled and chunked, then the chunk list
    shuffled. Unshuffled: K ascending, rows in order. num_workers > 0 prefetches whole batches in forked workers."""

    def __init__(self, shards: _KBucketShards, rows: Dict[int, torch.Tensor], batch_size: int, shuffle: bool,
                 drop_undersized: bool, seed: int, num_workers: int = 0):
        self.shards, self.rows = shards, rows
        self.batch_size, self.shuffle, self.drop_undersized = int(batch_size), shuffle, drop_undersized
        self.seed, self.num_workers = int(seed), int(num_workers)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _plan(self) -> List[Tuple[int, torch.Tensor]]:
        plan: List[Tuple[int, torch.Tensor]] = []
        gen = torch.Generator().manual_seed(self._epoch + self.seed)
        bs = self.batch_size
        for K in sorted(self.rows):
            idx = self.rows[K]
            n = idx.numel()
            if n == 0 or (n < bs and self.drop_undersized):
                continue
            order = torch.randperm(n, generator=gen) if self.shuffle else torch.arange(n)
            for i in range(0, n, bs):
                chunk = order[i:i + bs]
                if chunk.numel() < bs and self.drop_undersized:
                    continue
                plan.append((K, idx[chunk]))
        if self.shuffle:
            plan = [plan[i] for i in torch.randperm(len(plan), generator=gen).tolist()]
        return plan

    def __len__(self) -> int:
        bs = self.batch_size
        return sum((n // bs) if self.drop_undersized else -(-n // bs)
                   for n in (int(v.numel()) for v in self.rows.values()))

    def __iter__(self):
        plan = self._plan()
        if self.num_workers > 0 and len(plan) > 1:
            # fork: the workers share the mmap'd shards (spawn would pickle them)
            batches = torch.utils.data.DataLoader(
                _PlanBatches(self.shards, plan), batch_size=None, shuffle=False,
                num_workers=min(self.num_workers, len(plan)), multiprocessing_context="fork")
        else:
            batches = (self.shards.batch(K, raw) for K, raw in plan)
        for batch in batches:
            yield batch
        self._epoch += 1


def split_train_val_by_instance(n_instances: int, val_ratio: float, seed: int = 42
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(train_iids, val_iids): a random permutation of the instance ids, the first max(1, n * val_ratio) are val
    (same draw as the original trainer)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_instances, generator=g)
    n_val = max(1, int(n_instances * val_ratio))
    return perm[n_val:], perm[:n_val]


# ─────────────────────────────────────────────────────────────────────────────
# Router (LEHD-style subgraph encoder + dynamic-key routing over the bank)
# ─────────────────────────────────────────────────────────────────────────────

class _RouterDecoderLayer(nn.Module):
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


class LEHDStyleRouterStudentCVRP(nn.Module):
    """Sequence [depot ‖ unselected customers (x, y, demand) ‖ current] -> encoder + decoder layers ->
    q = state_proj([h_depot ‖ h_current ‖ load]) + graph_proj(mean h_unsel); dynamic keys K_m = V_K_m @ h_unsel;
    w = softmax(K_m · q / τ_r); y_hat = Σ_m w_m V_full_m over [depot, customers] (depot prob in V_full[:, m, 0])."""

    def __init__(
        self,
        num_heuristics: int,
        N: int,
        embed_dim: int = 128,
        num_decoder_layers: int = 3,
        num_encoder_layers: int = 1,
        num_heads: int = 8,
        qkv_dim: int = 16,
        ff_hidden: int = 512,
        routing_temperature: Optional[float] = None,
    ):
        super().__init__()
        self.num_heuristics = num_heuristics
        self.N = N
        self.embed_dim = embed_dim
        self.routing_temperature = routing_temperature or (embed_dim ** 0.5)

        # Embeddings
        self.embed_depot    = nn.Linear(2, embed_dim)
        self.embed_customer = nn.Linear(3, embed_dim)   # [x, y, demand]
        self.embed_current  = nn.Linear(2, embed_dim)

        # Cross-node mixing
        self.encoder_layers = nn.ModuleList([
            _RouterDecoderLayer(embed_dim, num_heads, ff_hidden, qkv_dim)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            _RouterDecoderLayer(embed_dim, num_heads, ff_hidden, qkv_dim)
            for _ in range(num_decoder_layers)
        ])

        # Query construction
        self.state_proj = nn.Linear(2 * embed_dim + 1, embed_dim, bias=False)
        self.graph_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def encode_subgraph(
        self,
        unselected_locs: torch.Tensor,    # [B, K, 2]
        unselected_demand: torch.Tensor,  # [B, K]
        current_loc: torch.Tensor,        # [B, 2]
        depot_loc: torch.Tensor,          # [B, 2]
    ):
        d = self.embed_depot(depot_loc).unsqueeze(1)                          # [B, 1, D]
        c = self.embed_customer(
            torch.cat([unselected_locs, unselected_demand.unsqueeze(-1)], -1)
        )                                                                     # [B, K, D]
        e = self.embed_current(current_loc).unsqueeze(1)                      # [B, 1, D]
        x = torch.cat([d, c, e], dim=1)                                       # [B, K+2, D]
        for layer in self.encoder_layers:
            x = layer(x)
        for layer in self.decoder_layers:
            x = layer(x)
        return x[:, 0], x[:, 1:-1], x[:, -1]                                  # h_depot, h_unsel, h_current

    def compute_query(self, h_depot, h_current, h_unsel, load):
        state_ctx = self.state_proj(
            torch.cat([h_depot, h_current, load.unsqueeze(-1)], dim=-1)
        )                                                                     # [B, D]
        graph_ctx = self.graph_proj(h_unsel.mean(dim=1))                      # [B, D]
        return state_ctx + graph_ctx                                          # [B, D]

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        V_full: torch.Tensor,    # [B, M, N+1]  scattered (depot at idx 0)
        V_K: torch.Tensor,       # [B, M, K]    customer-side gather
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        unselected_locs   = batch["unselected_locs"]
        unselected_demand = batch["unselected_demand"]
        current_loc       = batch["current_loc"]
        depot_loc         = batch["depot_loc"]
        load              = batch["load"]

        h_depot, h_unsel, h_current = self.encode_subgraph(
            unselected_locs, unselected_demand, current_loc, depot_loc
        )                                                                     # [B,D], [B,K,D], [B,D]

        q = self.compute_query(h_depot, h_current, h_unsel, load)             # [B, D]

        # Routing weights (dynamic K)
        K_dyn = torch.bmm(V_K.detach(), h_unsel)                              # [B, M, D]
        scores = torch.bmm(K_dyn, q.unsqueeze(-1)).squeeze(-1) / self.routing_temperature
        w = F.softmax(scores, dim=-1)                                         # [B, M]

        # Mixture directly on N+1 — depot prob baked into V_full[:, m, 0]
        y_hat_full = (w.unsqueeze(-1) * V_full).sum(dim=1)                    # [B, N+1]

        if return_scores:
            return y_hat_full, w, scores
        return y_hat_full, w


_ROUTER_DEFAULTS = {"embed_dim": 128, "num_encoder_layers": 1, "num_decoder_layers": 3, "num_heads": 8,
                    "qkv_dim": 16, "ff_hidden": 512, "routing_temperature": 1.0}


# ─────────────────────────────────────────────────────────────────────────────
# Programs: output -> probabilities, CPU checks
# ─────────────────────────────────────────────────────────────────────────────

def _program_probs(fn: Callable, args: tuple, unsel_idx: torch.Tensor, N: int, tau_h: float
                   ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """[B, K+1] program output -> (V_full [B, N+1], V_K [B, K], ok). A raise / wrong shape / wrong device falls back
    to zeros with -inf at the infeasible customers (ok=False); depot -> index 0, customers -> unselected_idx, one
    softmax over N+1 after /tau_h, depot -inf at the depot; NaN rows (non-finite output) become 0."""
    unsel_locs, unsel_dem, cur_loc, dep_loc, load, feas = args
    device = feas.device
    B, K = feas.shape
    at_depot = (cur_loc == dep_loc).all(dim=-1)                               # [B] bool
    ok = True
    try:
        out = fn(*args)
        if out.shape != (B, K + 1):
            raise ValueError(f"shape {tuple(out.shape)} expected ({B}, {K + 1})")
        if out.device != device:
            raise ValueError(f"output on {out.device}, inputs on {device}")
        out = out.float()
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as e:
        warnings.warn(f"program runtime error: {e}; depot=0 + customers=0")
        out = torch.zeros(B, K + 1, device=device)
        out[:, :K] = out[:, :K].masked_fill(~feas, float("-inf"))
        ok = False

    sharp = out / tau_h
    logits_full = torch.full((B, N + 1), float("-inf"), device=device, dtype=sharp.dtype)
    logits_full[:, 0] = sharp[:, K]
    logits_full.scatter_(1, unsel_idx, sharp[:, :K])
    # at-depot rows: the depot can't be selected from the depot
    logits_full[:, 0] = torch.where(
        at_depot, torch.full_like(logits_full[:, 0], float("-inf")),
        logits_full[:, 0],
    )
    V_full = torch.nan_to_num(F.softmax(logits_full, dim=-1), nan=0.0)        # [B, N+1]
    V_K = V_full.gather(1, unsel_idx)                                         # [B, K]
    return V_full, V_K, ok


def _ulps(y: float, k: int) -> float:
    """y moved by k float32 ulps (k < 0: down)."""
    t = torch.tensor(float(y))
    for _ in range(abs(k)):
        t = torch.nextafter(t, torch.tensor(-1.0 if k < 0 else 2.0))
    return float(t)


def _edge_batch(K: int = 20, seed: int = 1) -> tuple:
    """Geometric edge cases. Around an anchor (the depot, or the current node), customers lie on the rays at angle 0
    and pi, one / two / four float32 ulps below and one above angle 0 and pi (an angle in [0, 2 pi) built as
    atan2 + 2 pi then rounds to exactly 2 pi, i.e. one bin past the last), on the vertical rays and at the anchor
    itself; plus a duplicate pair of customers and coordinates exactly 0 and 1. Rows vary the depot position, the
    current node (at the depot, away from it) and the load (1.0, 0.3)."""
    g = torch.Generator().manual_seed(seed)
    r = 0.2
    rows = []
    for depot in ((0.5, 0.5), (0.3, 0.7)):
        for anchor_is_depot, cur in ((True, depot), (True, (0.45, 0.6)), (False, (0.45, 0.6))):
            ax, ay = depot if anchor_is_depot else cur
            pts = [(ax + r, ay), (ax + r, _ulps(ay, -1)), (ax + r, _ulps(ay, -2)), (ax + r, _ulps(ay, -4)),
                   (ax + r, _ulps(ay, 1)), (ax - r, ay), (ax - r, _ulps(ay, -1)), (ax - r, _ulps(ay, 1)),
                   (ax, ay + r), (ax, ay - r), (ax, ay), (1.0, ay), (0.0, 0.0), (1.0, 1.0)]
            locs = torch.rand(K, 2, generator=g) * 0.3 + 0.35
            for j, p in enumerate(pts):
                locs[j] = torch.tensor(p)
            locs[len(pts) + 1] = locs[len(pts)]                                # duplicate pair
            for load in (1.0, 0.3):
                rows.append((locs.clone(), torch.tensor(cur), torch.tensor(depot), load))
    B = len(rows)
    unsel = torch.stack([x[0] for x in rows])
    cur = torch.stack([x[1] for x in rows])
    dep = torch.stack([x[2] for x in rows])
    load = torch.tensor([x[3] for x in rows])
    dem = torch.randint(1, 10, (B, K), generator=g).float() / 50
    feas = dem <= load[:, None]
    return unsel, dem, cur, dep, load, feas


_ARG_NAMES = ("unselected_locs", "unselected_demand", "current_loc", "depot_loc", "load", "feasibility_mask")


def _run_check(fn: Callable, args: tuple, label: str) -> torch.Tensor:
    try:
        out = fn(*args)
    except Exception as e:
        raise ValueError(f"{label}: raised {type(e).__name__}: {e}") from None
    if not isinstance(out, torch.Tensor):
        raise ValueError(f"{label}: returned {type(out).__name__}, not a tensor")
    B, K = args[5].shape
    if tuple(out.shape) != (B, K + 1):
        raise ValueError(f"{label}: output shape {tuple(out.shape)}, expected ({B}, {K + 1})")
    if out.device != args[5].device:
        raise ValueError(f"{label}: output on {out.device}, inputs on {args[5].device}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Failure cases (prompt text; geometry only, no node ids)
# ─────────────────────────────────────────────────────────────────────────────

def _theta_deg(dx: float, dy: float) -> float:
    """atan2(dy, dx) in degrees, range [-180, 180]."""
    return round(math.degrees(math.atan2(dy, dx)), 1)


def _case_info(row: Dict, probs: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> Dict:
    """Features of one decision step under a distribution `probs` [N+1] (port of the old _featurize_fault_case)."""
    locs_b = row["unselected_locs"].float()                               # [K, 2]
    dem_b  = row["unselected_demand"].float()                             # [K]
    feas_b = row["feasibility_mask"].bool()                               # [K]
    idx_b  = row["unselected_idx"].long()                                 # [K]
    cur = row["current_loc"].float()                                      # [2]
    dep = row["depot_loc"].float()                                        # [2]
    load_b = float(row["load"].float().item())
    mask_b = mask.bool()
    V_full_b = probs.float()                                              # [N+1]
    teacher_full_b = teacher.float()                                      # [N+1]
    K = locs_b.size(0)

    cur_pos = (round(float(cur[0]), 3), round(float(cur[1]), 3))
    depot_pos = (round(float(dep[0]), 3), round(float(dep[1]), 3))
    dist_cur_to_depot = round(float((cur - dep).pow(2).sum().sqrt().item()), 3)

    # Distances + polar (depot-anchored)
    dcur = (locs_b - cur.unsqueeze(0)).pow(2).sum(-1).sqrt()              # [K]
    ddep = (locs_b - dep.unsqueeze(0)).pow(2).sum(-1).sqrt()              # [K]
    rel_x = locs_b[:, 0] - dep[0]
    rel_y = locs_b[:, 1] - dep[1]
    theta_dep = torch.atan2(rel_y, rel_x) * (180.0 / math.pi)             # [K]
    cw_sav = ddep + dist_cur_to_depot - dcur                              # [K]

    cur_theta = _theta_deg(float(cur[0]) - depot_pos[0],
                           float(cur[1]) - depot_pos[1])

    # Centroid of the unvisited customers
    centroid = locs_b.mean(dim=0)
    dist_cur_to_centroid = round(
        float(((centroid[0]-cur[0])**2 + (centroid[1]-cur[1])**2).sqrt().item()), 3)
    depot_to_centroid = round(
        float(((centroid[0]-dep[0])**2 + (centroid[1]-dep[1])**2).sqrt().item()), 3)

    # Sorted-by-θ parallel lists (for arm / bimodality detection)
    order = torch.argsort(theta_dep)
    theta_sorted = [round(float(theta_dep[i]), 1) for i in order]
    r_sorted     = [round(float(ddep[i]),      3) for i in order]
    dcur_sorted  = [round(float(dcur[i]),      3) for i in order]
    dem_sorted   = [round(float(dem_b[i]),     3) for i in order]
    feas_sorted  = [bool(feas_b[i].item())          for i in order]

    n_feas = int(feas_b.sum().item())

    def _node_dict(k: int) -> Dict:
        return {
            "pos": (round(float(locs_b[k, 0]), 3),
                    round(float(locs_b[k, 1]), 3)),
            "dcur": round(float(dcur[k].item()), 3),
            "ddepot": round(float(ddep[k].item()), 3),
            "theta_from_depot": round(float(theta_dep[k].item()), 1),
            "demand": round(float(dem_b[k].item()), 3),
            "cw_sav": round(float(cw_sav[k].item()), 3),
            "feasible": bool(feas_b[k].item()),
        }

    def _depot_dict(prob_val: float) -> Dict:
        return {"is_depot": True, "prob_full": round(prob_val, 4), "pos": depot_pos, "dcur": dist_cur_to_depot}

    def _topk_full(probs_full: torch.Tensor) -> list:
        masked = probs_full.masked_fill(~mask_b, float("-inf"))
        out = []
        for global_idx in masked.argsort(descending=True)[:2].tolist():
            if global_idx == 0:
                out.append(_depot_dict(float(probs_full[0].item())))
            else:
                local_k = (idx_b == global_idx).nonzero(as_tuple=True)[0]
                if local_k.numel() == 1:
                    out.append(_node_dict(int(local_k.item())))
        return out

    teacher_top2 = _topk_full(teacher_full_b)
    heu_top2     = _topk_full(V_full_b)
    teacher_chose_depot = bool(teacher_top2 and teacher_top2[0].get("is_depot"))
    heu_chose_depot     = bool(heu_top2     and heu_top2[0].get("is_depot"))
    # forced = no customer is feasible; proactive = some customer was feasible but the teacher returned anyway
    teacher_depot_mode = ("forced" if n_feas == 0 else "proactive") if teacher_chose_depot else None

    # Nearest 5 feasible (by dcur)
    dcur_feas = dcur.clone().masked_fill(~feas_b, float("inf"))
    nf_idx = dcur_feas.argsort()[:min(5, n_feas)].tolist()

    return {
        "current_load":           round(load_b, 3),
        "K":                      K,
        "n_feasible":             n_feas,
        "n_infeasible":           K - n_feas,
        "at_depot":               bool((cur == dep).all().item()),
        "cur_pos":                cur_pos,
        "depot_pos":              depot_pos,
        "dist_cur_to_depot":      dist_cur_to_depot,
        "cur_theta_from_depot":   cur_theta,
        "demand_remaining":       round(float(dem_b.sum().item()), 3),
        "min_unsel_demand":       round(float(dem_b.min().item()), 3),
        "max_unsel_demand":       round(float(dem_b.max().item()), 3),
        "dist_cur_to_centroid":   dist_cur_to_centroid,
        "depot_to_centroid":      depot_to_centroid,
        "unvisited_theta_sorted": theta_sorted,
        "unvisited_r_sorted":     r_sorted,
        "unvisited_dcur_sorted":  dcur_sorted,
        "unvisited_dem_sorted":   dem_sorted,
        "unvisited_feas_sorted":  feas_sorted,
        "teacher_depot_prob":     round(float(teacher_full_b[0].item()), 4),
        "heuristic_depot_prob":   round(float(V_full_b[0].item()), 4),
        "teacher_chose_depot":    teacher_chose_depot,
        "heuristic_chose_depot":  heu_chose_depot,
        "teacher_depot_mode":     teacher_depot_mode,
        "nearest_feasible":       [_node_dict(k) for k in nf_idx],
        "heuristic_top2":         heu_top2,
        "teacher_top2":           teacher_top2,
    }


def _fmt_node(x: Dict) -> str:
    """A candidate by geometry only — no stable id is exposed."""
    if x.get("is_depot"):
        return (f"DEPOT pos={x['pos']} dcur={x['dcur']} "
                f"prob_full={x.get('prob_full','?')}")
    return (f"pos={x['pos']} dcur={x['dcur']} ddepot={x['ddepot']} "
            f"θ_dep={x['theta_from_depot']}° dem={x['demand']} "
            f"sav={x['cw_sav']} feas={x['feasible']}")


_WS_ONLY = re.compile(r"^[ \t]+$", re.MULTILINE)


def _render_code_case(si: Dict) -> str:
    """The case block of the code backward prompt (the old text, including its indentation)."""
    nf = "\n".join("  " + _fmt_node(x) for x in si["nearest_feasible"])
    hl = "\n".join("  " + _fmt_node(x) for x in si["heuristic_top2"])
    tl = "\n".join("  " + _fmt_node(x) for x in si["teacher_top2"])
    theta_str = ", ".join(str(x) for x in si["unvisited_theta_sorted"]) or "(none)"
    r_str     = ", ".join(str(x) for x in si["unvisited_r_sorted"])     or "(none)"
    dcur_str  = ", ".join(str(x) for x in si["unvisited_dcur_sorted"])  or "(none)"
    dem_str   = ", ".join(str(x) for x in si["unvisited_dem_sorted"])   or "(none)"
    feas_str  = ", ".join("T" if f else "F" for f in si["unvisited_feas_sorted"]) or "(none)"
    depot_block = (f"  teacher chose depot? {si['teacher_chose_depot']}"
                   f"  (mode={si['teacher_depot_mode']}, prob={si['teacher_depot_prob']})\n"
                   f"  heuristic chose depot? {si['heuristic_chose_depot']}"
                   f"  (prob={si['heuristic_depot_prob']})")
    text = "\n".join([
        "    STATE:",
        f"      cur_pos={si['cur_pos']}  cur_θ_from_depot={si['cur_theta_from_depot']}°",
        f"      depot_pos={si['depot_pos']}  dist_cur_to_depot={si['dist_cur_to_depot']}",
        f"      load={si['current_load']}  K={si['K']}  feasible={si['n_feasible']}  infeasible={si['n_infeasible']}",
        f"      demand_remaining={si['demand_remaining']}  min/max demand={si['min_unsel_demand']}/{si['max_unsel_demand']}",
        f"      dist_cur→centroid={si['dist_cur_to_centroid']}  depot→centroid={si['depot_to_centroid']}",
        "",
        "    DEPOT VOTES:",
        f"    {depot_block}",
        "",
        "    UNVISITED CUSTOMERS (sorted by θ from depot — gaps reveal route arms):",
        f"      θ° :   {theta_str}",
        f"      r  :   {r_str}     (distance from depot)",
        f"      dcur:  {dcur_str}  (distance from current)",
        f"      dem :  {dem_str}",
        f"      feas:  {feas_str}",
        "",
        "    NEAREST FEASIBLE (top 5 by dcur, geometry only):",
        f"    {nf}",
        "",
        "    HEURISTIC top-2 (over customers + depot):",
        f"    {hl}",
        "    TEACHER top-2 (over customers + depot):",
        f"    {tl}",
    ])
    return _WS_ONLY.sub("", text)


def _render_desc_case(si: Dict) -> str:
    """The case block of the description backward prompt (the old text, including its indentation)."""
    hl = "\n".join("  " + _fmt_node(x) for x in si["heuristic_top2"])
    tl = "\n".join("  " + _fmt_node(x) for x in si["teacher_top2"])
    text = "\n".join([
        f"      State: load={si['current_load']}  K={si['K']}  feasible={si['n_feasible']}",
        f"             cur_θ_from_depot={si['cur_theta_from_depot']}°",
        f"             dist_cur_to_depot={si['dist_cur_to_depot']}",
        f"             dist_cur→centroid={si['dist_cur_to_centroid']}",
        "      DEPOT VOTES:",
        f"        teacher chose depot? {si['teacher_chose_depot']}  (mode={si['teacher_depot_mode']}, p={si['teacher_depot_prob']})",
        f"        heuristic chose depot? {si['heuristic_chose_depot']}  (p={si['heuristic_depot_prob']})",
        f"      HEURISTIC top-2 (geometry only): {hl}",
        f"      TEACHER top-2 (geometry only):   {tl}",
    ])
    return _WS_ONLY.sub("", text)


def _masked_kl(teacher: torch.Tensor, probs: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(teacher ‖ probs) over the mask, along the last dim."""
    contrib = teacher * (teacher.clamp_min(eps).log() - probs.clamp_min(eps).log())
    return contrib.masked_fill(~mask, 0.0).sum(dim=-1)


def _fault_info(row: Dict, V_bank: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> Dict:
    """Features of one bank fault: the step under the programs' mean distribution, plus the programs' top-1 votes,
    the smallest program KL and the teacher's farness ratio (its customer's distance / nearest feasible distance)."""
    V_bank = V_bank.float()
    teacher = teacher.float()
    mask = mask.bool()
    info = _case_info(row, V_bank.mean(dim=0), teacher, mask)
    info["min_program_kl"] = float(_masked_kl(teacher.unsqueeze(0), V_bank, mask.unsqueeze(0)).min().item())
    votes = Counter(V_bank.masked_fill(~mask.unsqueeze(0), float("-inf")).argmax(dim=-1).tolist())
    idx = row["unselected_idx"].long()
    locs = row["unselected_locs"].float()
    vote_list = []
    for g, n in votes.most_common(3):
        if g == 0:
            vote_list.append(f"DEPOT ×{n}")
        else:
            k = (idx == g).nonzero(as_tuple=True)[0]
            if k.numel() == 1:
                kk = int(k.item())
                vote_list.append(f"pos=({round(float(locs[kk, 0]), 3)}, {round(float(locs[kk, 1]), 3)}) ×{n}")
    info["bank_votes"] = vote_list
    info["n_programs"] = int(V_bank.size(0))
    ft = info["teacher_top2"][0] if info["teacher_top2"] else None
    nearest = info["nearest_feasible"][0]["dcur"] if info["nearest_feasible"] else None
    info["farness_ratio"] = (round(ft["dcur"] / max(nearest, 1e-6), 3)
                             if ft is not None and not ft.get("is_depot") and nearest is not None else None)
    return info


def _quartiles(vals: List[float]) -> str:
    v = sorted(vals)
    if not v:
        return "(none)"
    q = [v[min(len(v) - 1, int(p * len(v)))] for p in (0.25, 0.5, 0.75)]
    return f"min={round(v[0], 3)}  q25={round(q[0], 3)}  median={round(q[1], 3)}  q75={round(q[2], 3)}  max={round(v[-1], 3)}"


# ─────────────────────────────────────────────────────────────────────────────
# Setting
# ─────────────────────────────────────────────────────────────────────────────

@register
class LEHDCVRP(Setting):
    name = "lehd_cvrp"
    problem = "cvrp"
    # Training of the released student: the original trainer's defaults for the router and
    # tau_h, the flags shared by its rounds (batch 4096, lr 1e-4, bf16 AMP, no lr schedule), TextGrad accepts on a
    # 0.5% bank-KL gain over 10 val batches, adds of 10 candidates refined for 5 epochs and kept at >= 0.001 coverage
    # gain.
    defaults = {
        "data": {"val_ratio": 0.05},
        "generate": {"problem_size": 100, "batch_size": 512, "target_per_bucket": 4096, "nco_root": None,
                     "teacher_capacity_rule": True},
        "bank": {"num_programs": 24, "tau_h": 0.1},
        "router": dict(_ROUTER_DEFAULTS),
        "train": {"batch_size": 4096, "lr": 1e-4, "amp": True, "scheduler": "none"},
        "textgrad": {"backward_every_n": 6000, "min_improvement": 0.005, "eval_batches": 10},
        "add": {"n_candidates": 10, "min_gain": 0.001, "pretrain_epochs": 5},
        "prompts": {"teacher": "LEHD-CVRP"},
    }

    def __init__(self):
        self.N: Optional[int] = None          # problem size, set by build_data (from the manifest)

    def _n(self) -> int:
        if self.N is None:
            raise RuntimeError("lehd_cvrp: the problem size comes from the data; call build_data first")
        return self.N

    # ---- data ---------------------------------------------------------------------------------------------------
    def build_data(self, cfg, split: Optional[Dict] = None) -> Data:
        path = _cfg(cfg, "data.path")
        shards = _KBucketShards(path)
        self.N = shards.N
        if split is None:
            val_ratio, seed = float(_cfg(cfg, "data.val_ratio")), int(_cfg(cfg, "run.seed"))
            _, val_iids = split_train_val_by_instance(shards.n_instances, val_ratio, seed=seed)
            split = {"format": _FORMAT, "n_instances": shards.n_instances, "val_ratio": val_ratio, "seed": seed,
                     "val_instance_ids": sorted(val_iids.tolist())}
        elif split.get("format") != _FORMAT or int(split["n_instances"]) != shards.n_instances:
            raise ValueError(f"lehd_cvrp: the saved split ({split.get('format')}, n_instances="
                             f"{split.get('n_instances')}) does not belong to {path} "
                             f"({_FORMAT}, n_instances={shards.n_instances})")
        val_ids = torch.as_tensor(split["val_instance_ids"], dtype=torch.long)

        # One pass over the instance ids; a K-bucket without rows on one side gets an empty index (not "all rows").
        is_val: Dict[int, torch.Tensor] = {}
        train_rows: Dict[int, torch.Tensor] = {}
        val_rows: Dict[int, torch.Tensor] = {}
        for K in shards.shards:
            v = torch.isin(shards.instance_id(K), val_ids)
            is_val[K] = v
            val_rows[K] = v.nonzero(as_tuple=True)[0].int()
            train_rows[K] = (~v).nonzero(as_tuple=True)[0].int()

        bs = int(_cfg(cfg, "train.batch_size"))
        seed = int(_cfg(cfg, "run.seed"))
        nw = int(_cfg(cfg, "data.num_workers", 0) or 0)
        train = _KBatches(shards, train_rows, bs, shuffle=True, drop_undersized=True, seed=seed, num_workers=nw)
        val = _KBatches(shards, val_rows, bs, shuffle=False, drop_undersized=False, seed=seed, num_workers=nw)

        def scan(scan_seed: int):
            return _KBatches(shards, train_rows, bs, shuffle=True, drop_undersized=False, seed=scan_seed,
                             num_workers=nw)

        def subset(row_ids, batch_size: int, shuffle: bool, subset_seed: int):
            ids = torch.as_tensor(row_ids, dtype=torch.long).flatten()
            Ks, raw = ids >> _ROW_SHIFT, ids & _ROW_MASK
            rows: Dict[int, torch.Tensor] = {}
            for K in Ks.unique().tolist():
                r = raw[Ks == K]
                if K not in is_val or bool((r >= shards.n_rows(K)).any()):
                    raise ValueError(f"lehd_cvrp: row ids of K={K} are not rows of the dataset")
                if bool(is_val[K][r].any()):
                    raise ValueError(f"lehd_cvrp: {int(is_val[K][r].sum())} row ids of K={K} are val rows")
                rows[K] = r
            return _KBatches(shards, rows, batch_size, shuffle=shuffle, drop_undersized=False, seed=subset_seed,
                             num_workers=nw)

        return Data(train=train, val=val, problem_size=shards.N, scan=scan, subset=subset,
                    set_epoch=train.set_epoch, split=split)

    def row_ids(self, batch: Dict) -> torch.Tensor:
        return batch["_row"]

    def instance_ids(self, batch: Dict) -> torch.Tensor:
        return batch["_inst"]

    def target(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """teacher_full [B, N+1] (depot prob at 0, customer probs at unselected_idx) and mask_full [B, N+1]: the depot
        iff not at the depot, a customer iff unvisited and feasible."""
        N = self._n()
        teacher_probs_K  = batch["teacher_probs_K"].float()
        teacher_depot    = batch["teacher_depot_prob"].float()
        unselected_idx   = batch["unselected_idx"].long()
        feasibility_mask = batch["feasibility_mask"].bool()
        B, K = teacher_probs_K.shape
        device = teacher_probs_K.device

        teacher_full = torch.zeros(B, N + 1, device=device, dtype=torch.float32)
        teacher_full[:, 0] = teacher_depot
        teacher_full.scatter_(1, unselected_idx, teacher_probs_K)

        at_depot = (batch["current_loc"] == batch["depot_loc"]).all(dim=-1).to(device)    # [B]
        mask_full = torch.zeros(B, N + 1, dtype=torch.bool, device=device)
        mask_full[:, 0] = ~at_depot
        mask_full.scatter_(1, unselected_idx, feasibility_mask)
        return teacher_full, mask_full

    # ---- programs -----------------------------------------------------------------------------------------------
    def program_args(self, batch: Dict) -> tuple:
        return (batch["unselected_locs"].float(), batch["unselected_demand"].float(),
                batch["current_loc"].float(), batch["depot_loc"].float(),
                batch["load"].float(), batch["feasibility_mask"].bool())

    def program_probs(self, fn: Callable, batch: Dict, tau_h: float
                      ) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        return _program_probs(fn, self.program_args(batch), batch["unselected_idx"].long(), self._n(),
                              float(tau_h))

    def check_program(self, fn: Callable) -> None:
        g = torch.Generator().manual_seed(0)
        B, K = 4, 17
        unsel = torch.rand(B, K, 2, generator=g)
        dem = torch.rand(B, K, generator=g) * 0.3
        cur = torch.rand(B, 2, generator=g)
        dep = torch.rand(B, 2, generator=g)
        ld = torch.rand(B, generator=g)
        feas = dem <= ld[:, None]
        args = (unsel, dem, cur, dep, ld, feas)
        before = [a.clone() for a in args]
        out = _run_check(fn, args, f"K={K}")
        for name, a, b in zip(_ARG_NAMES, args, before):
            if not torch.equal(a, b):
                raise ValueError(f"K={K}: modifies its input {name} in place")
        if not (torch.isfinite(out[:, :K]) | (out[:, :K] == float("-inf"))).all():
            raise ValueError(f"K={K}: customer logits must be finite or -inf")
        if not torch.isfinite(out[:, K]).all():
            raise ValueError(f"K={K}: the depot logit (last column) must be finite")

        unsel1 = torch.rand(B, 1, 2, generator=g)
        dem1 = torch.rand(B, 1, generator=g) * 0.3
        out1 = _run_check(fn, (unsel1, dem1, cur, dep, ld, dem1 <= ld[:, None]), "K=1")
        if not torch.isfinite(out1[:, 1]).all():
            raise ValueError("K=1: the depot logit (last column) must be finite")

        none = torch.zeros(B, K, dtype=torch.bool)
        out0 = _run_check(fn, (unsel, dem, cur, dep, ld, none), "no feasible customer")
        if not torch.isfinite(out0[:, K]).all():
            raise ValueError("no feasible customer: the depot logit (last column) must be finite")

        _run_check(fn, _edge_batch(), "geometric edge cases")

    # ---- router -------------------------------------------------------------------------------------------------
    def _router_kwargs(self, cfg) -> Dict:
        return {k: _cfg(cfg, f"router.{k}", v) for k, v in _ROUTER_DEFAULTS.items()}

    def build_router(self, cfg, num_programs: int) -> nn.Module:
        return LEHDStyleRouterStudentCVRP(num_heuristics=num_programs, N=self.N, **self._router_kwargs(cfg))

    def router_forward(self, model: nn.Module, batch: Dict, V_full: torch.Tensor, V_K: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """V_full [B, M, N+1], V_K [B, M, K] (the programs stacked on dim 1)."""
        return model(batch, V_full, V_K, return_scores=True)

    def ckpt_args(self, cfg, num_programs: int) -> Dict:
        kw = self._router_kwargs(cfg)
        return {"heuristic_temperature": float(_cfg(cfg, "bank.tau_h")),
                "embed_dim": kw["embed_dim"], "num_encoder_layers": kw["num_encoder_layers"],
                "num_decoder_layers": kw["num_decoder_layers"], "num_heads": kw["num_heads"],
                "qkv_dim": kw["qkv_dim"], "ff_hidden": kw["ff_hidden"],
                "routing_temperature": kw["routing_temperature"]}

    # ---- prompts ------------------------------------------------------------------------------------------------
    def prompt_vars(self, cfg, problem_size: int) -> Dict[str, str]:
        """teacher: the teacher's name (prompts.teacher; "LEHD-CVRP"); teacher_family: the name without "-CVRP"
        (for "<family>-style"); problem_size, n_plus_1 (action space), k_min / k_max (range of K); tau_h."""
        N = int(problem_size)
        teacher = _cfg(cfg, "prompts.teacher", None) or "LEHD-CVRP"
        family = teacher[:-len("-CVRP")] if teacher.endswith("-CVRP") else teacher
        return {"teacher": teacher, "teacher_family": family, "problem_size": str(N), "n_plus_1": str(N + 1),
                "k_min": "1", "k_max": str(N - 1), "tau_h": f"{float(_cfg(cfg, 'bank.tau_h')):g}"}

    def failure_case(self, row: Dict, V_m: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor, kind: str) -> str:
        if kind == "code":
            return _render_code_case(_case_info(row, V_m, teacher, mask))
        if kind == "desc":
            return _render_desc_case(_case_info(row, V_m, teacher, mask))
        raise ValueError(f"unknown failure-case kind {kind!r}")

    def fault_case(self, row: Dict, V_bank: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> str:
        si = _fault_info(row, V_bank, teacher, mask)
        theta_str = ", ".join(str(x) for x in si["unvisited_theta_sorted"]) or "(none)"
        nf = " | ".join(_fmt_node(x) for x in si["nearest_feasible"][:3]) or "(none)"
        bk = " | ".join(_fmt_node(x) for x in si["heuristic_top2"]) or "(none)"
        tk = " | ".join(_fmt_node(x) for x in si["teacher_top2"]) or "(none)"
        votes = " | ".join(si["bank_votes"]) or "(none)"
        lines = [
            f"Case [min program KL={si['min_program_kl']:.4f}]:",
            f"  STATE: cur_pos={si['cur_pos']}  load={si['current_load']}  at_depot={si['at_depot']}",
            f"         depot_pos={si['depot_pos']}  dist_cur_to_depot={si['dist_cur_to_depot']}",
            f"  POLAR θ from depot: cur={'(at depot)' if si['at_depot'] else str(si['cur_theta_from_depot']) + '°'}",
            f"  REMAINING: {si['K']} unvisited ({si['n_infeasible']} capacity-locked)"
            f"  demand_remaining={si['demand_remaining']}  min/max demand={si['min_unsel_demand']}/{si['max_unsel_demand']}",
            f"             d(cur→bulk)={si['dist_cur_to_centroid']}  d(depot→bulk)={si['depot_to_centroid']}",
            f"             unvisited θ° (sorted): {theta_str}",
            f"  NEAREST FEASIBLE (top 3 by dcur): {nf}",
            f"  BANK top-2 (mean of the {si['n_programs']} programs): {bk}",
            f"  BANK top-1 votes (programs per pick): {votes}",
            f"  TEACHER top-2: {tk}",
        ]
        if si["teacher_chose_depot"] or si["heuristic_chose_depot"]:
            t_label = f"DEPOT({si['teacher_depot_mode']})" if si["teacher_chose_depot"] else "customer"
            b_label = "DEPOT" if si["heuristic_chose_depot"] else "customer"
            lines.append(f"  ⚠ DEPOT CHOICE: teacher={t_label}  bank={b_label}")
        return "\n".join(lines)

    def fault_summary(self, rows: List[Dict], V_banks: List[torch.Tensor], teachers: List[torch.Tensor],
                      masks: List[torch.Tensor]) -> str:
        infos = [_fault_info(r, v, t, m) for r, v, t, m in zip(rows, V_banks, teachers, masks)]
        if not infos:
            return ""
        t_dep = [i for i in infos if i["teacher_chose_depot"]]
        b_dep = [i for i in infos if i["heuristic_chose_depot"]]
        far = [i["farness_ratio"] for i in infos if i["farness_ratio"] is not None]
        return "\n".join([
            f"Statistics over the {len(infos)} failure cases:",
            f"  - teacher chose DEPOT (proactive): {sum(1 for i in t_dep if i['teacher_depot_mode'] == 'proactive')}",
            f"  - teacher chose DEPOT (forced):    {sum(1 for i in t_dep if i['teacher_depot_mode'] == 'forced')}",
            f"  - bank    chose DEPOT:             {len(b_dep)}",
            f"  - mismatch (teacher=customer, bank=DEPOT): {sum(1 for i in b_dep if not i['teacher_chose_depot'])}",
            f"  - mismatch (teacher=DEPOT, bank=customer): {sum(1 for i in t_dep if not i['heuristic_chose_depot'])}",
            f"  - vehicle at the depot: {sum(1 for i in infos if i['at_depot'])}",
            f"  - load: {_quartiles([i['current_load'] for i in infos])}",
            f"  - K (unvisited): {_quartiles([float(i['K']) for i in infos])}",
            f"  - farness ratio (teacher's customer distance / nearest feasible distance, {len(far)} customer picks): "
            f"{_quartiles(far)}; > 1.5 in {sum(1 for x in far if x > 1.5)}",
        ])

    # ---- data generation ----------------------------------------------------------------------------------------
    def generate(self, cfg) -> None:
        """Runs the generator python -m epb.teachers.lehd_cvrp in a process of its own (greedy LEHD rollouts;
        NCO_code's LEHD model code from generate.nco_root, the clone or its single_objective/) and checks the written
        format. With
        generate.teacher_capacity_rule (default true) it passes --teacher-capacity-rule: the stored feasibility mask
        and the transition follow the teacher's capacity rule (an exact fill fits, demand <= load + 1e-6), the rule
        the evaluation applies; false reproduces the data of the paper's runs (no tolerance)."""
        out = _cfg(cfg, "data.path")
        if os.path.exists(os.path.join(out, "manifest.pt")):
            raise FileExistsError(f"{out} already holds a dataset (manifest.pt)")
        N = int(_cfg(cfg, "generate.problem_size", 100))
        n_inst = int(_cfg(cfg, "generate.num_instances"))
        from ..teachers import command
        args = ["--checkpoint", _cfg(cfg, "generate.teacher_ckpt"),
                "--problem-size", N, "--num-instances", n_inst,
                "--seed", int(_cfg(cfg, "generate.seed")), "--output", out,
                "--batch-size", int(_cfg(cfg, "generate.batch_size", 512)),
                "--target-per-bucket", int(_cfg(cfg, "generate.target_per_bucket", 4096))]
        if _cfg(cfg, "generate.teacher_capacity_rule", True):
            args.append("--teacher-capacity-rule")
        nco = _cfg(cfg, "generate.nco_root", None)
        if nco:
            args += ["--nco-root", os.path.abspath(nco)]
        cmd, env = command(_GENERATOR, args)
        print("[lehd_cvrp] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=env)
        _check_generated(out, N, n_inst)

    # ---- evaluation (epb.eval) ----------------------------------------------------------------------------------
    # Engines, policies and their rules: the evaluation section at the end of the file.
    search_default = "rrc"
    budget_default = {100: 200, 500: 500}

    def eval_args(self, parser) -> None:
        parser.add_argument("--rrc-seed", type=int, default=_RRC_SEED,
                            help="lehd_cvrp: seed of the RRC segment draws (the generator is reseeded at every call)")
        parser.add_argument("--sub-min", type=int, default=_RRC_SUB_MIN, help="lehd_cvrp: shortest RRC segment")

    def eval_policy(self, method: str, a):
        """teacher: the LEHD-CVRP model of --teacher-ckpt (code from --nco-root). epb / expert / frozen: the router of
        --ckpt over the bank of --bank (soft K and V, --tau-h, --drop). nam / moet: <pkg>/<method>_lehd.py with
        <pkg>/lehd_cvrp100/best.pt (or --ckpt). tree: <pkg>/trees/lehd_cvrp100.pkl at --depth (default d12). agent:
        --pkg agents/fable51 or agents/gpt6."""
        device = torch.device(a.device)
        if method == "teacher":
            return _EvalTeacher.load(a, device)
        if method in _ROUTER_METHODS:
            return _EvalRouter.load(self, method, a, device)
        if method in ("nam", "moet"):
            return _EvalScorer.load(self, method, a, device)
        if method == "tree":
            return _EvalTree.load(self, a, device)
        if method == "agent":
            return _load_agent(self, a, device)
        raise ValueError(f"lehd_cvrp: unknown method {method!r}")

    def eval_greedy(self, policy, inst: Dict[str, torch.Tensor], a) -> torch.Tensor:
        out = torch.cat([res[1].cpu() for _, _, _, res in _eval_greedy(self, policy, inst, a)])
        _report(policy)
        return out

    def eval_search(self, policy, inst: Dict[str, torch.Tensor], a) -> torch.Tensor:
        if a.search != "rrc":
            raise ValueError(f"lehd_cvrp: search {a.search!r}; the LEHD settings search with rrc")
        base = int(a.slice.split(":")[0]) if a.slice else 0
        out = []
        for lo, hi, t, (order, cost, _) in _eval_greedy(self, policy, inst, a):
            self.N = t[1].size(1)
            perm, cost = policy.rrc_start(t, order, cost)
            name = a.method + (f"_{a.tag}" if a.tag else "")
            out.append(_rrc(policy, t[3], t[4], perm, cost, int(a.budget), int(a.sub_min), int(a.rrc_seed),
                            tag=f"{name} {base + lo}:{base + hi}")[1].cpu())
        _report(policy)
        return torch.cat(out)

    def eval_top1(self, teacher, policy, inst: Dict[str, torch.Tensor], a) -> Dict[str, torch.Tensor]:
        out: Dict[str, List[torch.Tensor]] = {k: [] for k in ("hits", "states", "hits_all", "states_all")}
        for lo, hi, t, (_, _, seqs) in _eval_greedy(self, teacher, inst, a):
            self.N = t[1].size(1)
            st = _teacher_states(seqs, t[4].cpu())
            choice = st["tnext"].clone() if policy is teacher else policy.choices(st, t[3], t[4])
            hit = (choice == st["tnext"]).double()
            prim = ~st["forced"]
            ii, n = st["inst"], hi - lo
            one = torch.ones_like(hit)
            out["hits"].append(torch.zeros(n, dtype=torch.float64).index_add_(0, ii[prim], hit[prim]))
            out["states"].append(torch.zeros(n, dtype=torch.float64).index_add_(0, ii[prim], one[prim]))
            out["hits_all"].append(torch.zeros(n, dtype=torch.float64).index_add_(0, ii, hit))
            out["states_all"].append(torch.zeros(n, dtype=torch.float64).index_add_(0, ii, one))
        _report(policy)
        return {k: torch.cat(v) for k, v in out.items()}


def _check_generated(path: str, N: int, n_instances: int) -> None:
    """The written dataset has the K-bucket format, the requested N / instance count, every listed shard, and shards
    (smallest and largest K) with the expected fields, shapes and a normalised teacher distribution."""
    m = torch.load(os.path.join(path, "manifest.pt"), weights_only=False)
    if m.get("format") != _FORMAT or int(m["N"]) != N or int(m["n_instances"]) != n_instances:
        raise ValueError(f"{path}: manifest {m.get('format')}, N={m.get('N')}, n_instances={m.get('n_instances')}; "
                         f"expected {_FORMAT}, N={N}, n_instances={n_instances}")
    per_K = m["per_K"]
    if not per_K or sum(v["n_rows"] for v in per_K.values()) != m["total_rows"]:
        raise ValueError(f"{path}: per-K row counts do not add up to total_rows")
    for K, v in per_K.items():
        if not 1 <= int(K) <= N:
            raise ValueError(f"{path}: K-bucket {K} outside [1, {N}]")
        for fname in v["shards"]:
            if not os.path.exists(os.path.join(path, fname)):
                raise ValueError(f"{path}: missing shard {fname}")
    for K in {min(per_K), max(per_K)}:
        s = torch.load(os.path.join(path, per_K[K]["shards"][0]), weights_only=True, mmap=True, map_location="cpu")
        n = s["instance_id"].size(0)
        shapes = {"depot_loc": (n, 2), "current_loc": (n, 2), "load": (n,), "unselected_idx": (n, K),
                  "unselected_locs": (n, K, 2), "unselected_demand": (n, K), "feasibility_mask": (n, K),
                  "teacher_probs_K": (n, K), "teacher_depot_prob": (n,)}
        for k, shape in shapes.items():
            if k not in s or tuple(s[k].shape) != shape:
                raise ValueError(f"{path}: shard of K={K}: field {k} has shape "
                                 f"{tuple(s[k].shape) if k in s else None}, expected {shape}")
        idx = s["unselected_idx"].long()
        if int(idx.min()) < 1 or int(idx.max()) > N or s["feasibility_mask"].dtype != torch.bool:
            raise ValueError(f"{path}: shard of K={K}: unselected_idx outside [1, {N}] or a non-bool mask")
        total = s["teacher_probs_K"].float().sum(-1) + s["teacher_depot_prob"].float()
        if not torch.allclose(total, torch.ones_like(total), atol=2e-2):
            raise ValueError(f"{path}: shard of K={K}: teacher probabilities do not sum to 1")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (epb.eval): greedy from the depot, random re-construct, teacher-forced top-1
# ─────────────────────────────────────────────────────────────────────────────
#
# A tour is a customer order [B, N] (ids 1..N) plus the physical sequence its decoder produced (0 = depot, from the
# depot back to it). A cost is the length of what the method's own decoder did (its own depot timing, not a replay
# that returns only when a customer does not fit), float32 as the rollouts compute it.
#
#   greedy  from the depot. Teacher: its greedy order, replayed by the teacher deciding every depot return (checked
#           against its one-pass greedy). Step policies (router-bank students, NAM, MoET, the GPT-6 agent): one move
#           over [depot, customers] per step from the LEHD sub-problem; a forced return when away from the depot and
#           no customer fits.
#   rrc     random re-construct of the customer order, from the greedy: every iteration draws one segment (start,
#           length >= --sub-min) for the whole batch from a CPU generator seeded with --rrc-seed at every call, the
#           method re-decodes the segment's customers as a sub-problem from the depot, replays the spliced order with
#           its own depot timing, and each instance keeps the new order if it is shorter.
#   top1    along the teacher's greedy physical sequence (a via-depot step c -> 0 -> j is two states, the final return
#           is not a state), the method's next node at every state vs the teacher's. Primary states have more than
#           one feasible action (customers with demand <= load + 1e-6; the depot when away from it).
#
# The segment draws do not depend on the batching and acceptance is per instance, so --slice and --batch split the
# work without changing the RRC moves; float32 decoding can still differ with the batch, and a bad program output
# falls back for every row of its call (below), so the rows it affects depend on the batch. Router methods decode
# with the bank and the router above (soft K and V), with a stricter rule for a bad program output than training's:
# a wrong shape, a non-finite depot logit or a non-finite logit at a feasible customer falls back for the whole call
# like a raise. The step policies are the reference on
# the CPU (on the GPU the path is launch-bound); the tree and the Fable agent ran on the GPU.
#
# Two baselines bring their own engine. The decision tree acts in LEHD's 2N space (customer j direct or via the
# depot, one customer per step), which the [depot, customers] engine cannot express (a via decision picks the next
# customer at the customer state, not at the depot); it runs on its package's 2N rollout and replay. The Fable agent
# accumulates its costs move by move inside its engine and its RRC accepts on those costs, so it runs on its own
# rollout and replay too. Both use the RRC loop below.

_EVAL_EPS = 1e-6                  # capacity tolerance of every rollout: demand <= load + eps fits (the teacher's rule,
#                                   an exact fill fits; the rows of epb.generate's data use the same rule)
_EVAL_CAPACITY = 1.0
_EVAL_BATCH = 32                  # instances per decoding call (--batch)
_RRC_SEED = 1334
_RRC_SUB_MIN = 4
_TOP1_CHUNK = 256                 # teacher states per router call of the top-1 pass
_ROUTER_METHODS = ("epb", "expert", "frozen")


def _eval_tensors(inst: Dict[str, torch.Tensor], device) -> Tuple[torch.Tensor, ...]:
    """(depot [B,1,2], nodes [B,N,2], demand [B,N], locs_full [B,N+1,2], dem_full [B,N+1]) on device, index 0 = depot."""
    depot, nodes, dem = (inst[k].to(device) for k in ("depot_xy", "node_xy", "node_demand"))
    locs_full = torch.cat([depot, nodes], dim=1)
    dem_full = torch.cat([torch.zeros(dem.size(0), 1, device=dem.device), dem], dim=1)
    return depot, nodes, dem, locs_full, dem_full


def _seq_cost(locs_full: torch.Tensor, seqs: List[List[int]]) -> torch.Tensor:
    """[B] float32 length of each physical sequence."""
    costs = torch.zeros(len(seqs), device=locs_full.device)
    for i, seq in enumerate(seqs):
        pts = locs_full[i].index_select(0, torch.tensor(seq, device=locs_full.device, dtype=torch.long))
        costs[i] = (pts[1:] - pts[:-1]).norm(dim=-1).sum()
    return costs


def _subproblem(locs_full: torch.Tensor, dem_full: torch.Tensor, rows: torch.Tensor, cand: torch.Tensor,
                cur: torch.Tensor, load: torch.Tensor, K: int) -> Dict[str, torch.Tensor]:
    """The batch of the LEHD sub-problem at instances `rows` (indices into locs_full): their K candidates (cand [n, N+1]
    bool, ascending id order), current node cur [n] and load [n]."""
    n = rows.numel()
    unsel_idx = cand.nonzero(as_tuple=False)[:, 1].view(n, K)
    r = rows.unsqueeze(1).expand(-1, K)
    unsel_dem = dem_full[r, unsel_idx]
    return {"unselected_idx": unsel_idx, "unselected_locs": locs_full[r, unsel_idx], "unselected_demand": unsel_dem,
            "current_loc": locs_full[rows, cur], "depot_loc": locs_full[rows, 0], "load": load,
            "feasibility_mask": unsel_dem <= load.unsqueeze(-1) + _EVAL_EPS}


def _masked_argmax_full(y_hat: torch.Tensor, batch: Dict[str, torch.Tensor], at_depot: torch.Tensor) -> torch.Tensor:
    """argmax of y_hat [n, N+1] over the feasible customers and, away from the depot, the depot."""
    mask = torch.zeros_like(y_hat, dtype=torch.bool)
    mask[:, 0] = ~at_depot
    mask.scatter_(1, batch["unselected_idx"], batch["feasibility_mask"])
    return y_hat.masked_fill(~mask, float("-inf")).argmax(dim=-1)


def _eval_rule(fn: Callable) -> Callable:
    """fn under the evaluation rule for a bad output: a wrong shape, a non-finite depot logit or a non-finite logit at a
    feasible customer raises, so _program_probs answers the whole call with its fallback."""
    def checked(unselected_locs, unselected_demand, current_loc, depot_loc, load, feasibility_mask):
        out = fn(unselected_locs, unselected_demand, current_loc, depot_loc, load, feasibility_mask)
        B, K = feasibility_mask.shape
        if out.shape != (B, K + 1):
            raise ValueError(f"shape {tuple(out.shape)} expected ({B}, {K + 1})")
        if not torch.isfinite(out[:, K]).all():
            raise ValueError("non-finite depot logit")
        if not torch.isfinite(out[:, :K][feasibility_mask]).all():
            raise ValueError("non-finite at feasible position")
        return out
    return checked


def _import_from(directory: str, module: str):
    """Import `module` with `directory` first on sys.path (the baseline packages import their siblings by name)."""
    import importlib
    d = os.path.abspath(os.path.expanduser(directory))
    if not os.path.isdir(d):
        raise ValueError(f"lehd_cvrp: {directory} is not a directory")
    if d not in sys.path:
        sys.path.insert(0, d)
    return importlib.import_module(module)


def _trained_size_dir(setting: "LEHDCVRP") -> str:
    return f"{setting.name}{setting.defaults['generate']['problem_size']}"


def _eval_greedy(setting: "LEHDCVRP", policy, inst: Dict[str, torch.Tensor], a) -> List[tuple]:
    """Per chunk of a.batch instances: (lo, hi, tensors of _eval_tensors, (order, cost, physical sequences or None));
    kept on the policy for the search and the top-1 pass on the same instances."""
    import time
    bs = int(a.batch or _EVAL_BATCH)
    key = (inst["node_xy"].data_ptr(), tuple(inst["node_xy"].shape), bs)
    if key not in policy.cache:
        name = "teacher" if isinstance(policy, _EvalTeacher) else a.method + (f"_{a.tag}" if a.tag else "")
        base = int(a.slice.split(":")[0]) if a.slice else 0                     # chunk bounds printed in set indices
        n = inst["node_xy"].size(0)
        out = []
        for lo in range(0, n, bs):
            hi = min(lo + bs, n)
            t0 = time.time()
            t = _eval_tensors({k: v[lo:hi] for k, v in inst.items()}, policy.device)
            setting.N = t[1].size(1)
            res = policy.greedy(*t)
            print(f"  [greedy {name} {base + lo}:{base + hi}] mean_cost={res[1].mean().item():.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            out.append((lo, hi, t, res))
        policy.cache[key] = out
    return policy.cache[key]


def _report(policy) -> None:
    if isinstance(policy, _EvalRouter) and policy.fallbacks:
        print("lehd_cvrp: program fallbacks so far (decoding calls answered by the fallback): "
              + ", ".join(f"bank position {m}: {n}" for m, n in sorted(policy.fallbacks.items())), flush=True)


@torch.no_grad()
def _rrc(policy, locs_full: torch.Tensor, dem_full: torch.Tensor, perm: torch.Tensor, costs: torch.Tensor,
         budget: int, sub_min: int, seed: int, tag: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random re-construct (port of the original evaluator's RRC loop, no cap on the segment length):
    `budget` segment draws (start uniform in [0, N), length uniform in [sub_min, N], cut at the end of the order;
    a draw shorter than sub_min is skipped), shared by the whole batch. Returns (order [B, N], cost [B])."""
    B, N = perm.shape
    perm, cur = perm.clone(), costs.clone()
    g = torch.Generator(device="cpu").manual_seed(seed)
    log_every = max(1, budget // 4)
    n_acc = 0
    for it in range(budget):
        first = int(torch.randint(0, N, (1,), generator=g).item())
        length = int(torch.randint(sub_min, N + 1, (1,), generator=g).item())
        length = min(length, N - first)
        if length < sub_min:
            continue
        new_sub = policy.repair(locs_full, dem_full, perm[:, first:first + length])
        new_perm = perm.clone()
        new_perm[:, first:first + length] = new_sub[:, :length]
        new_cost, _ = policy.replay(locs_full, dem_full, new_perm)
        accept = new_cost < cur - 1e-9
        if accept.any():
            perm = torch.where(accept.unsqueeze(-1), new_perm, perm)
            cur = torch.where(accept, new_cost, cur)
            n_acc += int(accept.sum().item())
        if (it + 1) % log_every == 0 or it == budget - 1:
            print(f"  [rrc {tag}] it {it + 1}/{budget}  L={length}  acc={int(accept.sum().item())}/{B}  "
                  f"mean_cost={cur.mean().item():.4f}  total_acc={n_acc}", flush=True)
    return perm, cur


def _teacher_states(seqs: List[List[int]], dem_full: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The decision states along the teacher's physical sequences (CPU, instance-major in sequence order): inst, t,
    cur, served [S, N+1] (customers already served), load (float32, replayed as the rollouts do: refill to 1 at the
    depot, then load - demand), tnext (the teacher's next node), forced (at most one feasible action). A via-depot move
    at the depot (zero length) is dropped."""
    B, N1 = dem_full.shape
    norm = []
    for s in seqs:
        if not (s[0] == 0 and s[-1] == 0 and s[-2] != 0):
            raise RuntimeError("lehd_cvrp: a teacher sequence does not start / end at the depot")
        out = [s[0]]
        for v in s[1:]:
            if not (v == 0 and out[-1] == 0):
                out.append(v)
        norm.append(out)
    L = max(len(s) for s in norm)
    P = torch.zeros(B, L, dtype=torch.long)
    n_st = torch.tensor([len(s) - 2 for s in norm])                           # moves minus the final return
    for i, s in enumerate(norm):
        P[i, :len(s)] = torch.tensor(s)
    ar = torch.arange(B)
    served = torch.zeros(B, N1, dtype=torch.bool)
    load = torch.full((B,), _EVAL_CAPACITY)
    rec: Dict[str, List[torch.Tensor]] = {k: [] for k in ("inst", "t", "cur", "served", "load", "tnext")}
    for t in range(L - 1):
        act = t < n_st
        if not bool(act.any()):
            break
        ii = ar[act]
        rec["inst"].append(ii)
        rec["t"].append(torch.full_like(ii, t))
        rec["cur"].append(P[ii, t])
        rec["served"].append(served[ii].clone())
        rec["load"].append(load[ii].clone())
        rec["tnext"].append(P[ii, t + 1])
        nxt = P[:, t + 1]
        to_dep = nxt == 0
        load = torch.where(act & to_dep, torch.full_like(load, _EVAL_CAPACITY),
                           torch.where(act & ~to_dep, load - dem_full[ar, nxt], load))
        cust = nxt[ii] != 0
        served[ii[cust], nxt[ii][cust]] = True
    st = {k: torch.cat(v) for k, v in rec.items()}
    o = torch.argsort(st["inst"] * (L + 1) + st["t"])
    st = {k: v[o] for k, v in st.items()}
    unv = ~st["served"]
    unv[:, 0] = False
    fits = unv & (dem_full[st["inst"]] <= st["load"].unsqueeze(-1) + _EVAL_EPS)
    st["forced"] = (fits.sum(-1) + (st["cur"] != 0).long()) <= 1
    tn, cu = st["tnext"], st["cur"]
    ok = ((tn == 0) & (cu != 0)) | ((tn != 0) & fits.gather(1, tn.unsqueeze(-1)).squeeze(-1))
    if not bool(ok.all()) or int(n_st.sum()) != st["inst"].numel():
        raise RuntimeError("lehd_cvrp: a teacher move is infeasible under its own rule")
    return st


# ---- teacher ------------------------------------------------------------------------------------------------------

class _EvalPolicy:
    """What every evaluated policy has: its device, the greedy results kept for search and top-1, and the start of the
    RRC (default: the greedy order at its greedy cost)."""

    def __init__(self, device):
        self.device = device
        self.cache: Dict = {}

    def rrc_start(self, t: tuple, order: torch.Tensor, cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return order, cost


class _EvalTeacher(_EvalPolicy):
    """The LEHD-CVRP teacher (NCO_code's VRPModel with the device patches of epb.teachers.lehd_cvrp) through the
    teacher code of epb.teachers.lehd_cvrp: loader, one-pass greedy, greedy order, replay of an order with the teacher's
    own depot timing, restricted greedy on a sub-problem."""

    def __init__(self, ckpt: str, nco_root: str, device):
        super().__init__(device)
        from ..teachers import lehd_cvrp as T
        self.T = T
        self.model = T.load_lehd_cvrp(ckpt, device, nco_root=nco_root)

    @classmethod
    def load(cls, a, device) -> "_EvalTeacher":
        if not a.teacher_ckpt or not a.nco_root:
            raise ValueError("lehd_cvrp: the teacher needs --teacher-ckpt (NCO_code format) and --nco-root "
                             "(a clone of CIAM-Group/NCO_code)")
        return cls(a.teacher_ckpt, a.nco_root, device)

    def greedy(self, depot, nodes, dem, locs_full, dem_full):
        """(order [B, N], cost [B], physical sequences). The two-stage cost (order, then replay) must reproduce the
        one-pass greedy within 0.01% (mean)."""
        B = depot.size(0)
        order, _, _ = self.T.lehd_greedy_order(self.model, depot, nodes, dem, batch_size=B, device=self.device)
        cost, seqs = self.T.lehd_constrained_replay(self.model, locs_full, dem_full, order, batch_size=B,
                                                    device=self.device)
        official = self.T.lehd_cvrp_greedy(self.model, depot, nodes, dem, batch_size=B).to(self.device)
        dev_pct = abs(cost.mean() - official.mean()).item() / official.mean().item() * 100
        if not dev_pct < 0.01:
            raise RuntimeError(f"lehd_cvrp: the two-stage teacher greedy ({cost.mean().item():.6f}) does not reproduce "
                               f"the one-pass greedy ({official.mean().item():.6f}, {dev_pct:.4f}%)")
        return order, cost, seqs

    def repair(self, locs_full, dem_full, sub):
        return self.T.lehd_restricted_greedy(self.model, locs_full, dem_full, sub, batch_size=sub.size(0),
                                             device=self.device)

    def replay(self, locs_full, dem_full, perm):
        return self.T.lehd_constrained_replay(self.model, locs_full, dem_full, perm, batch_size=perm.size(0),
                                              device=self.device)


# ---- step policies: one move over [depot, customers] per step ------------------------------------------------------

class _StepPolicy(_EvalPolicy):
    """A policy that decides every physical move from the LEHD sub-problem batch: probs(batch) -> [n, N+1] over
    [depot, customers]. Decoded by _router_rollout / _router_replay and, at the teacher's states, by _router_choices."""

    def __init__(self, setting: "LEHDCVRP", device):
        super().__init__(device)
        self.setting = setting

    def probs(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def greedy(self, depot, nodes, dem, locs_full, dem_full):
        order, seqs = _router_rollout(self, locs_full, dem_full)
        return order, _seq_cost(locs_full, seqs), seqs

    def repair(self, locs_full, dem_full, sub):
        return _router_rollout(self, locs_full, dem_full, restrict_to=sub)[0]

    def replay(self, locs_full, dem_full, perm):
        return _router_replay(self, locs_full, dem_full, perm)

    def choices(self, st: Dict[str, torch.Tensor], locs_full: torch.Tensor, dem_full: torch.Tensor) -> torch.Tensor:
        return _router_choices(self, locs_full.to(self.device), dem_full.to(self.device), st)


class _EvalRouter(_StepPolicy):
    """A router-bank method: the bank's programs under _eval_rule -> Bank.compute_probs -> router_forward."""

    def __init__(self, setting: "LEHDCVRP", bank, model: nn.Module, device):
        super().__init__(setting, device)
        from ..bank import Program
        self.bank, self.model = bank, model
        self.programs = [Program(p.uid, p.code, p.description, _eval_rule(p.fn), p.update_mode, p.added_at)
                         for p in bank.programs]
        self.fallbacks: Counter = Counter()           # bank position -> decoding calls answered by the fallback

    @classmethod
    def load(cls, setting: "LEHDCVRP", method: str, a, device) -> "_EvalRouter":
        if not a.ckpt or not a.bank:
            raise ValueError(f"lehd_cvrp: {method} needs --ckpt (and a bank.json next to it, or --bank)")
        if (a.k_mode or "soft") != "soft" or (a.v_mode or "soft") != "soft":
            raise ValueError("lehd_cvrp: router methods decode with soft K and V only")
        from ..bank import Bank
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck.get("student_state_dict", ck))
        args = ck.get("args") if isinstance(ck.get("args"), Mapping) else {}
        tau_h = float(eval_option(a, "tau_h", args, "heuristic_temperature", setting.defaults["bank"]["tau_h"]))
        state_convention(a, args)        # both conventions build the same states here (validated only)
        bank = Bank(setting, tau_h).load(a.bank)
        if a.drop:
            bank.remove(list(a.drop))
        model = setting.build_router({"router": {k: args[k] for k in _ROUTER_DEFAULTS if k in args}}, len(bank))
        model.load_state_dict(sd)
        model.to(device).eval()
        # a program that runs out of GPU memory would be disabled by the bank: that changes the method, so abort
        warnings.filterwarnings("error", message=".*ran out of GPU memory.*")
        print(f"lehd_cvrp {method}: {len(bank)} programs, tau_h {tau_h:g}, router epoch {ck.get('epoch')}", flush=True)
        return cls(setting, bank, model, device)

    @torch.no_grad()
    def probs(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        V_full, V_K, ok = self.bank.compute_probs(batch, programs=self.programs)
        self.fallbacks.update((~ok).nonzero(as_tuple=True)[0].tolist())
        y_hat, _, _ = self.setting.router_forward(self.model, batch, V_full, V_K)
        return y_hat                                                          # [n, N+1]


class _EvalScorer(_StepPolicy):
    """NAM / MoET (<pkg>/<method>_lehd.py): the model scores the sub-problem like a program (customers, then the depot)
    and is the policy itself, its softmax at temperature 1 (a one-program bank with routing weight 1). An output the
    evaluation rule rejects is an error here, not a fallback."""

    def __init__(self, setting: "LEHDCVRP", fn: Callable, device):
        super().__init__(setting, device)
        self.fn = _eval_rule(fn)

    @classmethod
    def load(cls, setting: "LEHDCVRP", method: str, a, device) -> "_EvalScorer":
        if not a.pkg:
            raise ValueError(f"lehd_cvrp: {method} needs --pkg (the {method} package directory)")
        mod = _import_from(a.pkg, f"{method}_lehd")
        ckpt = a.ckpt or os.path.join(a.pkg, _trained_size_dir(setting), "best.pt")
        if method == "nam":
            model, ck = mod.load_nam(ckpt, device)
            fn, info = mod.NAMScorerCVRP(model), f"epoch {ck.get('epoch')}, {model.n_params()} parameters"
        else:
            model, ck = mod.load_moet(ckpt, device)
            fn, info = mod.MoETScorerCVRP(model), f"round {ck.get('round')}, K={model.K}"
        if ck.get("problem") != "cvrp":
            raise ValueError(f"lehd_cvrp: {ckpt} is a {ck.get('problem')} model")
        state_convention(a)
        print(f"lehd_cvrp {method}: {info}", flush=True)
        return cls(setting, fn, device)

    @torch.no_grad()
    def probs(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        V_full, _, ok = self.setting.program_probs(self.fn, batch, 1.0)
        if not ok:
            raise RuntimeError("lehd_cvrp: the model's output failed the evaluation rule (see the warning above)")
        return V_full


class _EvalGpt(_StepPolicy):
    """agents/gpt6 (whitebox_routing.evaluate_rrc_cvrp): WhiteboxAdapter puts the white-box LEHD distribution
    (whitebox_routing.cvrp.lehd_action_distribution) on [depot, customers]; EmptyBank only supplies the shapes. The
    greedy cost is the package's validate_physical_sequences (which also checks the solution)."""

    def __init__(self, setting: "LEHDCVRP", WB, device):
        super().__init__(setting, device)
        self.adapter, self.bank, self.validate = WB.WhiteboxAdapter(), WB.EmptyBank(), WB.validate_physical_sequences

    @classmethod
    def load(cls, setting: "LEHDCVRP", a, device) -> "_EvalGpt":
        return cls(setting, _import_from(a.pkg, "whitebox_routing.evaluate_rrc_cvrp"), device)

    @torch.no_grad()
    def probs(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.bank.set_N(self.setting.N)
        V_full, V_K = self.bank.compute_probs(batch, self.device)
        return self.adapter(batch, V_full, V_K)[0]

    def greedy(self, depot, nodes, dem, locs_full, dem_full):
        order, seqs = _router_rollout(self, locs_full, dem_full)
        cost, _ = self.validate(locs_full, dem_full, seqs)
        return order, cost.to(locs_full.device), seqs


@torch.no_grad()
def _router_rollout(policy: _StepPolicy, locs_full: torch.Tensor, dem_full: torch.Tensor,
                    restrict_to: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[List[int]]]:
    """Greedy decoding from the depot (port of the original evaluator's student greedy rollout). At every step the live
    instances are grouped by K (number of unvisited candidates), one call per group; a step in which some instance is
    forced back to the depot (away from it, no candidate fits) moves only the forced ones. restrict_to [B, L]: decode
    only these customers (the RRC repair). Returns (customer order [B, L], physical sequences)."""
    device = locs_full.device
    B, N1, _ = locs_full.shape
    N = N1 - 1
    policy.setting.N = N
    avail = torch.zeros(B, N + 1, dtype=torch.bool, device=device)
    if restrict_to is None:
        avail[:, 1:] = True
        L_target = torch.full((B,), N, device=device, dtype=torch.long)
    else:
        avail.scatter_(1, restrict_to, True)
        L_target = (restrict_to >= 1).sum(dim=-1)
    visited = ~avail                                                          # True = not a candidate
    visited[:, 0] = True
    current = torch.zeros(B, dtype=torch.long, device=device)
    load = torch.full((B,), _EVAL_CAPACITY, device=device)
    order: List[List[int]] = [[] for _ in range(B)]
    seqs: List[List[int]] = [[0] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(4 * N + 10):
        if done.all():
            break
        cand = (~visited) & (~done.unsqueeze(-1))
        cand[:, 0] = False
        ks = cand.sum(dim=-1)
        closing = (ks == 0) & (~done)
        if closing.any():
            for i, c in zip(closing.nonzero(as_tuple=True)[0].tolist(), current[closing].tolist()):
                if c != 0:
                    seqs[i].append(0)
            done[closing] = True
        any_feas = (cand & (dem_full <= load.unsqueeze(-1) + _EVAL_EPS)).any(dim=-1)
        at_depot = current == 0
        force = (~done) & (~at_depot) & (~any_feas) & (ks > 0)
        if force.any():
            for i in force.nonzero(as_tuple=True)[0].tolist():
                seqs[i].append(0)
            load[force] = _EVAL_CAPACITY
            current[force] = 0
            continue
        alive = (~done).nonzero(as_tuple=True)[0]
        if alive.numel() == 0:
            break
        alive = alive[any_feas[alive]]
        if alive.numel() == 0:
            continue
        ks_alive = ks[alive]
        for K in torch.unique(ks_alive).tolist():
            sub = alive[ks_alive == K]
            batch = _subproblem(locs_full, dem_full, sub, cand[sub], current[sub], load[sub], K)
            act = _masked_argmax_full(policy.probs(batch), batch, at_depot[sub])
            for i, a_ in zip(sub.tolist(), act.tolist()):
                seqs[i].append(a_)
                if a_ != 0:
                    order[i].append(a_)
            to_depot = act == 0
            if to_depot.any():
                load[sub[to_depot]] = _EVAL_CAPACITY
                current[sub[to_depot]] = 0
            to_cust = ~to_depot
            if to_cust.any():
                si, ai = sub[to_cust], act[to_cust]
                visited[si, ai] = True
                load[si] = load[si] - dem_full[si, ai]
                current[si] = ai
    for s in seqs:
        if s[-1] != 0:
            s.append(0)
    perm = torch.zeros(B, int(L_target.max().item()), dtype=torch.long, device=device)
    for i in range(B):
        if len(order[i]) != int(L_target[i].item()):
            warnings.warn(f"lehd_cvrp: instance {i} visited {len(order[i])}/{int(L_target[i].item())} customers")
        perm[i, :len(order[i])] = torch.tensor(order[i], device=device, dtype=torch.long)
    return perm, seqs


@torch.no_grad()
def _router_replay(policy: _StepPolicy, locs_full: torch.Tensor, dem_full: torch.Tensor, perm: torch.Tensor
                   ) -> Tuple[torch.Tensor, List[List[int]]]:
    """Cost of a fixed customer order [B, N] with the method's own depot timing (port of the original evaluator's
    student replay): before each customer the method returns to the depot if
    its probability of the depot beats that of the customer, or if the customer does not fit. Returns (cost [B],
    physical sequences)."""
    device = locs_full.device
    B, N1, _ = locs_full.shape
    N = N1 - 1
    policy.setting.N = N
    visited = torch.zeros(B, N + 1, dtype=torch.bool, device=device)
    visited[:, 0] = True
    current = torch.zeros(B, dtype=torch.long, device=device)
    load = torch.full((B,), _EVAL_CAPACITY, device=device)
    seqs: List[List[int]] = [[0] for _ in range(B)]
    for step in range(perm.size(1)):
        nxt = perm[:, step]
        cand = (~visited).clone()
        cand[:, 0] = False
        ks = cand.sum(dim=-1)
        for K in torch.unique(ks).tolist():
            sub = (ks == K).nonzero(as_tuple=True)[0]
            if K == 0 or sub.numel() == 0:
                continue
            batch = _subproblem(locs_full, dem_full, sub, cand[sub], current[sub], load[sub], K)
            load_sub = batch["load"]
            y_hat = policy.probs(batch)
            tgt = nxt[sub]
            depot_p = y_hat[:, 0]
            cust_p = y_hat.gather(1, tgt.unsqueeze(-1)).squeeze(-1)
            need_refill = dem_full[sub, tgt] > load_sub + _EVAL_EPS
            to_depot = ((depot_p > cust_p) | need_refill) & (current[sub] != 0)
            for i, d, c in zip(sub.tolist(), to_depot.tolist(), tgt.tolist()):
                if d:
                    seqs[i].append(0)
                seqs[i].append(c)
            base = torch.where(to_depot, torch.full_like(load_sub, _EVAL_CAPACITY), load[sub])
            load[sub] = base - dem_full[sub, tgt]
            visited[sub, tgt] = True
            current[sub] = tgt
    for s in seqs:
        if s[-1] != 0:
            s.append(0)
    return _seq_cost(locs_full, seqs), seqs


@torch.no_grad()
def _router_choices(policy: _StepPolicy, locs_full: torch.Tensor, dem_full: torch.Tensor,
                    st: Dict[str, torch.Tensor]) -> torch.Tensor:
    """[S] the method's next node at the teacher states st, decided as in _router_rollout: a forced return when away
    from the depot and no customer fits, otherwise the policy's argmax on K-grouped sub-problem batches."""
    dev = policy.device
    policy.setting.N = locs_full.size(1) - 1
    inst, cur, served, load = (st[k].to(dev) for k in ("inst", "cur", "served", "load"))
    cand = ~served
    cand[:, 0] = False
    ks = cand.sum(-1)
    any_feas = (cand & (dem_full[inst] <= load.unsqueeze(-1) + _EVAL_EPS)).any(-1)
    at_depot = cur == 0
    choice = torch.full((inst.numel(),), -1, dtype=torch.long, device=dev)
    force = (~at_depot) & (~any_feas) & (ks > 0)
    choice[force] = 0
    run = (~force) & (ks > 0) & any_feas
    for K in torch.unique(ks[run]).tolist():
        sel = (run & (ks == K)).nonzero(as_tuple=True)[0]
        for c0 in range(0, sel.numel(), _TOP1_CHUNK):
            sub = sel[c0:c0 + _TOP1_CHUNK]
            batch = _subproblem(locs_full, dem_full, inst[sub], cand[sub], cur[sub], load[sub], K)
            choice[sub] = _masked_argmax_full(policy.probs(batch), batch, at_depot[sub])
    if bool((choice < 0).any()):
        raise RuntimeError("lehd_cvrp: a teacher state without a candidate")
    return choice.cpu()


# ---- baselines with their own engine ------------------------------------------------------------------------------

class _EvalTree(_EvalPolicy):
    """Decision-tree surrogate (tree package: features.lehd_cvrp_features_r, gpu_tree.TreePolicy.score, masked argmax
    over the 2N actions [direct j if it fits | via j]) on the package's 2N engine: tree_rollout.lehd_cvrp_rollout
    (greedy, and the repair on the segment as a sub-problem) and lehd_cvrp_replay (fixed order, via vs direct by the
    tree's scores, cost tree_rollout.cost_from_order_via). As in the package's RRC loop, the search starts from the
    replay cost of the greedy order and replays with the pairwise distances of the whole instances. At a teacher state
    the 2N action maps to the next node: via -> the depot (away from it), direct j -> j; at the depot both are j."""
    STATE_CHUNK = 512                 # teacher states per tree call of the top-1 pass
    DIST_CHUNK = 64                   # instances per pairwise-distance call of the top-1 pass

    def __init__(self, pol, T, FT, GT, device):
        super().__init__(device)
        self.pol, self.T, self.FT, self.GT = pol, T, FT, GT
        self._D: Optional[torch.Tensor] = None

    @classmethod
    def load(cls, setting: "LEHDCVRP", a, device) -> "_EvalTree":
        import pickle
        if not a.pkg:
            raise ValueError("lehd_cvrp: tree needs --pkg (the tree package directory)")
        T = _import_from(a.pkg, "tree_rollout")
        FT, GT = _import_from(a.pkg, "features"), _import_from(a.pkg, "gpu_tree")
        depth = getattr(a, "depth", None) or "d12"
        with open(os.path.join(a.pkg, "trees", f"{_trained_size_dir(setting)}.pkl"), "rb") as fh:
            pl = pickle.load(fh)
        if depth == "staged":
            pol = GT.TreePolicy("staged", [pl["trees"][f"staged{i}"] for i in range(3)], device,
                                stage_feat_idx=pl["stage_feature_index"], stage_thresholds=pl["stage_thresholds"])
        elif depth in pl["trees"]:
            pol = GT.TreePolicy(depth, [pl["trees"][depth]], device)
        else:
            raise ValueError(f"lehd_cvrp: --depth {depth!r}: one of {sorted(pl['trees'])} or staged")
        print(f"lehd_cvrp tree: {depth}", flush=True)
        return cls(pol, T, FT, GT, device)

    def greedy(self, depot, nodes, dem, locs_full, dem_full):
        order, seqs, _, _ = self.T.lehd_cvrp_rollout(locs_full, dem_full, tree=self.pol)
        return order, self.T.cost_from_seqs(locs_full, seqs), seqs

    def rrc_start(self, t, order, cost):
        self._D = self.FT.pairwise_dist(t[3])
        cost0, _ = self.T.lehd_cvrp_replay(t[3], t[4], order, tree=self.pol, D=self._D, hull=None)
        return order, cost0

    def repair(self, locs_full, dem_full, sub):
        return self.T.lehd_cvrp_rollout(locs_full, dem_full, tree=self.pol, restrict_to=sub, need_seqs=False)[0]

    def replay(self, locs_full, dem_full, perm):
        return self.T.lehd_cvrp_replay(locs_full, dem_full, perm, tree=self.pol, D=self._D, hull=None,
                                       need_seqs=False)

    @torch.no_grad()
    def choices(self, st: Dict[str, torch.Tensor], locs_full: torch.Tensor, dem_full: torch.Tensor) -> torch.Tensor:
        dev = self.device
        locs, dem = locs_full.to(dev), dem_full.to(dev)
        N = locs.size(1) - 1
        D = torch.cat([self.FT.pairwise_dist(locs[i:i + self.DIST_CHUNK]) for i in range(0, locs.size(0),
                                                                                      self.DIST_CHUNK)])
        acts = []
        for c0 in range(0, st["inst"].numel(), self.STATE_CHUNK):
            sl = slice(c0, c0 + self.STATE_CHUNK)
            ii, cu, ld = st["inst"][sl].to(dev), st["cur"][sl].to(dev), st["load"][sl].to(dev)
            unvis = ~st["served"][sl].to(dev)
            unvis[:, 0] = False
            lf, df = locs[ii], dem[ii]
            fits = df[:, 1:] <= ld.unsqueeze(1) + _EVAL_EPS
            feas = torch.cat([unvis[:, 1:] & fits, unvis[:, 1:]], dim=1)
            feats = self.FT.lehd_cvrp_features_r(lf, df, D[ii], cu, unvis, ld)
            acts.append(self.GT.masked_argmax(self.pol.score(feats), feas).cpu())
        act, cur = torch.cat(acts), st["cur"]
        return torch.where(cur == 0, act % N + 1, torch.where(act >= N, torch.zeros_like(act), act + 1))


class _EvalFable(_EvalPolicy):
    """agents/fable51 on its own LEHD-CVRP engine (rollout.cvrp_engine: cvrp_greedy_from_depot, cvrp_constrained_replay;
    policies.cvrp_policy with the frozen parameters of eval/common_eval.load_frozen; customer distances by
    torch.cdist of the chunk). At a teacher state: its _step_probs and argmax (index 0 = depot)."""
    STATE_CHUNK = 1024                # teacher states per call of the top-1 pass

    def __init__(self, engine, params, device):
        super().__init__(device)
        self.E, self.p = engine, params
        self._Dc: Optional[torch.Tensor] = None

    @classmethod
    def load(cls, a, device) -> "_EvalFable":
        root = os.path.abspath(a.pkg)
        engine = _import_from(root, "rollout.cvrp_engine")
        _, params, _ = _import_from(os.path.join(root, "eval"), "common_eval").load_frozen("cvrp")
        return cls(engine, params, device)

    def greedy(self, depot, nodes, dem, locs_full, dem_full):
        order, cost = self.E.cvrp_greedy_from_depot(locs_full, dem_full, self.p, D_full=torch.cdist(nodes, nodes))
        return order, cost, None

    def rrc_start(self, t, order, cost):
        self._Dc = torch.cdist(t[1], t[1])
        return order, cost

    def repair(self, locs_full, dem_full, sub):
        return self.E.cvrp_greedy_from_depot(locs_full, dem_full, self.p, restrict_to=sub, D_full=self._Dc)[0]

    def replay(self, locs_full, dem_full, perm):
        return self.E.cvrp_constrained_replay(locs_full, dem_full, perm, self.p, D_full=self._Dc), None

    @torch.no_grad()
    def choices(self, st: Dict[str, torch.Tensor], locs_full: torch.Tensor, dem_full: torch.Tensor) -> torch.Tensor:
        dev = self.device
        locs, dem = locs_full.to(dev), dem_full.to(dev)
        nodes = locs[:, 1:].contiguous()
        Dc = torch.cdist(nodes, nodes)
        out = []
        for c0 in range(0, st["inst"].numel(), self.STATE_CHUNK):
            sl = slice(c0, c0 + self.STATE_CHUNK)
            ii = st["inst"][sl].to(dev)
            visited = st["served"][sl].to(dev).clone()
            visited[:, 0] = True
            pr, _, _ = self.E._step_probs(locs[ii], dem[ii], visited, st["cur"][sl].to(dev), st["load"][sl].to(dev),
                                          self.p, Dc[ii], _EVAL_EPS)
            out.append(pr.argmax(-1).cpu())
        return torch.cat(out)


def _load_agent(setting: "LEHDCVRP", a, device):
    """--pkg agents/gpt6 (whitebox_routing/) or agents/fable51 (rollout/, policies/, eval/)."""
    if not a.pkg:
        raise ValueError("lehd_cvrp: agent needs --pkg (an agent package directory)")
    if os.path.isdir(os.path.join(a.pkg, "whitebox_routing")):
        return _EvalGpt.load(setting, a, device)
    if os.path.isfile(os.path.join(a.pkg, "rollout", "cvrp_engine.py")):
        return _EvalFable.load(a, device)
    raise ValueError(f"lehd_cvrp: {a.pkg}: no agent policy (whitebox_routing/ or rollout/cvrp_engine.py)")
