"""POMO-CVRP setting: a POMO teacher on the capacitated VRP (depot at index 0, customers 1..N, A = N + 1).

  * Data: the decision steps of epb.teachers.pomo_cvrp, either a sharded
    directory (manifest.pt + shard_NNNN.pt, generator --no-merge) streamed one shard at a time, or a single .pt
    file. Every row carries its global row id "_row" (shard offset + local index) and an instance id "_inst" (a
    hash of the instance's coordinates and demands, identical on all rows of one instance).
  * Split: sharded data by shard (seeded shuffle of the shard list, the rule of the original trainer), a single
    file by instance (seeded shuffle of the instance groups). The generator flushes a shard between two decision
    steps, so the instances of one generation batch straddle adjacent shards: the rows of adjacent train shards
    that belong to an instance of a val shard are excluded from the train split, so no instance is on both sides.
  * Programs: heuristic(locs, demand, current_node, mask, visited, load, first_node) -> logits [B, N+1];
    V_m = masked_softmax(logits / tau_h) over the feasible nodes (as in the original trainer).
  * Router: CVRPHeuristicAttentionStudent (POMO-style encoder over depot + customers, load-aware context query,
    dynamic keys K_m = V_m @ h); parameter names are unchanged, so released checkpoints load with strict=True.
  * Prompt cases: the per-case text of the original TextGrad backward prompts and of the add round's cluster prompt.
  * Evaluation (epb.eval): every instance is decoded from all N first customers (POMO trajectories, argmax, no
    augmentation); greedy = trajectory 0 (first customer 1), search "ms" = the best of the N. Top-1 is
    teacher-forced along the teacher's greedy trajectory. Section "Evaluation" at the end of this file.
"""

import bisect
import hashlib
import math
import os
import random
import subprocess
import sys
import textwrap
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler, default_collate, get_worker_info

from . import Data, Setting, eval_option, register, state_convention


_FEEDFORWARD_HIDDEN = 512          # router encoder feed-forward width (both released students)
_ROUTING_TEMPERATURE = 1.0         # scores = K . q / T
_TEACHER = "POMO-CVRP"             # prompt name of the teacher unless cfg.prompts.teacher is set
_N_CANDIDATES = 5                  # nearest feasible nodes listed in a code failure case
_GENERATOR = "pomo_cvrp"           # the generator module, python -m epb.teachers.pomo_cvrp

_MISSING = object()


def _cfg(cfg, dotted: str, default=_MISSING):
    """cfg.a.b (attribute or dict access); `default` when the key is absent or null."""
    node = cfg
    for k in dotted.split("."):
        try:
            node = node[k] if isinstance(node, dict) else getattr(node, k)
        except (KeyError, AttributeError):
            if default is _MISSING:
                raise KeyError(f"config key {dotted!r} is required")
            return default
    if node is None and default is not _MISSING:
        return default
    return node


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask = mask.bool()
    has_valid = mask.any(dim=dim, keepdim=True)
    safe = logits.masked_fill(~mask, float("-inf"))
    safe = torch.where(has_valid, safe, torch.zeros_like(safe))
    probs = F.softmax(safe, dim=dim)
    probs = torch.where(mask, probs, torch.zeros_like(probs))
    probs = torch.where(has_valid, probs, torch.zeros_like(probs))
    return probs


def _kl_mean(teacher: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL(teacher || pred) averaged over the feasible nodes (the KL the original prompts report). [..., A] -> [...]."""
    eps = 1e-9
    mask = mask.bool()
    p = teacher.clamp(min=eps)
    q = pred.clamp(min=eps)
    kl = (p * (p.log() - q.log())).masked_fill(~mask, 0.0)
    return kl.sum(dim=-1) / mask.float().sum(dim=-1).clamp(min=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Data: rows, instance ids, shard stream, map-style views
# ─────────────────────────────────────────────────────────────────────────────

def _row_dict(s: Dict, li: int) -> Dict[str, torch.Tensor]:
    return {
        "locs":           s["locs"][li].float(),
        "demand":         s["demand"][li].float(),
        "mask":           s["mask"][li].float(),
        "teacher_probs":  s["teacher_probs"][li].float(),
        "teacher_logits": s["teacher_logits"][li].float(),
        "current_node":   s["current_node"][li].long(),
        "first_node":     s["first_node"][li].long(),
        "visited":        s["visited"][li].float(),
        "load":           s["load"][li].float(),
        "step_i":         s["step_i"][li].float(),
    }


def _instance_ids(locs: torch.Tensor, demand: torch.Tensor) -> torch.Tensor:
    """[B] int64 instance id: a 56-bit blake2b hash of the float32 bytes of each row's coordinates and demands (every
    decision step of an instance stores identical copies of both)."""
    B = locs.size(0)
    x = torch.cat([locs.detach().reshape(B, -1), demand.detach().reshape(B, -1)], dim=1).float().cpu().contiguous()
    x = x.numpy()
    return torch.tensor([int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=7).digest(), "little") for r in x],
                        dtype=torch.int64)


def _collate(rows: List[Dict]) -> Dict:
    batch = default_collate(rows)
    batch["_inst"] = _instance_ids(batch["locs"], batch["demand"])
    return batch


def _load_shard(path: str, mmap: bool = False) -> Dict:
    try:
        return torch.load(path, map_location="cpu", mmap=mmap, weights_only=True)
    except Exception:
        # weights_only=True can fail on older saved files; fall back
        return torch.load(path, map_location="cpu", weights_only=False)


def _shard_instance_ids(path: str, chunk: int = 1 << 16) -> np.ndarray:
    s = _load_shard(path, mmap=True)
    n = s["locs"].size(0)
    return torch.cat([_instance_ids(s["locs"][i:i + chunk], s["demand"][i:i + chunk])
                      for i in range(0, n, chunk)]).numpy()


def _ranges(flags: np.ndarray) -> List[List[int]]:
    """[start, end) runs of True in a 1-D bool array."""
    d = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def _straddling_rows(shard_dir: str, files: List[str], val_files: List[str]) -> Dict[str, List[List[int]]]:
    """Rows of train shards that belong to an instance of a val shard, {shard file: [[start, end), ...]}. The steps
    of one instance occupy a run of adjacent shards (the generator writes them in generation order), so only the
    train shards next to a val shard are hashed, walking outwards while they still share val instances."""
    val_set = set(val_files)
    val_inst = np.unique(np.concatenate([_shard_instance_ids(os.path.join(shard_dir, f)) for f in val_files]))
    pos = {f: i for i, f in enumerate(files)}
    exclude: Dict[str, List[List[int]]] = {}
    done = set()
    for f in val_files:
        for step in (-1, 1):
            i = pos[f] + step
            while 0 <= i < len(files) and files[i] not in val_set and files[i] not in done:
                done.add(files[i])
                hit = np.isin(_shard_instance_ids(os.path.join(shard_dir, files[i])), val_inst)
                if not hit.any():
                    break
                exclude[files[i]] = _ranges(hit)
                i += step
    return exclude


class CVRPShardStreamDataset(IterableDataset):
    """Streaming loader for the sharded teacher format.

    Reads ONE shard fully into RAM at a time (sequential disk I/O), shuffles its rows, yields per-row dicts, frees
    it, moves to the next shard. Shard order is reshuffled per epoch (call set_epoch). Workers split the shard list
    disjointly. Rows in `exclude` ({shard file: [[start, end), ...]}) are skipped; every row carries its global id
    "_row" = offsets[shard] + local index.
    """

    def __init__(self, shard_dir: str, shard_files: List[str], offsets: Dict[str, int],
                 exclude: Optional[Dict[str, List[List[int]]]] = None, shuffle: bool = True, seed: int = 0):
        self.shard_dir = shard_dir
        self.shard_files = list(shard_files)
        self.offsets = dict(offsets)
        self.exclude = dict(exclude or {})
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, e: int):
        self._epoch = int(e)

    def __iter__(self):
        info = get_worker_info()
        files = list(self.shard_files)
        wid = 0
        if info is not None:
            files = files[info.id::info.num_workers]
            wid = info.id
        rng = random.Random(self.seed * 100003 + self._epoch * 131 + wid)
        if self.shuffle:
            rng.shuffle(files)
        for sf in files:
            s = _load_shard(os.path.join(self.shard_dir, sf))
            n = s["teacher_probs"].size(0)
            order = list(range(n))
            if self.shuffle:
                rng.shuffle(order)
            skip = np.zeros(n, dtype=bool)
            for a, b in self.exclude.get(sf, []):
                skip[a:b] = True
            off = self.offsets[sf]
            for li in order:
                if skip[li]:
                    continue
                row = _row_dict(s, li)
                row["_row"] = off + li
                yield row
            del s


class _ShardRows(Dataset):
    """Map-style (mmap) view of given global rows of a sharded directory; each process opens the shards it reads.
    DataLoader workers are spawned (rl4co's torchrl import sets the start method): the row ids are a tensor, which
    reaches the workers through shared memory, and the open shards are not part of the pickled state."""

    def __init__(self, shard_dir: str, files: List[str], starts: List[int], rows: np.ndarray):
        self.shard_dir, self.files, self.starts = shard_dir, files, starts
        self.rows = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.int64))
        self._open: Dict[int, Dict] = {}

    def __getstate__(self):
        return {**self.__dict__, "_open": {}}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict:
        gid = int(self.rows[i])
        si = bisect.bisect_right(self.starts, gid) - 1
        s = self._open.get(si)
        if s is None:
            s = self._open[si] = _load_shard(os.path.join(self.shard_dir, self.files[si]), mmap=True)
        row = _row_dict(s, gid - self.starts[si])
        row["_row"] = gid
        return row


class _FileRows(Dataset):
    """Map-style view of given rows of a single-file dataset (tensors in RAM)."""

    def __init__(self, tensors: Dict[str, torch.Tensor], rows: np.ndarray):
        self.t, self.rows = tensors, torch.from_numpy(np.ascontiguousarray(rows, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict:
        gid = int(self.rows[i])
        row = _row_dict(self.t, gid)
        row["_row"] = gid
        return row


class _Permutation(Sampler):
    """Seeded permutation of range(n), redrawn per epoch (set_epoch) from an explicit generator."""

    def __init__(self, n: int, seed: int, per_epoch: bool = True):
        self.n, self.seed, self.per_epoch, self.epoch = n, seed, per_epoch, 0

    def set_epoch(self, e: int):
        self.epoch = int(e)

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        s = self.seed * 100003 + self.epoch * 131 if self.per_epoch else self.seed
        return iter(torch.randperm(self.n, generator=torch.Generator().manual_seed(s)).tolist())


def _loader(ds, batch_size: int, num_workers: int, sampler=None, persistent: bool = False) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=persistent and num_workers > 0,
                      collate_fn=_collate)


class _ShardedData:
    """Sharded directory: shard-level split, streamed train / val loaders, mmap views for scan / subset."""

    def __init__(self, shard_dir: str, seed: int, val_ratio: float, split: Optional[Dict]):
        manifest_path = os.path.join(shard_dir, "manifest.pt")
        if not os.path.exists(manifest_path):
            raise ValueError(f"Sharded dir without manifest.pt: {shard_dir}")
        man = torch.load(manifest_path, weights_only=False)
        self.shard_dir, self.seed = shard_dir, seed
        self.files = list(man["shard_files"])
        shards = [_load_shard(os.path.join(shard_dir, f), mmap=True) for f in self.files]
        self.sizes = [int(s["teacher_probs"].size(0)) for s in shards]
        self.problem_size = int(shards[0]["locs"].size(1)) - 1
        del shards
        self.starts = [0]
        for sz in self.sizes[:-1]:
            self.starts.append(self.starts[-1] + sz)
        self.total = sum(self.sizes)

        if split is None:
            # the original trainer's shard split: seeded shuffle of the shard list, the first val_ratio are val
            r = random.Random(seed)
            shuf = self.files[:]
            r.shuffle(shuf)
            n_val = max(1, int(len(shuf) * val_ratio))
            val_set = set(shuf[:n_val])
            val_files = [f for f in self.files if f in val_set]
            split = {"kind": "shard", "seed": seed, "val_ratio": val_ratio,
                     "shard_files": self.files, "shard_sizes": self.sizes,
                     "train_shards": [f for f in self.files if f not in val_set], "val_shards": val_files,
                     "exclude": _straddling_rows(shard_dir, self.files, val_files)}
        elif split.get("kind") != "shard" or list(split["shard_files"]) != self.files \
                or list(split["shard_sizes"]) != self.sizes:
            raise ValueError(f"split does not match the shards of {shard_dir}")
        if not split["train_shards"]:
            raise ValueError(f"{shard_dir}: {len(self.files)} shard(s) leave no train shard after the shard-level "
                             f"split; generate more instances or use a single .pt file (instance-level split)")
        self.split = split
        self.train_files = list(split["train_shards"])
        self.val_files = list(split["val_shards"])
        self.exclude = {f: [list(x) for x in v] for f, v in split["exclude"].items()}

        pos = {f: i for i, f in enumerate(self.files)}
        self.is_train_shard = np.zeros(len(self.files), dtype=bool)
        self.is_train_shard[[pos[f] for f in self.train_files]] = True
        ex = sorted((self.starts[pos[f]] + a, self.starts[pos[f]] + b) for f, v in self.exclude.items() for a, b in v)
        self.ex_lo = np.array([a for a, _ in ex], dtype=np.int64)
        self.ex_hi = np.array([b for _, b in ex], dtype=np.int64)
        self._train_rows = None
        n_ex = int((self.ex_hi - self.ex_lo).sum())
        print(f"[pomo_cvrp] {len(self.files)} shards -> {len(self.train_files)} train + {len(self.val_files)} val "
              f"({self.total:,} steps; {n_ex:,} train-shard steps of val instances excluded)")

    def is_train(self, ids: np.ndarray) -> np.ndarray:
        si = np.searchsorted(np.array(self.starts), ids, side="right") - 1
        ok = (ids >= 0) & (ids < self.total) & self.is_train_shard[si.clip(0, len(self.files) - 1)]
        if self.ex_lo.size:
            j = np.searchsorted(self.ex_lo, ids, side="right") - 1
            ok &= ~((j >= 0) & (ids < self.ex_hi[j.clip(0)]))
        return ok

    def train_rows(self) -> np.ndarray:
        if self._train_rows is None:
            pos = {f: i for i, f in enumerate(self.files)}
            rows = np.concatenate([np.arange(self.starts[pos[f]], self.starts[pos[f]] + self.sizes[pos[f]],
                                             dtype=np.int64) for f in self.train_files])
            self._train_rows = rows[self.is_train(rows)]
        return self._train_rows

    def view(self, rows: np.ndarray) -> Dataset:
        return _ShardRows(self.shard_dir, self.files, self.starts, rows)

    def loaders(self, batch_size: int, num_workers: int):
        offsets = dict(zip(self.files, self.starts))
        train = CVRPShardStreamDataset(self.shard_dir, self.train_files, offsets, self.exclude,
                                       shuffle=True, seed=self.seed)
        val = CVRPShardStreamDataset(self.shard_dir, self.val_files, offsets, None, shuffle=False, seed=self.seed)

        def set_epoch(e: int):
            # streaming datasets reshuffle shard + row order per epoch
            train.set_epoch(e)
            val.set_epoch(e)
        return (_loader(train, batch_size, num_workers),
                _loader(val, batch_size, max(1, num_workers // 2)), set_epoch)


class _FileData:
    """Single .pt file: instance-level split, map-style loaders. Rows are grouped by the original trainer's key
    int(sum(locs) * 1e6), identical on all steps of an instance; this key is not the `_inst` hash."""

    def __init__(self, path: str, seed: int, val_ratio: float, split: Optional[Dict]):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict) or "demand" not in data or "load" not in data:
            raise ValueError(
                f"Expected CVRP teacher data with 'demand' and 'load' keys; "
                f"got {list(data.keys()) if isinstance(data, dict) else type(data)}"
            )
        self.t = {
            "locs":           data["locs"].float(),
            "demand":         data["demand"].float(),
            "mask":           data["mask"].float(),
            "teacher_probs":  data["teacher_probs"].float(),
            "teacher_logits": data["teacher_logits"].float(),
            "current_node":   data["current_node"].long().view(-1),
            "first_node":     data["first_node"].long().view(-1),
            "visited":        data["visited"].float(),
            "load":           data["load"].float().view(-1),
            "step_i":         data["step_i"].float().view(-1),
        }
        del data
        self.seed = seed
        n = self.t["locs"].size(0)
        self.problem_size = int(self.t["locs"].size(1)) - 1
        # the original trainer's instance groups: coordinate sum of the row (identical on all steps of an instance)
        flat = self.t["locs"].reshape(n, -1)
        groups = [int(flat[i].sum().item() * 1e6) for i in range(n)]
        if split is None:
            unique_groups = sorted(set(groups))
            random.Random(seed).shuffle(unique_groups)
            n_val_groups = max(1, int(len(unique_groups) * val_ratio))
            split = {"kind": "instance", "seed": seed, "val_ratio": val_ratio, "n_rows": n,
                     "val_groups": sorted(unique_groups[:n_val_groups])}
        elif split.get("kind") != "instance" or int(split["n_rows"]) != n:
            raise ValueError(f"split does not match the rows of {path}")
        self.split = split
        val_groups = set(split["val_groups"])
        self.train_mask = np.array([g not in val_groups for g in groups], dtype=bool)
        self._train_rows = np.flatnonzero(self.train_mask).astype(np.int64)
        self.val_rows = np.flatnonzero(~self.train_mask).astype(np.int64)
        print(f"[pomo_cvrp] instance-level split: {len(set(groups))} groups -> {len(val_groups)} val | "
              f"rows train={len(self._train_rows):,}, val={len(self.val_rows):,}")

    def is_train(self, ids: np.ndarray) -> np.ndarray:
        ok = (ids >= 0) & (ids < self.train_mask.size)
        return ok & self.train_mask[ids.clip(0, self.train_mask.size - 1)]

    def train_rows(self) -> np.ndarray:
        return self._train_rows

    def view(self, rows: np.ndarray) -> Dataset:
        return _FileRows(self.t, rows)

    def loaders(self, batch_size: int, num_workers: int):
        sampler = _Permutation(len(self._train_rows), self.seed)
        return (_loader(self.view(self._train_rows), batch_size, num_workers, sampler, persistent=True),
                _loader(self.view(self.val_rows), batch_size, num_workers, persistent=True),
                sampler.set_epoch)


def _check_generated(path: str, problem_size: int) -> None:
    """Format check of a generated sharded directory (manifest, every shard's size, fields of the first shard)."""
    man = torch.load(os.path.join(path, "manifest.pt"), weights_only=False)
    if man.get("format") != "sharded_v1" or not man.get("shard_files"):
        raise RuntimeError(f"{path}: unexpected manifest {sorted(man)}")
    sizes = [int(_load_shard(os.path.join(path, f), mmap=True)["teacher_probs"].size(0)) for f in man["shard_files"]]
    if sum(sizes) != int(man["total_steps"]):
        raise RuntimeError(f"{path}: shards hold {sum(sizes)} steps, manifest says {man['total_steps']}")
    s = _load_shard(os.path.join(path, man["shard_files"][0]), mmap=True)
    n, A = sizes[0], problem_size + 1
    shapes = {"teacher_logits": (n, A), "teacher_probs": (n, A), "mask": (n, A), "locs": (n, A, 2),
              "demand": (n, A), "current_node": (n,), "first_node": (n,), "visited": (n, A), "load": (n,),
              "step_i": (n,)}
    for k, shp in shapes.items():
        if k not in s or tuple(s[k].shape) != shp:
            raise RuntimeError(f"{path}: field {k} has shape {tuple(s[k].shape) if k in s else None}, expected {shp}")
    m = min(n, 4096)
    mask, tp = s["mask"][:m].bool(), s["teacher_probs"][:m]
    if not mask.any(-1).all() or (tp.sum(-1) - 1).abs().max() > 1e-4 or tp.masked_fill(mask, 0).abs().max() > 1e-6:
        raise RuntimeError(f"{path}: teacher probabilities / feasibility masks are malformed")
    if (s["demand"][:m, 0] != 0).any() or not ((s["load"][:m] >= -1e-4) & (s["load"][:m] <= 1 + 1e-4)).all():
        raise RuntimeError(f"{path}: depot demand / remaining load out of range")
    print(f"[pomo_cvrp] generated {len(sizes)} shards, {sum(sizes):,} decision steps, N={problem_size} -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Program pre-flight (CPU): synthetic batch with every node index + geometric edge cases
# ─────────────────────────────────────────────────────────────────────────────

def _preflight_edge_batch(n_customers: int = 20):
    """Geometric edge cases for _preflight_cvrp_heuristic. Around an anchor (the depot, or the current customer 1),
    customers lie on the rays at angle 0 and pi, one / two / four float32 ulps below and one above angle 0 and pi
    (an angle in [0, 2 pi) built as atan2 + 2 pi then rounds to exactly 2 pi, i.e. one bin past the last), on the
    vertical rays and at the anchor itself; plus a duplicate pair of customers and coordinates exactly 0 and 1.
    Rows vary the depot position, current_node (0, 1, N), first_node (1, N) and the load."""
    N1 = n_customers + 1
    g = torch.Generator().manual_seed(1)
    r = 0.2

    def ulps(y, k):
        t = torch.tensor(float(y))
        for _ in range(abs(k)):
            t = torch.nextafter(t, torch.tensor(-1.0 if k < 0 else 2.0))
        return float(t)

    rows = []
    for depot in ((0.5, 0.5), (0.3, 0.7)):
        for anchor_is_depot in (True, False):
            locs = torch.rand(N1, 2, generator=g) * 0.3 + 0.35
            locs[0] = torch.tensor(depot)
            if not anchor_is_depot:
                locs[1] = torch.tensor((0.45, 0.6))
            ax, ay = (float(v) for v in locs[0 if anchor_is_depot else 1])
            pts = [(ax + r, ay), (ax + r, ulps(ay, -1)), (ax + r, ulps(ay, -2)), (ax + r, ulps(ay, -4)),
                   (ax + r, ulps(ay, 1)), (ax - r, ay), (ax - r, ulps(ay, -1)), (ax - r, ulps(ay, 1)),
                   (ax, ay + r), (ax, ay - r), (ax, ay), (1.0, ay), (0.0, 0.0), (1.0, 1.0)]
            for j, (x, y) in enumerate(pts):
                locs[2 + j] = torch.tensor((x, y))
            locs[2 + len(pts) + 1] = locs[2 + len(pts)]                     # duplicate pair
            for cur in (0, 1, n_customers):
                for first, full in ((1, True), (n_customers, False)):
                    rows.append((locs.clone(), cur, first, full))
    B = len(rows)
    locs = torch.stack([x[0] for x in rows])
    cur = torch.tensor([x[1] for x in rows])
    first = torch.tensor([x[2] for x in rows])
    demand = torch.cat([torch.zeros(B, 1), torch.randint(1, 10, (B, n_customers), generator=g).float() / 40], dim=-1)
    load = torch.tensor([1.0 if x[3] else 0.3 for x in rows])
    vis = torch.zeros(B, N1)
    vis[torch.arange(B), cur] = (cur > 0).float()
    vis[torch.arange(B), first] = 1.0
    mask = (vis == 0) & (demand <= load[:, None])
    mask[:, 0] = cur != 0
    mask[~mask.any(-1), 0] = True
    return locs, demand, cur, mask, vis, load, first


def _preflight_cvrp_heuristic(fn: Callable, n_customers: int = 20, seed: int = 0) -> None:
    """CPU pre-flight of a heuristic on a batch that covers every node index: current_node 0..N (depot and each
    customer), first_node 1..N, random visited sets / feasibility masks (at least one feasible node) / loads. Raises on
    an exception, a wrong output shape or a non-finite value on a feasible node (e.g. an off-by-one index that only
    fails at the last node, which on the GPU is an uncatchable device-side assert). A second batch of geometric edge
    cases (_preflight_edge_batch) must run without an exception and return the right shape (e.g. a polar bin index
    that reaches the bin count when an angle rounds to exactly 2 pi); its outputs may be non-finite, which
    program_probs handles at run time."""
    locs_e, demand_e, cur_e, mask_e, vis_e, load_e, first_e = _preflight_edge_batch(n_customers)
    out_e = fn(locs_e, demand_e, cur_e, mask_e, vis_e, load_e, first_e)
    if tuple(out_e.shape) != tuple(mask_e.shape):
        raise ValueError(f"pre-flight (edge cases): expected {tuple(mask_e.shape)}, got {tuple(out_e.shape)}")
    g = torch.Generator().manual_seed(seed)
    N1 = n_customers + 1
    B = 2 * N1
    locs = torch.rand(B, N1, 2, generator=g)
    demand = torch.cat([torch.zeros(B, 1), torch.randint(1, 10, (B, n_customers), generator=g).float() / 40], dim=-1)
    cur = torch.arange(B) % N1
    first = 1 + torch.arange(B).flip(0) % n_customers
    vis = (torch.rand(B, N1, generator=g) < 0.5).float()
    vis[:, 0] = 0
    vis[torch.arange(B), cur] = (cur > 0).float()
    vis[torch.arange(B), first] = 1.0
    load = torch.rand(B, generator=g)
    load[: B // 4] = 1.0
    mask = (vis == 0) & (demand <= load[:, None])
    mask[:, 0] = cur != 0
    none = ~mask.any(-1)
    mask[none, 0] = True
    out = fn(locs, demand, cur, mask, vis, load, first)
    if tuple(out.shape) != (B, N1):
        raise ValueError(f"pre-flight: expected ({B},{N1}), got {tuple(out.shape)}")
    if not torch.isfinite(out.float()[mask]).all():
        raise ValueError("pre-flight: non-finite output on a feasible node")


# ─────────────────────────────────────────────────────────────────────────────
# Router: POMO encoder + load-aware decoder query + dynamic K
# ─────────────────────────────────────────────────────────────────────────────

class CVRPHeuristicAttentionStudent(nn.Module):
    """
    Encoder:
      Linear(2 → D)                        for depot
      Linear(3 → D) over [x, y, demand]    for customer
      + N_layers MHA stack (POMO style)

    Each input node gets an extra `is_depot` flag and a `visited` flag concatenated
    *before* embedding, but to keep it close to POMO we use separate projections
    and add visited / capacity signals in a load-aware decoder query instead.

    Decoder query (VRPContext-style):
      q = project_context([h_current ‖ load_scalar])          [B, D]
        + graph_context = mean(h, dim=1) projected            [B, D]

    Routing (dynamic):
      K_m = V_m @ h            [B, M, D]
      scores = K · q / τ       [B, M]
      w = softmax(scores)
      ŷ = Σ w_m · V_m
    """

    def __init__(
        self,
        num_heuristics: int,
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        routing_temperature: float = 1.0,
    ):
        super().__init__()
        from rl4co.models.nn.graph.attnnet import GraphAttentionNetwork   # not needed by the data workers
        self.embed_dim = embed_dim
        self.num_heuristics = num_heuristics
        self.routing_temperature = routing_temperature

        self.embed_depot = nn.Linear(2, embed_dim)           # depot: (x, y) — depot is always re-visitable
        self.embed_customer = nn.Linear(4, embed_dim)        # customer: (x, y, demand, visited)
        self.encoder = GraphAttentionNetwork(
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_layers=num_encoder_layers,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
        )

        # Decoder context: [current_node_emb ‖ load_scalar] → embed_dim
        self.project_context = nn.Linear(embed_dim + 1, embed_dim, bias=False)
        self.project_graph = nn.Linear(embed_dim, embed_dim, bias=False)

    def encode(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        locs    = batch["locs"]                      # [B, N+1, 2]
        demand  = batch["demand"]                    # [B, N+1]
        visited = batch["visited"]                   # [B, N+1]

        depot_emb = self.embed_depot(locs[:, :1, :])                       # [B, 1, D]
        customer_inp = torch.cat([
            locs[:, 1:, :],
            demand[:, 1:, None],
            visited[:, 1:, None],
        ], dim=-1)                                                         # [B, N, 4]
        customer_emb = self.embed_customer(customer_inp)                   # [B, N, D]
        h_init = torch.cat([depot_emb, customer_emb], dim=1)               # [B, N+1, D]

        h = self.encoder(h_init)                                           # [B, N+1, D]
        graph_ctx = self.project_graph(h.mean(dim=1))                      # [B, D]
        return h, graph_ctx

    def compute_query(self, h: torch.Tensor, graph_ctx: torch.Tensor, batch: Dict) -> torch.Tensor:
        B = h.size(0)
        batch_idx = torch.arange(B, device=h.device)
        cur = batch["current_node"].long().view(B)
        load = batch["load"].float().view(B)

        cur_emb = h[batch_idx, cur]                                        # [B, D]
        ctx_in = torch.cat([cur_emb, load[:, None]], dim=-1)               # [B, D+1]
        step_ctx = self.project_context(ctx_in)                            # [B, D]
        return step_ctx + graph_ctx                                        # [B, D]

    def forward(self, batch: Dict, V: torch.Tensor, return_scores: bool = False,
                V_out: torch.Tensor = None):
        # V drives the routing key K_m = V_m @ h; V_out is mixed into the output
        # y_hat = Σ w_m·V_out_m. They default to the same tensor (training path);
        # passing them separately lets eval decouple K-one-hot from V-one-hot.
        if V_out is None:
            V_out = V
        h, graph_ctx = self.encode(batch)
        q = self.compute_query(h, graph_ctx, batch)                        # [B, D]

        # K_m = V_m @ h  →  V_m-weighted node embedding
        K = torch.bmm(V.detach(), h)                                       # [B, M, D]
        scores = torch.bmm(K, q.unsqueeze(-1)).squeeze(-1) / self.routing_temperature
        w = F.softmax(scores, dim=-1)                                      # [B, M]
        y_hat = (w.unsqueeze(-1) * V_out).sum(dim=1)                       # [B, N+1]

        if return_scores:
            return y_hat, w, scores
        return y_hat, w


# ─────────────────────────────────────────────────────────────────────────────
# Prompt cases: per-decision-step features (geometry, capacity, depot choice)
# ─────────────────────────────────────────────────────────────────────────────

def _theta_deg(dx: float, dy: float) -> float:
    """Angle from positive x-axis (east) in degrees, range [-180, 180].

    For depot-relative polar coords (θ from depot), pass dx = cand.x - depot.x,
    dy = cand.y - depot.y. Useful for detecting route-arm coherence.
    """
    return round(math.degrees(math.atan2(dy, dx)), 1)


def _node_record(i: int, locs: torch.Tensor, demand: torch.Tensor,
                 cur_pos: Tuple, depot_pos: Tuple, dist_cur_to_depot: float) -> Dict:
    """Build a single (node, pos, dcur, ddepot, theta_from_depot, demand, cw_savings) record."""
    pos = (round(float(locs[i, 0]), 3), round(float(locs[i, 1]), 3))
    d_cur = float(((locs[i, 0] - cur_pos[0]) ** 2
                   + (locs[i, 1] - cur_pos[1]) ** 2) ** 0.5)
    d_dep = float(((locs[i, 0] - depot_pos[0]) ** 2
                   + (locs[i, 1] - depot_pos[1]) ** 2) ** 0.5)
    theta = _theta_deg(float(locs[i, 0]) - depot_pos[0],
                       float(locs[i, 1]) - depot_pos[1])
    # Clarke-Wright savings: dist(cur,depot) + dist(node,depot) - dist(cur,node)
    # HIGH savings = cur and node are both far from depot but close to each other,
    # so consolidating into one trip saves distance.
    savings = dist_cur_to_depot + d_dep - d_cur
    return {
        "node":             int(i),
        "pos":              pos,
        "dist_to_cur":      round(d_cur, 3),
        "dist_to_depot":    round(d_dep, 3),
        "theta_from_depot": theta,
        "demand":           round(float(demand[i]), 3),
        "cw_savings":       round(savings, 3),
    }


def _topk_info_with_demand(probs: torch.Tensor, locs: torch.Tensor, demand: torch.Tensor, cur_pos: Tuple,
                           depot_pos: Tuple, dist_cur_to_depot: float, mask: torch.Tensor, k: int = 2) -> List[Dict]:
    """Top-k feasible nodes by probability, with depot- and savings-aware fields."""
    probs_m = probs.masked_fill(~mask, float("-inf"))
    result = []
    for idx in probs_m.argsort(descending=True):
        if not mask[idx]:
            continue
        result.append(_node_record(int(idx), locs, demand, cur_pos, depot_pos, dist_cur_to_depot))
        if len(result) >= k:
            break
    return result


def _nearest_feasible(locs: torch.Tensor, demand: torch.Tensor, mask: torch.Tensor, cur_pos: Tuple,
                      depot_pos: Tuple, dist_cur_to_depot: float, n: int = 10) -> List[Dict]:
    result = []
    for i in range(locs.size(0)):
        if not mask[i]:
            continue
        result.append(_node_record(i, locs, demand, cur_pos, depot_pos, dist_cur_to_depot))
    result.sort(key=lambda x: x["dist_to_cur"])
    return result[:n]


def _sample_info(row: Dict, V_m: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> Dict:
    """Features of one decision step: state, depot-anchored polar coordinates, remaining work, the nearest feasible
    candidates, the top-2 of V_m and of the teacher, and the depot-choice flags."""
    cur_idx   = int(row["current_node"].item())
    first_idx = int(row["first_node"].item()) if "first_node" in row else 0
    load      = float(row["load"].item())
    mask_b    = mask.bool().cpu()
    locs_b    = row["locs"].cpu()                    # [N+1, 2]
    dem_b     = row["demand"].cpu()
    visited   = row["visited"].cpu() if "visited" in row else torch.zeros_like(mask_b)
    V_m       = V_m.detach().float().cpu()
    teacher   = teacher.detach().float().cpu()
    cur_pos   = (round(float(locs_b[cur_idx, 0]), 3),
                 round(float(locs_b[cur_idx, 1]), 3))
    depot_pos = (round(float(locs_b[0, 0]), 3),
                 round(float(locs_b[0, 1]), 3))
    first_pos = (round(float(locs_b[first_idx, 0]), 3),
                 round(float(locs_b[first_idx, 1]), 3))

    # Distances from current
    dist_to_cur = torch.norm(locs_b - locs_b[cur_idx], dim=-1)
    dist_cur_to_depot = round(float(dist_to_cur[0]), 3)

    # Depot-anchored polar coords for the key reference points
    cur_theta_from_depot   = _theta_deg(cur_pos[0]   - depot_pos[0], cur_pos[1]   - depot_pos[1])
    first_theta_from_depot = _theta_deg(first_pos[0] - depot_pos[0], first_pos[1] - depot_pos[1])

    # Unvisited customer mask (depot excluded)
    customer_mask = torch.zeros_like(mask_b)
    customer_mask[1:] = (visited[1:] < 0.5)
    n_unvisited = int(customer_mask.sum().item())

    # Feasible customers (mask is True ∧ not depot)
    customer_feas = mask_b.clone()
    customer_feas[0] = False
    n_feasible = int(customer_feas.sum().item())
    n_infeasible = max(0, n_unvisited - n_feasible)
    demand_remaining = round(float((dem_b * customer_mask.float()).sum().item()), 3)

    # Per-customer polar coords (θ from depot, r from depot) of the unvisited customers sorted by θ, plus
    # per-customer demand on a parallel array: the LLM reads the arm structure from gaps in the sorted θ sequence.
    if n_unvisited > 0:
        unv_locs = locs_b[customer_mask]                              # [K, 2]
        unv_dem  = dem_b[customer_mask]                               # [K]
        centroid = unv_locs.mean(dim=0)
        dist_cur_to_centroid = round(float(((centroid[0] - cur_pos[0]) ** 2
                                            + (centroid[1] - cur_pos[1]) ** 2) ** 0.5), 3)
        depot_to_centroid    = round(float(((centroid[0] - locs_b[0, 0]) ** 2
                                            + (centroid[1] - locs_b[0, 1]) ** 2) ** 0.5), 3)
        rel_x = unv_locs[:, 0] - depot_pos[0]
        rel_y = unv_locs[:, 1] - depot_pos[1]
        theta_deg = torch.atan2(rel_y, rel_x) * (180.0 / math.pi)     # [K]
        r_dep     = torch.sqrt(rel_x * rel_x + rel_y * rel_y)         # [K]
        cur_dx    = unv_locs[:, 0] - cur_pos[0]
        cur_dy    = unv_locs[:, 1] - cur_pos[1]
        d_cur     = torch.sqrt(cur_dx * cur_dx + cur_dy * cur_dy)     # [K]
        order = torch.argsort(theta_deg)
        unvisited_theta_sorted = [round(float(theta_deg[i]), 1) for i in order]
        unvisited_r_sorted     = [round(float(r_dep[i]),     3) for i in order]
        unvisited_dcur_sorted  = [round(float(d_cur[i]),     3) for i in order]
        unvisited_dem_sorted   = [round(float(unv_dem[i]),   3) for i in order]
    else:
        dist_cur_to_centroid = 0.0
        depot_to_centroid = 0.0
        unvisited_theta_sorted = []
        unvisited_r_sorted     = []
        unvisited_dcur_sorted  = []
        unvisited_dem_sorted   = []

    nearest_feasible = _nearest_feasible(locs_b, dem_b, mask_b, cur_pos, depot_pos, dist_cur_to_depot, n=10)
    heuristic_top2 = _topk_info_with_demand(V_m, locs_b, dem_b, cur_pos, depot_pos, dist_cur_to_depot, mask_b)
    teacher_top2 = _topk_info_with_demand(teacher, locs_b, dem_b, cur_pos, depot_pos, dist_cur_to_depot, mask_b)

    # Depot-choice flags: forced = no customer was feasible (mask only allowed the depot);
    # proactive = at least one customer was feasible but the teacher picked the depot.
    teacher_chose_depot   = bool(teacher_top2) and teacher_top2[0]["node"] == 0
    heuristic_chose_depot = bool(heuristic_top2) and heuristic_top2[0]["node"] == 0
    teacher_depot_mode = ("forced" if n_feasible == 0 else "proactive") if teacher_chose_depot else None

    return {
        "cur_pos":                cur_pos,
        "current_load":           round(load, 3),
        "at_depot":               (cur_idx == 0),
        "first_pos":              first_pos,
        "depot_pos":              depot_pos,
        "dist_cur_to_depot":      dist_cur_to_depot,
        "cur_theta_from_depot":   cur_theta_from_depot,
        "first_theta_from_depot": first_theta_from_depot,
        "n_unvisited":            n_unvisited,
        "infeasible_count":       n_infeasible,
        "demand_remaining":       demand_remaining,
        "dist_cur_to_centroid":   dist_cur_to_centroid,
        "depot_to_centroid":      depot_to_centroid,
        "unvisited_theta_sorted": unvisited_theta_sorted,
        "unvisited_r_sorted":     unvisited_r_sorted,
        "unvisited_dcur_sorted":  unvisited_dcur_sorted,
        "unvisited_dem_sorted":   unvisited_dem_sorted,
        "teacher_chose_depot":    teacher_chose_depot,
        "teacher_depot_mode":     teacher_depot_mode,      # "forced" | "proactive" | None
        "heuristic_chose_depot":  heuristic_chose_depot,
        "nearest_feasible":       nearest_feasible,
        "heuristic_top2":         heuristic_top2,
        "teacher_top2":           teacher_top2,
    }


def _fmt_node(x: Dict) -> str:
    return (f"node{x['node']} pos={x['pos']} "
            f"dcur={x['dist_to_cur']} ddepot={x.get('dist_to_depot', '?')} "
            f"θ={x.get('theta_from_depot', '?')}° "
            f"dem={x.get('demand', '?')} sav={x.get('cw_savings', '?')}")


def _list_str(xs: List) -> str:
    return ", ".join(str(x) for x in xs) if xs else "(none)"


def _code_case(si: Dict) -> str:
    """Case of the code backward prompt, from STATE to the depot-choice block (lines keep the indentation they have
    in the rendered prompt; the first line is indented by the template)."""
    nf_lines = [_fmt_node(x) for x in si["nearest_feasible"][:_N_CANDIDATES]]
    h_lines = [_fmt_node(x) for x in si["heuristic_top2"]]
    t_lines = [_fmt_node(x) for x in si["teacher_top2"]]
    t_depot, h_depot = si["teacher_chose_depot"], si["heuristic_chose_depot"]
    t_label = (f"DEPOT({si['teacher_depot_mode']})" if t_depot
               else f"customer (node {si['teacher_top2'][0]['node']})" if si["teacher_top2"] else "?")
    h_label = ("DEPOT" if h_depot
               else f"customer (node {si['heuristic_top2'][0]['node']})" if si["heuristic_top2"] else "?")
    depot_choice_block = f"DEPOT CHOICE: teacher={t_label}  heuristic={h_label}"
    if t_depot:
        depot_choice_block += textwrap.dedent(f"""

    NOTE — TEACHER chose DEPOT ({si['teacher_depot_mode']}):
      • "forced":    no customer was feasible (load too low for any unvisited customer).
                     Heuristic SHOULD also output depot here; if not, sharpen the
                     depot logit when n_feasible_customers == 0.
      • "proactive": there were feasible customers but teacher returned anyway.
                     Diagnose why depot beat the available customers — common reasons:
                       - load too low to serve nearby customers efficiently
                       - remaining unvisited cluster is closer to depot than to current
                       - current arm is essentially exhausted, pivot needed
        """).rstrip()
    return "\n".join([
        "STATE:",
        f"      cur_pos={si['cur_pos']}  load={si['current_load']}  at_depot={si['at_depot']}",
        f"      first_pos={si['first_pos']}  depot_pos={si['depot_pos']}  "
        f"dist_cur_to_depot={si['dist_cur_to_depot']}",
        "",
        "    DEPOT-ANCHORED POLAR (θ from depot, range [-180°, 180°]):",
        f"      cur_θ={si['cur_theta_from_depot']}°    first_θ={si['first_theta_from_depot']}°",
        "      (θ measures route-arm direction from depot. cur_θ vs first_θ tells whether",
        "       the route is still in its starting arm.)",
        "",
        "    REMAINING WORK:",
        f"      {si['n_unvisited']} unvisited ({si['infeasible_count']} capacity-locked)  "
        f"total demand={si['demand_remaining']}",
        f"      centroid: d(cur→bulk)={si['dist_cur_to_centroid']}  d(depot→bulk)={si['depot_to_centroid']}",
        "      unvisited (sorted by θ from depot; r = depot dist, dcur = current dist):",
        f"        θ°:   {_list_str(si['unvisited_theta_sorted'])}",
        f"        r:    {_list_str(si['unvisited_r_sorted'])}",
        f"        dcur: {_list_str(si['unvisited_dcur_sorted'])}",
        f"        dem:  {_list_str(si['unvisited_dem_sorted'])}",
        "",
        f"    CANDIDATES ({_N_CANDIDATES} nearest feasible, sorted by dcur):",
        "      " + "\n".join("  " + l for l in nf_lines),
        "",
        "    HEURISTIC top-2:",
        "      " + "\n".join("  " + l for l in h_lines),
        "    TEACHER top-2:",
        "      " + "\n".join("  " + l for l in t_lines),
        "",
        "    " + depot_choice_block,
    ])


def _desc_case(si: Dict) -> str:
    """Case of the description backward prompt, from cur_pos to the teacher's top-2."""
    h_lines = [_fmt_node(x) for x in si["heuristic_top2"]]
    t_lines = [_fmt_node(x) for x in si["teacher_top2"]]
    t_label = f"DEPOT({si['teacher_depot_mode']})" if si["teacher_chose_depot"] else "customer"
    h_label = "DEPOT" if si["heuristic_chose_depot"] else "customer"
    return "\n".join([
        f"cur_pos={si['cur_pos']}  load={si['current_load']}  at_depot={si['at_depot']}",
        f"      depot_pos={si['depot_pos']}  dist_cur_to_depot={si['dist_cur_to_depot']}",
        f"      polar from depot: cur_θ={si['cur_theta_from_depot']}°  first_θ={si['first_theta_from_depot']}°",
        f"      unvisited (n={si['n_unvisited']}, {si['infeasible_count']} capacity-locked; sorted by θ):",
        f"        θ°:   {_list_str(si['unvisited_theta_sorted'])}",
        f"        dcur: {_list_str(si['unvisited_dcur_sorted'])}",
        f"        dem:  {_list_str(si['unvisited_dem_sorted'])}",
        f"      DEPOT CHOICE: teacher={t_label}  heuristic={h_label}",
        "",
        "      Heuristic top-2:",
        "        " + "\n".join("    " + l for l in h_lines),
        "      Teacher   top-2:",
        "        " + "\n".join("    " + l for l in t_lines),
    ])


def _fault_info(row: Dict, V_bank: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> Dict:
    """Features of one bank fault: the step's features with the uniform bank mixture's top-2 in place of a
    heuristic's, and the smallest per-program KL."""
    V_bank = V_bank.detach().float().cpu()
    teacher = teacher.detach().float().cpu()
    mask_b = mask.bool().cpu()
    M = V_bank.size(0)
    info = _sample_info(row, V_bank[0], teacher, mask_b)
    info.pop("heuristic_top2", None)
    info.pop("heuristic_chose_depot", None)
    w = torch.full((1, M), 1.0 / M)
    y_hat = (w.unsqueeze(-1) * V_bank.unsqueeze(0)).sum(dim=1)[0]                 # [N+1]
    bank_top2 = _topk_info_with_demand(y_hat, row["locs"].cpu(), row["demand"].cpu(), info["cur_pos"],
                                       info["depot_pos"], info["dist_cur_to_depot"], mask_b, k=2)
    info["bank_combined_top2"] = bank_top2
    info["bank_chose_depot"] = (bank_top2[0]["node"] == 0) if bank_top2 else False
    info["portfolio_min_kl"] = float(_kl_mean(teacher.unsqueeze(0).expand(M, -1), V_bank,
                                              mask_b.unsqueeze(0).expand(M, -1)).min())
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Setting
# ─────────────────────────────────────────────────────────────────────────────

@register
class PomoCVRP(Setting):
    """POMO teacher, CVRP (sizes 50 and 100 in the paper). See the module docstring."""
    name = "pomo_cvrp"
    problem = "cvrp"
    # from the training of both released students: router 128-d / 6 layers / 8 heads,
    # heuristic temperature 0.1, lr 1e-4, bfloat16 autocast, no lr schedule, 5% val shards; data generation with
    # 20 POMO starts per instance and 512 instances per generator batch
    defaults = {
        "data": {"val_ratio": 0.05},
        "router": {"embed_dim": 128, "num_layers": 6, "num_heads": 8},
        "bank": {"tau_h": 0.1},
        "train": {"lr": 1e-4, "amp": True, "scheduler": "none"},
        "generate": {"problem_size": None, "pomo_size": 20, "batch_size": 512},
        "prompts": {"teacher": _TEACHER},
    }

    def __init__(self):
        self._problem_size: Optional[int] = None

    # ---- data ---------------------------------------------------------------------------------------------------
    def build_data(self, cfg, split: Optional[Dict] = None) -> Data:
        path = _cfg(cfg, "data.path")
        seed = int(_cfg(cfg, "run.seed"))
        val_ratio = float(_cfg(cfg, "data.val_ratio"))
        num_workers = int(_cfg(cfg, "data.num_workers", 0))
        batch_size = int(_cfg(cfg, "train.batch_size"))
        src = (_ShardedData if os.path.isdir(path) else _FileData)(path, seed, val_ratio, split)
        self._problem_size = src.problem_size
        train, val, set_epoch = src.loaders(batch_size, num_workers)

        def scan(scan_seed: int):
            # the fault scan of an add round: a seeded permutation of the train rows (map-style view)
            rows = src.train_rows()
            perm = torch.randperm(len(rows), generator=torch.Generator().manual_seed(int(scan_seed))).numpy()
            return _loader(src.view(rows[perm]), batch_size, num_workers)

        def subset(row_ids: torch.Tensor, bs: int, shuffle: bool, sub_seed: int):
            ids = torch.as_tensor(row_ids).detach().cpu().long().view(-1).numpy()
            if not src.is_train(ids).all():
                raise ValueError(f"subset: {int((~src.is_train(ids)).sum())} row ids are not train rows")
            sampler = _Permutation(len(ids), int(sub_seed), per_epoch=False) if shuffle else None
            return _loader(src.view(ids), bs, num_workers, sampler)

        return Data(train=train, val=val, problem_size=src.problem_size, scan=scan, subset=subset,
                    set_epoch=set_epoch, split=src.split)

    def row_ids(self, batch: Dict) -> torch.Tensor:
        return batch["_row"]

    def instance_ids(self, batch: Dict) -> torch.Tensor:
        if "_inst" in batch:
            return batch["_inst"]
        return _instance_ids(batch["locs"], batch["demand"]).to(batch["locs"].device)

    def target(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        return batch["teacher_probs"], batch["mask"].bool()

    # ---- programs -----------------------------------------------------------------------------------------------
    def program_args(self, batch: Dict) -> tuple:
        mask = batch["mask"].bool()
        B = mask.size(0)
        return (batch["locs"], batch["demand"], batch["current_node"].long().view(B), mask,
                batch["visited"], batch["load"].float().view(B), batch["first_node"].long().view(B))

    def program_probs(self, fn: Callable, batch: Dict, tau_h: float
                      ) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        args = self.program_args(batch)
        mask = args[3]
        B, N1 = mask.shape
        ok = True
        try:
            logits = fn(*args).float()
            if logits.shape != (B, N1):
                raise ValueError(f"shape {logits.shape} != ({B},{N1})")
            if not torch.isfinite(logits).all():
                raise ValueError("non-finite logits")
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            warnings.warn(f"program runtime error: {e}. Using fallback.")
            logits = torch.zeros(B, N1, device=mask.device)
            ok = False
        logits = logits / tau_h
        probs = masked_softmax(logits.masked_fill(~mask, float("-inf")), mask)
        return probs, None, ok

    def check_program(self, fn: Callable) -> None:
        try:
            # sanity check with a tiny CVRP-like batch (explicit generator: the global RNG is left untouched)
            g = torch.Generator().manual_seed(0)
            B, N1 = 2, 5
            out = fn(torch.rand(B, N1, 2, generator=g),
                     torch.cat([torch.zeros(B, 1), torch.rand(B, N1 - 1, generator=g) * 0.3], dim=-1),
                     torch.zeros(B, dtype=torch.long), torch.ones(B, N1, dtype=torch.bool), torch.zeros(B, N1),
                     torch.ones(B), torch.ones(B, dtype=torch.long))
            if out.shape != (B, N1):
                raise ValueError(f"Expected ({B},{N1}), got {tuple(out.shape)}")
            if not torch.isfinite(out).all():
                raise ValueError("non-finite output in sanity check")
            _preflight_cvrp_heuristic(fn)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"{type(e).__name__}: {e}") from e

    # ---- router -------------------------------------------------------------------------------------------------
    def build_router(self, cfg, num_programs: int) -> nn.Module:
        return CVRPHeuristicAttentionStudent(
            num_heuristics=num_programs,
            embed_dim=int(_cfg(cfg, "router.embed_dim")),
            num_encoder_layers=int(_cfg(cfg, "router.num_layers")),
            num_heads=int(_cfg(cfg, "router.num_heads")),
            feedforward_hidden=_FEEDFORWARD_HIDDEN,
            routing_temperature=_ROUTING_TEMPERATURE,
        )

    def router_forward(self, model: nn.Module, batch: Dict, V_full: torch.Tensor, V_K: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return model(batch, V_full, return_scores=True)

    def ckpt_args(self, cfg, num_programs: int) -> Dict:
        # the fields the original evaluator's student loader reads (problem_size only names the size in
        # the bank's prompts; it is known once build_data has run)
        args = {
            "embed_dim":             int(_cfg(cfg, "router.embed_dim")),
            "num_encoder_layers":    int(_cfg(cfg, "router.num_layers")),
            "num_heads":             int(_cfg(cfg, "router.num_heads")),
            "feedforward_hidden":    _FEEDFORWARD_HIDDEN,
            "routing_temperature":   _ROUTING_TEMPERATURE,
            "heuristic_temperature": float(_cfg(cfg, "bank.tau_h")),
        }
        if self._problem_size is not None:
            args["problem_size"] = self._problem_size
        return args

    # ---- prompts ------------------------------------------------------------------------------------------------
    def prompt_vars(self, cfg, problem_size: int) -> Dict[str, str]:
        return {
            "teacher":      str(_cfg(cfg, "prompts.teacher", _TEACHER)),
            "problem_size": str(int(problem_size)),
            "n_plus_1":     str(int(problem_size) + 1),
            "tau_h":        f"{float(_cfg(cfg, 'bank.tau_h')):g}",
        }

    def failure_case(self, row: Dict, V_m: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor,
                     kind: str) -> str:
        """`${case}` of backward_code_user (kind "code": STATE ... depot-choice block) or backward_desc_user (kind
        "desc": cur_pos ... teacher top-2)."""
        si = _sample_info(row, V_m, teacher, mask)
        if kind == "code":
            return _code_case(si)
        if kind == "desc":
            return _desc_case(si)
        raise ValueError(f"failure_case kind must be 'code' or 'desc', got {kind!r}")

    def fault_case(self, row: Dict, V_bank: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> str:
        """One case of the cluster prompt without its leading "Case <i> ": "[portfolio_min_kl=...]:" and the state,
        polar, remaining-work, bank-mixture top-2 (the unweighted mean of the programs' V, no router weights; labelled
        "BANK output" in the prompt) and teacher top-2 lines. `${cases}` of cluster_user is
        "\\n\\n".join(f"Case {i} {fault_case}") with i = 1, 2, ... (the numbers case_indices refers to)."""
        info = _fault_info(row, V_bank, teacher, mask)
        bk, tk = info["bank_combined_top2"], info["teacher_top2"]
        bk_str = " | ".join(_fmt_node(x) for x in bk) if bk else "(none)"
        tk_str = " | ".join(_fmt_node(x) for x in tk) if tk else "(none)"
        t_depot, b_depot = info["teacher_chose_depot"], info["bank_chose_depot"]
        t_label = f"DEPOT({info['teacher_depot_mode']})" if t_depot else "customer"
        b_label = "DEPOT" if b_depot else "customer"
        depot_marker = f"\n  ⚠ DEPOT CHOICE: teacher={t_label}  bank={b_label}" if (t_depot or b_depot) else ""
        return "\n".join([
            f"[portfolio_min_kl={info['portfolio_min_kl']:.4f}]:",
            f"  STATE: cur_pos={info['cur_pos']}  load={info['current_load']}  at_depot={info['at_depot']}",
            f"         first_pos={info['first_pos']}  depot_pos={info['depot_pos']}  "
            f"dist_cur_to_depot={info['dist_cur_to_depot']}",
            f"  POLAR θ from depot: cur={info['cur_theta_from_depot']}°  first={info['first_theta_from_depot']}°",
            f"  REMAINING: {info['n_unvisited']} unvisited ({info['infeasible_count']} capacity-locked)",
            f"             d(cur→bulk)={info['dist_cur_to_centroid']}  d(depot→bulk)={info['depot_to_centroid']}",
            f"             unvisited θ° (sorted): {_list_str(info['unvisited_theta_sorted'])}",
            f"  BANK output top-2:    {bk_str}",
            f"  TEACHER top-2:        {tk_str}{depot_marker}",
        ])

    def fault_summary(self, rows: List[Dict], V_banks: List[torch.Tensor], teachers: List[torch.Tensor],
                      masks: List[torch.Tensor]) -> str:
        """Depot-choice statistics of the sampled faults (spotting depot-timing failure modes)."""
        if not rows:
            return ""
        infos = [_fault_info(r, V, t, m) for r, V, t, m in zip(rows, V_banks, teachers, masks)]
        t_depot = [i["teacher_chose_depot"] for i in infos]
        b_depot = [i["bank_chose_depot"] for i in infos]
        n_proactive = sum(1 for i in infos if i["teacher_chose_depot"] and i["teacher_depot_mode"] == "proactive")
        n_forced = sum(1 for i in infos if i["teacher_chose_depot"] and i["teacher_depot_mode"] == "forced")
        return (
            f"Depot-choice stats across {len(infos)} cases:\n"
            f"  - teacher chose DEPOT (proactive): {n_proactive}\n"
            f"  - teacher chose DEPOT (forced):    {n_forced}\n"
            f"  - bank    chose DEPOT:             {sum(b_depot)}\n"
            f"  - mismatch (teacher=customer, bank=DEPOT): {sum(1 for t, b in zip(t_depot, b_depot) if b and not t)}\n"
            f"  - mismatch (teacher=DEPOT, bank=customer): {sum(1 for t, b in zip(t_depot, b_depot) if t and not b)}"
        )

    # ---- data generation ----------------------------------------------------------------------------------------
    def generate(self, cfg) -> None:
        """Run the generator python -m epb.teachers.pomo_cvrp (--no-merge) in a process of its own with the teacher
        checkpoint cfg.generate.teacher_ckpt, then check the sharded output at cfg.data.path."""
        out = _cfg(cfg, "data.path")
        problem_size = int(_cfg(cfg, "generate.problem_size"))
        if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
            # the generator appends to an existing manifest and restarts the shard numbering at 0
            raise FileExistsError(f"{out} exists and is not empty")
        from ..teachers import command
        cmd, env = command(_GENERATOR, [
            "--checkpoint", _cfg(cfg, "generate.teacher_ckpt"),
            "--problem-size", problem_size,
            "--pomo-size", int(_cfg(cfg, "generate.pomo_size")),
            "--num-instances", int(_cfg(cfg, "generate.num_instances")),
            "--batch-size", int(_cfg(cfg, "generate.batch_size")),
            "--seed", int(_cfg(cfg, "generate.seed")),
            "--no-merge", "--output", out])
        print("[pomo_cvrp] " + " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)
        _check_generated(out, problem_size)

    # ---- evaluation (epb.eval) ----------------------------------------------------------------------------------
    search_default = "ms"

    def eval_policy(self, method: str, a):
        """teacher: the POMO model of --teacher-ckpt. epb / expert / frozen: the router of --ckpt (a best_model.pt
        written by the trainers) with the bank of --bank; tau_h: flag / decode.json, else the checkpoint's
        heuristic_temperature, else the setting's bank.tau_h. The evaluator's states equal the generator's rows, so
        both state conventions decode the same way here (the option is validated, nothing else)."""
        dev = torch.device(a.device)
        if method == "teacher":
            if not a.teacher_ckpt:
                raise ValueError("--method teacher needs --teacher-ckpt")
            return EvalPolicy("teacher", _load_teacher(a.teacher_ckpt, dev), dev)
        if method in ("nam", "moet", "tree", "agent"):
            return _baseline_policy(method, a, dev)
        if method not in ("epb", "expert", "frozen"):
            raise ValueError(f"pomo_cvrp: unknown method {method!r}")
        if not a.ckpt or not a.bank:
            raise ValueError(f"--method {method} needs --ckpt (a directory with best_model.pt and bank.json, or a "
                             f".pt with --bank)")
        from ..bank import Bank
        _guard_fallbacks()
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        args = ck["args"]
        tau_h = float(eval_option(a, "tau_h", args, "heuristic_temperature", self.defaults["bank"]["tau_h"]))
        state_convention(a, args)
        bank = Bank(self, tau_h).load(a.bank)
        M = len(bank)
        if a.drop:
            bank.remove(a.drop)
        model = CVRPHeuristicAttentionStudent(
            num_heuristics=M,
            embed_dim=args["embed_dim"],
            num_encoder_layers=args["num_encoder_layers"],
            num_heads=args["num_heads"],
            feedforward_hidden=args["feedforward_hidden"],
            routing_temperature=args["routing_temperature"],
        )
        model.load_state_dict(ck["model_state_dict"])
        model.to(dev).eval()
        k_onehot, v_onehot = a.k_mode == "onehot", a.v_mode == "onehot"
        print(f"[pomo_cvrp] {method}: epoch {ck.get('epoch')}, M={len(bank)}"
              + (f" (dropped {sorted(a.drop)} of {M})" if a.drop else "")
              + f", tau_h={tau_h:g}, k {'onehot' if k_onehot else 'soft'} / v {'onehot' if v_onehot else 'soft'}",
              flush=True)
        return EvalPolicy("router", model, dev, bank=bank, k_onehot=k_onehot, v_onehot=v_onehot)

    def eval_greedy(self, policy, inst: Dict[str, torch.Tensor], a) -> torch.Tensor:
        """POMO trajectory 0 (first customer 1) of the rollout from all N first customers (the same rollout as
        the search, run once: a trajectory's floats depend on the B x N batch it is decoded in). Tree and agents:
        the greedy run of their own protocol (see _Stepper.layout)."""
        if policy.kind == "stepper":
            return _stepper_rollouts(policy, inst, a, "greedy")[0][:, 0]
        return _pomo_rollouts(policy, inst, a)[0][:, 0]

    def eval_search(self, policy, inst: Dict[str, torch.Tensor], a) -> torch.Tensor:
        if a.search != "ms":
            raise ValueError(f"pomo_cvrp: search is 'ms' (best of all N first customers), not {a.search!r}")
        if policy.kind == "stepper":
            return _stepper_rollouts(policy, inst, a, "ms")[0].min(1).values
        return _pomo_rollouts(policy, inst, a)[0].min(1).values

    def eval_top1(self, teacher, policy, inst: Dict[str, torch.Tensor], a) -> Dict[str, torch.Tensor]:
        """States: every decision of the teacher's greedy trajectory after the forced first customer, up to and
        including the final return to the depot (the first depot visit with every customer served). Primary: more
        than one feasible action under the teacher's rule (depot counted). The policy chooses at each state with its
        own feasibility rule and inputs; a hit is its argmax == the teacher's next node."""
        return _top1(teacher, policy, inst, a)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (epb.eval): POMO rollouts of the teacher / a router + bank, teacher-forced top-1
# ─────────────────────────────────────────────────────────────────────────────
# Ported from the original trainer's evaluator (teacher and student rollouts, tour cost) and the teacher-forced
# top-1 of the unified evaluation; the float32 state updates, masks and argmax tie-breaking are
# theirs, so at the same batch size per-instance costs and top-1 counts matched the released results bit for bit on
# our hardware and software (another GPU or library version can change float32 results).
# Exception: a bank program whose CUDA output is not bitwise deterministic (scatter_add with colliding indices adds
# float atomics in a run-dependent order; the expert CVRP-100 bank has one) can flip a near-tie, so a rare POMO
# trajectory, and with it an instance's search cost, can differ between runs of the same code.
# Baselines (section at the end): NAM / MoET score the flat batch of the router rollout; the tree and the agents
# pick per state of _State in _construct, with the chunking and cost convention of their own engines.

_EVAL_BATCH = 64                    # instances per rollout batch (N <= 100) ...
_EVAL_BATCH_LARGE = 1               # ... and above N = 100 (N x N trajectories per instance)
_CAPACITY_EPS = 1e-5                # demand <= load + eps is feasible (both rollouts)
# POMO-CVRP architecture (yd-kwon/POMO) of the checkpoint passed with --teacher-ckpt
_TEACHER_PARAMS = dict(embedding_dim=128, sqrt_embedding_dim=128 ** 0.5, encoder_layer_num=6, qkv_dim=16, head_num=8,
                       logit_clipping=10, ff_hidden_dim=512, eval_type="argmax")

def _teacher_model_class():
    """CVRPModel of epb.teachers.pomo_cvrp."""
    from ..teachers.pomo_cvrp import CVRPModel
    return CVRPModel


def _load_teacher(ckpt_path: str, device: torch.device) -> nn.Module:
    model = _teacher_model_class()(**_TEACHER_PARAMS).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


def _guard_fallbacks() -> None:
    """A program that raises at run time is replaced by the uniform fallback with a warning: show every such
    warning, and abort when a program runs out of GPU memory (the bank would disable it for the rest of the run,
    which changes the method being evaluated)."""
    warnings.filterwarnings("always", message=".*runtime error.*")
    warnings.filterwarnings("error", message=".*out of GPU memory.*")


class EvalPolicy:
    """The teacher (kind "teacher": a POMO CVRPModel), a router + bank (kind "router"), a scorer of the router
    rollout's flat batch (kind "flat": NAM / MoET, `score`) or a per-state policy (kind "stepper": tree / agents,
    `stepper`). Caches the POMO rollouts of the last instance set, which greedy, search and the teacher trajectories
    of top-1 share."""

    def __init__(self, kind: str, model: nn.Module, device: torch.device, bank=None, k_onehot: bool = False,
                 v_onehot: bool = False, score: Optional[Callable] = None, stepper=None):
        self.kind, self.model, self.device, self.bank = kind, model, device, bank
        self.k_onehot, self.v_onehot = k_onehot, v_onehot
        self.score, self.stepper = score, stepper
        self.cache = None                   # (key, costs [n, N], trajectory-0 sequences [n, T])
        self.runs: Dict = {}                # stepper: layout key -> (costs [n, starts], start-1 sequences [n, T])


def _eval_batch(a, N: int) -> int:
    return a.batch or (_EVAL_BATCH if N <= 100 else _EVAL_BATCH_LARGE)


def _tour_cost(all_locs: torch.Tensor, selected_seq: torch.Tensor) -> torch.Tensor:
    """all_locs [B, N+1, 2], selected_seq [B, P, T] -> [B, P] sum of the euclidean leg lengths."""
    B, P, T = selected_seq.shape
    locs_BP = all_locs[:, None, :, :].expand(B, P, all_locs.size(1), 2)
    coords = locs_BP.gather(2, selected_seq[:, :, :, None].expand(B, P, T, 2))
    return torch.norm(coords[:, :, 1:, :] - coords[:, :, :-1, :], dim=-1).sum(dim=-1)


@torch.no_grad()
def _rollout_teacher(model: nn.Module, depot_xy, node_xy, node_demand, pomo_size: int):
    """Greedy POMO rollout of the teacher from first customers 1..P -> (selected_seq [B, P, T], cost [B, P])."""
    device = depot_xy.device
    B = depot_xy.size(0)
    N = node_xy.size(1)
    num_nodes = N + 1
    P = pomo_size
    all_locs = torch.cat([depot_xy, node_xy], dim=1)                                    # [B, N+1, 2]
    all_demand = torch.cat([torch.zeros(B, 1, device=device), node_demand], dim=1)      # [B, N+1]
    model.pre_forward(depot_xy, node_xy, node_demand)

    load = torch.ones(B, P, device=device)
    visited_ninf = torch.zeros(B, P, num_nodes, device=device)
    finished = torch.zeros(B, P, dtype=torch.bool, device=device)
    # POMO step 1: each pomo starts at a different customer
    current_node = torch.arange(1, P + 1, device=device)[None, :].expand(B, P).clone()
    bidx = torch.arange(B, device=device)[:, None].expand(B, P)
    pidx = torch.arange(P, device=device)[None, :].expand(B, P)
    demand_BP = all_demand[:, None, :].expand(B, P, num_nodes)
    load -= demand_BP.gather(2, current_node[:, :, None]).squeeze(2)
    visited_ninf[bidx, pidx, current_node] = float("-inf")
    at_depot = (current_node == 0)
    visited_ninf[:, :, 0][~at_depot] = 0
    ninf_mask = visited_ninf.clone()
    ninf_mask[load[:, :, None] + _CAPACITY_EPS < demand_BP] = float("-inf")
    finished |= (visited_ninf == float("-inf")).all(dim=2)
    ninf_mask[:, :, 0][finished] = 0

    selected_seq = [torch.zeros(B, P, dtype=torch.long, device=device), current_node]   # depot, first customer
    step = 2
    max_steps = N * 3 + 5
    while not finished.all() and step < max_steps:
        sel, _, _ = model(current_node, load, ninf_mask)
        sel = torch.where(finished, torch.zeros_like(sel), sel)                         # finished pomos park at depot
        selected_seq.append(sel)
        current_node = sel
        at_depot = (sel == 0)
        load -= demand_BP.gather(2, sel[:, :, None]).squeeze(2)
        load[at_depot] = 1.0
        visited_ninf[bidx, pidx, sel] = float("-inf")
        visited_ninf[:, :, 0][~at_depot] = 0
        ninf_mask = visited_ninf.clone()
        ninf_mask[load[:, :, None] + _CAPACITY_EPS < demand_BP] = float("-inf")
        finished |= (visited_ninf == float("-inf")).all(dim=2)
        ninf_mask[:, :, 0][finished] = 0
        step += 1

    selected_seq.append(torch.zeros(B, P, dtype=torch.long, device=device))            # final depot return
    selected_seq = torch.stack(selected_seq, dim=-1)                                    # [B, P, T]
    return selected_seq, _tour_cost(all_locs, selected_seq)


def _router_feas(visited: torch.Tensor, current_node: torch.Tensor, load: torch.Tensor, demand: torch.Tensor,
                 finished: torch.Tensor) -> torch.Tensor:
    """Feasible actions of a router-bank rollout, [..., N+1] bool: unvisited customers whose demand fits the load
    (+ eps) and the depot unless the vehicle stands on it; only the depot once finished or when nothing else is.
    visited: customer visits (the depot entry is never set)."""
    feas = (visited == 0)
    feas[..., 0] = current_node != 0
    feas = feas & ((load[..., None] + _CAPACITY_EPS) >= demand)
    only_depot = torch.zeros_like(feas)
    only_depot[..., 0] = True
    feas = torch.where(finished[..., None], only_depot, feas)
    return torch.where(feas.any(dim=-1, keepdim=True), feas, only_depot)


def _router_scores(policy: EvalPolicy, flat: Dict) -> torch.Tensor:
    """y_hat [rows, N+1] of the router + bank on a flat batch. k one-hot: routing key K_m = h[argmax V_m]; v one-hot:
    y_hat mixes one-hot argmax vectors instead of the program distributions. Kind "flat": the policy's scores."""
    if policy.kind == "flat":
        return policy.score(flat)
    V = policy.bank.compute_probs(flat)[0]                                              # [rows, M, N+1]
    V_oh = None
    if policy.k_onehot or policy.v_onehot:
        V_oh = torch.zeros_like(V).scatter_(-1, V.argmax(dim=-1, keepdim=True), 1.0)
    y_hat, _ = policy.model(flat, V_oh if policy.k_onehot else V, V_out=V_oh if policy.v_onehot else V)
    return y_hat


@torch.no_grad()
def _rollout_router(policy: EvalPolicy, depot_xy, node_xy, node_demand, pomo_size: int):
    """Greedy argmax rollout of a router + bank from first customers 1..P -> (selected_seq [B, P, T], cost [B, P])."""
    device = depot_xy.device
    B = depot_xy.size(0)
    N = node_xy.size(1)
    num_nodes = N + 1
    P = pomo_size
    BP = B * P
    all_locs = torch.cat([depot_xy, node_xy], dim=1)
    all_demand = torch.cat([torch.zeros(B, 1, device=device), node_demand], dim=1)

    load = torch.ones(B, P, device=device)
    visited = torch.zeros(B, P, num_nodes, device=device)
    finished = torch.zeros(B, P, dtype=torch.bool, device=device)
    first_customer = torch.arange(1, P + 1, device=device)[None, :].expand(B, P).clone()
    current_node = first_customer.clone()
    bidx = torch.arange(B, device=device)[:, None].expand(B, P)
    pidx = torch.arange(P, device=device)[None, :].expand(B, P)
    demand_BP = all_demand[:, None, :].expand(B, P, num_nodes)
    load -= demand_BP.gather(2, current_node[:, :, None]).squeeze(2)
    visited[bidx, pidx, current_node] = 1.0
    locs_BP = all_locs[:, None, :, :].expand(B, P, num_nodes, 2)

    selected_seq = [torch.zeros(B, P, dtype=torch.long, device=device), current_node.clone()]
    step = 2
    max_steps = N * 3 + 5
    while not finished.all() and step < max_steps:
        feas = _router_feas(visited, current_node, load, demand_BP, finished)
        flat = {
            "locs":         locs_BP.reshape(BP, num_nodes, 2),
            "demand":       demand_BP.reshape(BP, num_nodes),
            "visited":      visited.reshape(BP, num_nodes),
            "current_node": current_node.reshape(BP),
            "load":         load.reshape(BP),
            "first_node":   first_customer.reshape(BP),
            "mask":         feas.reshape(BP, num_nodes).float(),
        }
        y_hat = _router_scores(policy, flat).view(B, P, num_nodes)
        sel = y_hat.masked_fill(~feas, float("-inf")).argmax(dim=-1)                    # [B, P]
        sel = torch.where(finished, torch.zeros_like(sel), sel)
        selected_seq.append(sel)
        current_node = sel
        at_depot = (sel == 0)
        load -= demand_BP.gather(2, sel[:, :, None]).squeeze(2)
        load[at_depot] = 1.0
        cust = ~at_depot                                                                # mark customer visits only
        visited[bidx[cust], pidx[cust], sel[cust]] = 1.0
        finished = finished | ((visited[:, :, 1:] > 0).all(dim=-1) & at_depot)
        step += 1

    selected_seq.append(torch.zeros(B, P, dtype=torch.long, device=device))
    selected_seq = torch.stack(selected_seq, dim=-1)
    return selected_seq, _tour_cost(all_locs, selected_seq)


def rollout(policy: EvalPolicy, depot_xy, node_xy, node_demand, starts: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy POMO rollout of the teacher or a router + bank (kinds "teacher", "router", "flat") on one batch of
    instances from first customers 1..starts -> (node sequences [B, starts, T], costs [B, starts]). epb.eval runs it
    with starts = N on batches of the setting's size (float32 results depend on the batch)."""
    if policy.kind == "teacher":
        return _rollout_teacher(policy.model, depot_xy, node_xy, node_demand, starts)
    if policy.kind == "stepper":
        raise ValueError("rollout: tree / agent policies decode through their own engine (_stepper_rollouts)")
    return _rollout_router(policy, depot_xy, node_xy, node_demand, starts)


@torch.no_grad()
def _pomo_rollouts(policy: EvalPolicy, inst: Dict[str, torch.Tensor], a) -> Tuple[torch.Tensor, torch.Tensor]:
    """(costs [n, N] float32, one column per first customer 1..N; trajectory-0 node sequences [n, T]) of the policy
    on the instance set, in batches of _eval_batch instances; computed once per policy and instance set. Kind
    "stepper": the multi-start run of its own protocol (_stepper_rollouts)."""
    if policy.kind == "stepper":
        return _stepper_rollouts(policy, inst, a, "ms")
    d, x, q = inst["depot_xy"], inst["node_xy"], inst["node_demand"]
    n, N = q.shape
    bs = _eval_batch(a, N)
    key = (id(inst), tuple(q.shape), bs)
    if policy.cache is not None and policy.cache[0] == key:
        return policy.cache[1], policy.cache[2]
    t0 = time.time()
    costs, seqs = [], []
    for s in range(0, n, bs):
        e = min(s + bs, n)
        dd, xd, qd = d[s:e].to(policy.device), x[s:e].to(policy.device), q[s:e].to(policy.device)
        seq, c = rollout(policy, dd, xd, qd, N)
        costs.append(c.cpu())
        seqs.append(seq[:, 0].cpu())
        cc = torch.cat(costs)
        print(f"  {e}/{n}  greedy {cc[:, 0].mean():.4f}  ms {cc.min(1).values.mean():.4f}  ({time.time() - t0:.0f}s)",
              flush=True)
    T = max(z.size(1) for z in seqs)
    policy.cache = (key, torch.cat(costs), torch.cat([F.pad(z, (0, T - z.size(1))) for z in seqs]))
    return policy.cache[1], policy.cache[2]


class _State:
    """Construction state of B rows (one trajectory per instance) with the float32 updates of both rollouts:
    load = 1 - demand[first], then load - demand[next], reset to 1 at the depot. visited: customers served so far
    (the depot entry is always False). step: the POMO loop counter (2 at the first decision), so the current node
    is element step - 1 of the node sequence. prev: the node before the current one (the depot at the start)."""

    def __init__(self, d, x, q, first):
        B, N = q.shape
        dev = q.device
        self.B, self.N, self.C = B, N, N + 1
        self.ar = torch.arange(B, device=dev)
        self.locs = torch.cat([d, x], 1)                                                # [B, C, 2]
        self.dem = torch.cat([torch.zeros(B, 1, device=dev), q], 1)                     # [B, C]
        self.first = first.clone()
        self.cur = first.clone()
        self.prev = torch.zeros_like(first)
        self.load = torch.ones(B, device=dev)
        self.load -= self.dem[self.ar, first]
        self.visited = torch.zeros(B, self.C, dtype=torch.bool, device=dev)
        self.visited[self.ar, first] = True
        self.step = 2

    def finished(self) -> torch.Tensor:
        return self.visited[:, 1:].all(-1) & (self.cur == 0)

    def teacher_feas(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(feasible [B, C] bool, ninf_mask [B, C]): the teacher rollout's ninf_mask at this state, == 0 feasible."""
        vinf = torch.zeros(self.B, self.C, device=self.load.device)
        vinf[self.visited] = float("-inf")
        vinf[:, 0] = torch.where(self.cur == 0, float("-inf"), 0.0)
        ninf = vinf.clone()
        ninf[self.load[:, None] + _CAPACITY_EPS < self.dem] = float("-inf")
        ninf[:, 0][(vinf == float("-inf")).all(1)] = 0
        return ninf == 0, ninf

    def move(self, nxt: torch.Tensor) -> None:
        self.prev = self.cur
        self.cur = nxt
        self.load = torch.where(nxt == 0, torch.ones_like(self.load), self.load - self.dem[self.ar, nxt])
        self.visited[self.ar, nxt] = True
        self.visited[:, 0] = False
        self.step += 1


def _choose(policy: EvalPolicy, S: _State) -> torch.Tensor:
    """[B] the policy's argmax at the state, over its own feasible set (as in its own rollout)."""
    if policy.kind == "teacher":
        _, ninf = S.teacher_feas()
        sel, _, _ = policy.model(S.cur[:, None], S.load[:, None], ninf[:, None, :])
        return sel[:, 0]
    if policy.kind == "stepper":
        return policy.stepper.pick(S)
    visited = S.visited.float()
    feas = _router_feas(visited, S.cur, S.load, S.dem, S.finished())
    flat = {"locs": S.locs, "demand": S.dem, "visited": visited, "current_node": S.cur, "load": S.load,
            "first_node": S.first, "mask": feas.float()}
    return _router_scores(policy, flat).masked_fill(~feas, float("-inf")).argmax(dim=-1)


def _t_end(seq: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """[n] index of the first depot visit after which every customer is served; checks the solution."""
    n, N = q.shape
    out = torch.empty(n, dtype=torch.long)
    for i in range(n):
        s = seq[i].tolist()
        served, load = set(), 1.0
        if s[0] != 0 or s[1] != 1:
            raise RuntimeError(f"instance {i}: the teacher trajectory does not start depot -> customer 1")
        for t in range(1, len(s)):
            v = s[t]
            if v == 0:
                load = 1.0
                if len(served) == N:
                    out[i] = t
                    break
            else:
                if v in served:
                    raise RuntimeError(f"instance {i}: customer {v} served twice")
                served.add(v)
                load -= float(q[i, v - 1])
                if load < -_CAPACITY_EPS:
                    raise RuntimeError(f"instance {i}: capacity violated")
        else:
            raise RuntimeError(f"instance {i}: the teacher trajectory never completes")
    return out


@torch.no_grad()
def _forced_pass(policy: EvalPolicy, d, x, q, force: torch.Tensor, t_end: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Walk B instances along the teacher's node sequences force [B, >= max t_end + 2]; at every state t < t_end
    compare the policy's choice with force[:, t + 1]. Rows past their t_end stay in the batch (moved to the depot)."""
    B, N = q.shape
    S = _State(d, x, q, torch.ones(B, dtype=torch.long, device=q.device))
    if policy.kind == "teacher":
        policy.model.pre_forward(S.locs[:, :1], S.locs[:, 1:], S.dem[:, 1:])
    if policy.kind == "stepper":
        policy.stepper.prepare(N)
        policy.stepper.begin(S, S.locs, 1)
    cnt = {k: torch.zeros(B, dtype=torch.long, device=q.device) for k in ("hits", "states", "hits_all", "states_all")}
    Tm = int(t_end.max())
    max_steps = N * 3 + 5
    while S.step < max_steps and S.step - 1 < Tm:
        t = S.step - 1                                                                  # index of the current node
        sel = torch.where(S.finished(), torch.zeros_like(S.cur), _choose(policy, S))
        teacher_feas, _ = S.teacher_feas()
        nxt = force[:, t + 1]
        act = t < t_end
        prim = act & (teacher_feas.sum(1) > 1)
        hit = (sel == nxt) & act
        cnt["hits"] += hit & prim
        cnt["states"] += prim
        cnt["hits_all"] += hit
        cnt["states_all"] += act
        S.move(nxt)
    if S.step - 1 < Tm:
        raise RuntimeError("teacher-forced pass stopped before the end of the teacher trajectories")
    return cnt


@torch.no_grad()
def _top1(teacher: EvalPolicy, policy: EvalPolicy, inst: Dict[str, torch.Tensor], a) -> Dict[str, torch.Tensor]:
    if teacher.kind != "teacher":
        raise ValueError("eval_top1: the first policy must be the teacher")
    d, x, q = inst["depot_xy"], inst["node_xy"], inst["node_demand"]
    n, N = q.shape
    bs = _eval_batch(a, N)
    _, seq = _pomo_rollouts(teacher, inst, a)                   # the teacher's greedy trajectories (trajectory 0)
    t_end = _t_end(seq, q)
    Tm = int(t_end.max())
    force = F.pad(seq, (0, max(0, Tm + 2 - seq.size(1))))
    out = {k: torch.zeros(n, dtype=torch.long) for k in ("hits", "states", "hits_all", "states_all")}
    t0 = time.time()
    dev = policy.device
    for s in range(0, n, bs):
        e = min(s + bs, n)
        r = _forced_pass(policy, d[s:e].to(dev), x[s:e].to(dev), q[s:e].to(dev), force[s:e].to(dev),
                         t_end[s:e].to(dev))
        for k in out:
            out[k][s:e] = r[k].cpu()
        print(f"  top-1 {e}/{n}  {out['hits'][:e].sum() / out['states'][:e].sum():.4f}  ({time.time() - t0:.0f}s)",
              flush=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Baselines (epb.eval): NAM / MoET, decision tree, agent policies
# ─────────────────────────────────────────────────────────────────────────────
# Each baseline is a scorer in the engines above: NAM / MoET score the flat batch of the router rollout (the loop
# their own evaluator used); the tree and the agents are per-state policies on _State, rolled out by _construct with
# the chunking of their own engines (the floats of a row can depend on the rows decoded with it) and their cost
# convention. Top-1 asks the same scorers at the teacher's states. Package code is loaded by path from --pkg.

_PKG_MODULES: Dict[str, object] = {}


def _pkg_module(path: str):
    """A baseline package file loaded by path under a private name (cached); sys.path is restored afterwards, so a
    file that extends it to import its siblings leaves it as it was."""
    import importlib.util
    path = os.path.abspath(path)
    if path not in _PKG_MODULES:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path}: not found")
        name = "epb_pomo_cvrp_" + hashlib.sha1(path.encode()).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        saved = list(sys.path)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path[:] = saved
        _PKG_MODULES[path] = mod
    return _PKG_MODULES[path]


def _baseline_policy(method: str, a, device: torch.device) -> EvalPolicy:
    """nam / moet / tree / agent from --pkg, the package directory under epb/baselines (tree: also --depth). Models
    that depend on N are loaded at the first batch."""
    if not a.pkg or not os.path.isdir(a.pkg):
        raise ValueError(f"--method {method} needs --pkg, its package directory under epb/baselines")
    if method in ("nam", "moet"):
        return EvalPolicy("flat", None, device, score=_AdditiveScorer(method, a.pkg, device))
    if method == "tree":
        return EvalPolicy("stepper", None, device, stepper=_TreeStepper(a.pkg, getattr(a, "depth", None), device))
    if os.path.exists(os.path.join(a.pkg, "rollout", "cvrp_engine.py")):
        return EvalPolicy("stepper", None, device, stepper=_FableStepper(a.pkg))
    if os.path.exists(os.path.join(a.pkg, "whitebox_routing", "cvrp.py")):
        return EvalPolicy("stepper", None, device, stepper=_GptStepper(a.pkg))
    raise ValueError(f"{a.pkg}: not a CVRP agent package (rollout/cvrp_engine.py or whitebox_routing/cvrp.py)")


class _AdditiveScorer:
    """NAM / MoET (<pkg>/<method>_pomo.py, model <pkg>/pomo_cvrp<N>/best.pt, MoET with its soft gate):
    <method>_logits_cvrp on the router rollout's flat batch, i.e. the inputs of the bank programs."""

    def __init__(self, method: str, pkg: str, device: torch.device):
        self.method, self.pkg, self.device, self.N = method, pkg, device, None

    def __call__(self, flat: Dict) -> torch.Tensor:
        N = flat["locs"].size(1) - 1
        if self.N != N:
            mod = _pkg_module(os.path.join(self.pkg, f"{self.method}_pomo.py"))
            path = os.path.join(self.pkg, f"pomo_cvrp{N}", "best.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path}: the package has no {self.method} model for POMO-CVRP N={N}")
            load = mod.load_nam if self.method == "nam" else mod.load_moet
            self.model, _ = load(path, self.device)
            self.logits, self.N = getattr(mod, f"{self.method}_logits_cvrp"), N
        return self.logits(self.model, flat["locs"], flat["demand"], flat["current_node"], flat["mask"].bool(),
                           flat["visited"], flat["load"], flat["first_node"])


class _Stepper:
    """A per-state baseline for _construct and top-1. layout(N, which) -> (instances per call, groups of first
    customers), which = "greedy" | "ms": the chunking of the package's own engine. cost: "seq" (leg lengths summed
    over the node sequence, as the POMO rollouts above), "seq_pad" (the same with one more depot column: the tree
    engine runs one all-finished step more) or "legs" (a running float32 sum, leg by leg, as the agents' engines)."""
    cost = "seq"

    def prepare(self, N: int) -> None:
        """Load what depends on N."""

    def layout(self, N: int, which: str) -> Tuple[int, List[torch.Tensor]]:
        raise NotImplementedError

    def begin(self, S: "_State", inst_locs: torch.Tensor, P: int) -> None:
        """Static inputs of a batch: S has b * P rows (instance-major), inst_locs [b, N+1, 2]."""

    def pick(self, S: "_State") -> torch.Tensor:
        """[rows] the next node of every row."""
        raise NotImplementedError


class _TreeStepper(_Stepper):
    """Decision tree (<pkg>/trees/pomo_cvrp<N>.pkl, scikit-learn; depth d4 | d6 | d8 | d12 | staged, default staged at
    N = 100 and d12 otherwise, the fidelity table's) on the features of <pkg>/features.py (static pairwise distances
    and convex hull, current / first / previous node, unvisited customers, step), evaluated by <pkg>/gpu_tree.py:
    argmax of the leaf probability over the teacher's feasible set. Engine: tree_rollout.pomo_cvrp_rollout
    (greedy: 64 instances from customer 1; multi-start: 1600 // N instances x N starts)."""
    cost = "seq_pad"

    def __init__(self, pkg: str, depth: Optional[str], device: torch.device):
        self.pkg, self.depth_flag, self.device, self.N = pkg, depth, device, None
        self.FT = _pkg_module(os.path.join(pkg, "features.py"))
        self.G = _pkg_module(os.path.join(pkg, "gpu_tree.py"))

    def prepare(self, N: int) -> None:
        if self.N == N:
            return
        import pickle
        path = os.path.join(self.pkg, "trees", f"pomo_cvrp{N}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path}: the package has no POMO-CVRP tree for N={N}")
        with open(path, "rb") as f:
            pl = pickle.load(f)
        depth = self.depth_flag or ("staged" if N == 100 else "d12")
        if depth == "staged":
            self.tree = self.G.TreePolicy("staged", [pl["trees"][f"staged{i}"] for i in range(3)], self.device,
                                          stage_feat_idx=pl["stage_feature_index"],
                                          stage_thresholds=pl["stage_thresholds"])
        elif depth in pl["trees"]:
            self.tree = self.G.TreePolicy(depth, [pl["trees"][depth]], self.device)
        else:
            raise ValueError(f"{path}: no tree of depth {depth!r} (one of {sorted(pl['trees'])} or staged)")
        self.step_max = float(N + max(1, int(N * 0.35)))
        self.N = N

    def layout(self, N: int, which: str) -> Tuple[int, List[torch.Tensor]]:
        if which == "greedy":
            return 64, [torch.tensor([1])]
        return max(1, 1600 // N), [torch.arange(1, N + 1)]

    def begin(self, S, inst_locs, P):
        b, C = inst_locs.shape[:2]
        self.D = self.FT.pairwise_dist(inst_locs)[:, None].expand(b, P, C, C).reshape(b * P, C, C)
        self.hull = self.FT.static_hull(inst_locs)[:, None].expand(b, P, C).reshape(b * P, C)

    def pick(self, S):
        feas, _ = S.teacher_feas()
        unvis = ~S.visited
        unvis[:, 0] = False
        feats = self.FT.cvrp_features(S.locs, S.dem, self.D, self.hull, S.cur, S.first, S.prev, unvis, feas, S.load,
                                      S.step, self.step_max)
        return self.tree.score(feats).masked_fill(~feas, float("-inf")).argmax(-1)


class _FableStepper(_Stepper):
    """Agent policy of <pkg>/rollout/cvrp_engine.py (_step_probs, i.e. cvrp_probs of policies/cvrp_policy.py with
    the per-row customer distances) and the frozen parameters of policies/frozen_params.json; argmax. Engine:
    cvrp_pomo_eval (4096 // N instances x N starts per call; greedy is start 1 of the same run)."""
    cost = "legs"

    def __init__(self, pkg: str):
        import json
        self.E = _pkg_module(os.path.join(pkg, "rollout", "cvrp_engine.py"))
        d = json.load(open(os.path.join(pkg, "policies", "frozen_params.json")))
        if "cvrp_frozen_at" not in d.get("meta", {}):
            raise ValueError(f"{pkg}: the CVRP parameters are not frozen")
        self.p = self.E.CVRPParams(**d["cvrp"])

    def layout(self, N: int, which: str) -> Tuple[int, List[torch.Tensor]]:
        return max(1, 4096 // N), [torch.arange(1, N + 1)]

    def begin(self, S, inst_locs, P):
        self.D = torch.cdist(S.locs[:, 1:], S.locs[:, 1:])

    def pick(self, S):
        vis = S.visited.clone()
        vis[:, 0] = True                                                # the engine marks the depot visited
        pr, _, _ = self.E._step_probs(S.locs, S.dem, vis, S.cur, S.load, self.p, self.D, self.E.EPS_POMO)
        return pr.argmax(-1)


class _GptStepper(_Stepper):
    """Agent policy of <pkg>/whitebox_routing/cvrp.py: pomo_action_distribution on the rows with a customer left
    (capacity tolerance 1e-6; the depot unless the vehicle stands on it), checked by evaluate.py's _check_probs;
    argmax; the depot once every customer is served. Engine: evaluate.construct_cvrp and evaluate_ms.multistart_cvrp
    (1000 instances and one first customer per call)."""
    cost = "legs"
    _TOL = 1e-6

    def __init__(self, pkg: str):
        self.G = _pkg_module(os.path.join(pkg, "whitebox_routing", "cvrp.py"))
        self.E = _pkg_module(os.path.join(pkg, "whitebox_routing", "evaluate.py"))

    def layout(self, N: int, which: str) -> Tuple[int, List[torch.Tensor]]:
        starts = [1] if which == "greedy" else range(1, N + 1)
        return 1000, [torch.tensor([s]) for s in starts]

    def pick(self, S):
        sel = torch.zeros_like(S.cur)
        alive = (~S.visited[:, 1:].all(-1)).nonzero(as_tuple=False).flatten()
        if len(alive):
            f = (~S.visited[alive]) & (S.dem[alive] <= S.load[alive, None] + self._TOL)
            f[:, 0] = S.cur[alive] != 0
            probs = self.G.pomo_action_distribution(S.locs[alive], S.dem[alive], S.cur[alive], f,
                                                    S.visited[alive].float(), S.load[alive], S.first[alive])
            self.E._check_probs(probs, f)
            sel[alive] = probs.argmax(-1)
        return sel


@torch.no_grad()
def _construct(stepper: _Stepper, d, x, q, first: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Roll out a per-state baseline on b instances from the first customers `first` [P] (b * P rows,
    instance-major) -> (node sequences [b, P, T], costs [b, P] in the stepper's cost convention)."""
    b, N = q.shape
    P = first.numel()

    def rep(t):
        return t[:, None].expand(b, P, *t.shape[1:]).reshape(b * P, *t.shape[1:])
    S = _State(rep(d), rep(x), rep(q), first.repeat(b))
    inst_locs = torch.cat([d, x], 1)
    stepper.begin(S, inst_locs, P)
    seq = [torch.zeros_like(S.cur), S.cur.clone()]
    max_steps = N * 3 + 5
    while S.step < max_steps:
        fin = S.finished()
        if bool(fin.all()):
            break
        sel = torch.where(fin, torch.zeros_like(S.cur), stepper.pick(S))
        S.move(sel)
        seq.append(sel)
    if not bool(S.finished().all()):
        raise RuntimeError(f"a {type(stepper).__name__} rollout did not finish within {max_steps} steps")
    seq.append(torch.zeros_like(S.cur))                                 # final depot return
    if stepper.cost == "seq_pad":
        seq.append(torch.zeros_like(S.cur))
    seq = torch.stack(seq, 1)                                           # [b * P, T]
    if stepper.cost == "legs":
        cost = torch.zeros(b * P, device=q.device)
        for t in range(1, seq.size(1)):
            cost = cost + (S.locs[S.ar, seq[:, t]] - S.locs[S.ar, seq[:, t - 1]]).norm(dim=-1)
        cost = cost.view(b, P)
    else:
        cost = _tour_cost(inst_locs, seq.view(b, P, -1))
    return seq.view(b, P, -1), cost


@torch.no_grad()
def _stepper_rollouts(policy: EvalPolicy, inst: Dict[str, torch.Tensor], a, which: str
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(costs [n, starts] float32, one column per first customer of the layout; start-1 node sequences [n, T]) of a
    per-state baseline under its own protocol (which = "greedy" | "ms"; --batch replaces the instances per call);
    cached per layout."""
    d, x, q = inst["depot_xy"], inst["node_xy"], inst["node_demand"]
    n, N = q.shape
    st, dev = policy.stepper, policy.device
    st.prepare(N)
    bs, groups = st.layout(N, which)
    bs = a.batch or bs
    key = (id(inst), tuple(q.shape), bs, tuple(tuple(g.tolist()) for g in groups))
    if key in policy.runs:
        return policy.runs[key]
    t0 = time.time()
    cols, seq0 = [], []
    for gi, g in enumerate(groups):
        cg = []
        for s in range(0, n, bs):
            e = min(s + bs, n)
            seq, c = _construct(st, d[s:e].to(dev), x[s:e].to(dev), q[s:e].to(dev), g.to(dev))
            cg.append(c.cpu())
            if int(g[0]) == 1:
                seq0.append(seq[:, 0].cpu())
            if len(groups) == 1:
                cc = torch.cat(cg)
                print(f"  {which} {e}/{n}  start-1 {cc[:, 0].mean():.4f}  best {cc.min(1).values.mean():.4f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
        cols.append(torch.cat(cg))
        if len(groups) > 1 and (gi % 10 == 9 or gi == len(groups) - 1):
            print(f"  {which} starts {gi + 1}/{len(groups)}  best {torch.cat(cols, 1).min(1).values.mean():.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    T = max(z.size(1) for z in seq0)
    policy.runs[key] = (torch.cat(cols, 1), torch.cat([F.pad(z, (0, T - z.size(1))) for z in seq0]))
    return policy.runs[key]
