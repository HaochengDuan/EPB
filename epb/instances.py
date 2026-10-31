"""Evaluation instance sets (`python -m epb.eval --instances ...`).

  main          the main-table test set of (setting, N), generated from its seed
  n500          the N=500 zero-shot set, shared by the POMO and LEHD settings, generated from its seed
  band:<set>    a set of the bank interventions (pomo_tsp: band_tsp100, pomo_cvrp: band_cvrp50), see BAND
  file:<path>   a file of instance tensors (with an optional "sha256" entry, checked on load)

Each set is drawn from fixed seeds and the same set is used for every method. The POMO-TSP sets are drawn on the GPU,
as by the original evaluator, so they need CUDA.

  TSP   locs [n, N, 2]
  CVRP  depot_xy [n, 1, 2], node_xy [n, N, 2], node_demand [n, N]  (demand divided by the capacity)
"""
import hashlib
from typing import Dict, Optional, Tuple

import torch

MAIN = {("pomo_tsp", 50), ("pomo_tsp", 100), ("pomo_cvrp", 50), ("pomo_cvrp", 100), ("lehd_tsp", 100),
        ("lehd_cvrp", 100)}

BAND = {"pomo_tsp": "band_tsp100", "pomo_cvrp": "band_cvrp50"}
BAND_SETS = {"band_tsp100": {"diag_band_s0": ("diag_band", 0), "diag_band_s3": ("diag_band", 3),
                             "diag_band_s5": ("diag_band", 5), "uniform_s3": ("uniform", 3)},
             "band_cvrp50": {**{f"diag_band_s{s}": ("diag_band", s) for s in (1, 2, 3, 4)},
                             **{f"uniform_s{s}": ("uniform", s) for s in (0, 1, 2)}}}
BAND_INSTANCES, BAND_WIDTH, BAND_CAPACITY = 128, 0.05, 40


def _pomo_tsp(N, seeds, n_per_seed, device):
    """torch.manual_seed(seed); torch.rand(n, N, 2) on the GPU, per seed, concatenated."""
    out = []
    for s in seeds:
        torch.manual_seed(s)
        out.append(torch.rand(n_per_seed, N, 2, device=device))
    return torch.cat(out).cpu()


def _tsp(N, n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, N, 2, generator=g)


def _cvrp_batched(N, n, seed, batch, capacity):
    """torch.manual_seed(seed), then (depot, nodes, demand) drawn in batches of `batch`."""
    torch.manual_seed(seed)
    d, x, q = [], [], []
    done = 0
    while done < n:
        b = min(batch, n - done)
        d.append(torch.rand(size=(b, 1, 2)))
        x.append(torch.rand(size=(b, N, 2)))
        q.append(torch.randint(1, 10, size=(b, N)) / float(capacity))
        done += b
    return torch.cat(d), torch.cat(x), torch.cat(q)


def _cvrp(N, n, seed, capacity):
    """One CPU generator seeded with `seed`: depot, nodes, then demand 1..9 / capacity."""
    g = torch.Generator().manual_seed(seed)
    d = torch.rand(size=(n, 1, 2), generator=g)
    x = torch.rand(size=(n, N, 2), generator=g)
    q = torch.randint(1, 10, size=(n, N), generator=g) / float(capacity)
    return d, x, q


def _band_customers(kind, N, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "uniform":
        return torch.rand(BAND_INSTANCES, N, 2, generator=g)
    t = torch.rand(BAND_INSTANCES, N, generator=g)
    w = (torch.rand(BAND_INSTANCES, N, generator=g) - 0.5) * BAND_WIDTH
    return torch.stack([t - w, t + w], -1).clamp_(0, 1)


def _band(name):
    case, s = name.split("/")
    kind, seed = BAND_SETS[case][s]
    if case == "band_tsp100":
        return {"locs": _band_customers(kind, 100, seed)}
    x = _band_customers(kind, 50, seed)
    g = torch.Generator().manual_seed(seed * 7919 + 17)
    depot = torch.rand(BAND_INSTANCES, 1, 2, generator=g)
    demand = torch.randint(1, 10, (BAND_INSTANCES, 50), generator=g).float()
    return {"depot_xy": depot, "node_xy": x, "node_demand": demand / BAND_CAPACITY}


def _cvrp_dict(d, x, q):
    return {"depot_xy": d, "node_xy": x, "node_demand": q}


def generate(name: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """The instances of set `name` (see name_of), drawn from their seeds."""
    if name in ("pomo_tsp50", "pomo_tsp100") and not (device == "cuda" and torch.cuda.is_available()):
        raise RuntimeError(f"{name} is drawn on the GPU (as by the original evaluator); CUDA is required")
    if name.startswith("band_"):
        return _band(name)
    if name == "pomo_tsp50":
        return {"locs": _pomo_tsp(50, [0, 1, 2], 500, device)}
    if name == "pomo_tsp100":
        return {"locs": _pomo_tsp(100, [0], 1000, device)}
    if name == "pomo_cvrp50":
        return _cvrp_dict(*_cvrp(50, 1000, 4242, 40))
    if name == "pomo_cvrp100":
        return _cvrp_dict(*_cvrp_batched(100, 1000, 4242, 64, 50))
    if name == "lehd_tsp100":
        return {"locs": _tsp(100, 1000, 1234 + 100)}
    if name == "lehd_cvrp100":
        return _cvrp_dict(*_cvrp(100, 256, 1234 + 100, 50))
    if name == "tsp500":
        return {"locs": _tsp(500, 128, 1234 + 500)}
    if name == "cvrp500":                                 # demand 1..9 / capacity 100 (the N=500 convention)
        return _cvrp_dict(*_cvrp(500, 128, 1234 + 500, 100))
    raise ValueError(f"unknown instance set {name!r}")


def digest(tensors: Dict[str, torch.Tensor]) -> str:
    """sha256 over the raw bytes of the instance tensors, in key order."""
    h = hashlib.sha256()
    for k in sorted(tensors):
        t = tensors[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(tuple(t.shape)).encode()); h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def save(path: str, tensors: Dict[str, torch.Tensor]) -> None:
    torch.save({**{k: v.cpu() for k, v in tensors.items()}, "sha256": digest(tensors)}, path)


def name_of(spec: str, setting: str, n: Optional[int]) -> str:
    """The name of instance set `spec` (main | n500 | band:<set>) for the setting."""
    problem = setting.split("_", 1)[1]
    if spec == "main":
        sizes = sorted(N for s, N in MAIN if s == setting)
        if n is None and len(sizes) == 1:          # the LEHD settings have one main-table size
            n = sizes[0]
        if (setting, n) not in MAIN:
            raise ValueError(f"no main-table instance set for {setting} N={n}; one of {sorted(MAIN)}")
        return f"{setting}{n}"
    if spec == "n500":
        return f"{problem}500"
    if spec.startswith("band:"):
        name = f"{BAND.get(setting)}/{spec[5:]}"
        if spec[5:] not in BAND_SETS.get(BAND.get(setting), {}):
            raise ValueError(f"no band set {spec[5:]!r} for {setting}; one of "
                             f"{sorted(BAND_SETS.get(BAND.get(setting), {}))}")
        return name
    raise ValueError(f"unknown instance set {spec!r} (main, n500, band:<set> or file:<path>)")


def load(spec: str, setting: str, n: Optional[int] = None, sl: Optional[Tuple[int, int]] = None
         ) -> Tuple[Dict[str, torch.Tensor], str]:
    """(instance tensors, sha256 of the full set); sl = (lo, hi) keeps a contiguous slice."""
    if spec.startswith("file:"):
        path = spec[5:]
        d = torch.load(path, map_location="cpu", weights_only=True)
        t = {k: v for k, v in d.items() if torch.is_tensor(v)}
        sha = digest(t)
        if "sha256" in d and d["sha256"] != sha:
            raise ValueError(f"{path}: instance tensors do not match the recorded sha256")
    else:
        name = name_of(spec, setting, n)
        t = generate(name)
        sha = digest(t)
    if sl is not None:
        t = {k: v[sl[0]:sl[1]] for k, v in t.items()}
    return t, sha
