import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nam"))
import nam_lehd as C

EPS = 1e-9

INF = float("inf")

TSP_STATE_FEATURES = ["n_u", "log_nu", "d_cur_first", "min_d", "mean_d", "ratio2", "spread", "d_cur_cent",
                      "d_cur_center", "mean_iso"]

CVRP_STATE_FEATURES = ["n_u", "log1p_nu", "load", "tot_dem", "frac_fit", "d_cur_depot", "min_d", "mean_d", "spread",
                       "is_depot"]

@torch.no_grad()
def tsp_state_features(unselected_locs, first_loc, current_loc):
    x = unselected_locs.float()
    B, K, _ = x.shape
    dev = x.device
    x_cur = current_loc.float()
    x_first = first_loc.float()
    n_u = torch.full((B,), float(K), device=dev)
    d_cur = (x - x_cur[:, None, :]).norm(dim=-1)
    d_cur_first = (x_cur - x_first).norm(dim=-1)
    min_d = d_cur.min(-1).values
    mean_d = d_cur.mean(-1)
    if K >= 2:
        two = d_cur.topk(2, dim=-1, largest=False).values
        ratio2 = (two[:, 1] / (two[:, 0] + EPS)).clamp(max=50.0)
    else:
        ratio2 = torch.ones(B, device=dev)
    c_U = x.mean(dim=1)
    spread = ((x - c_U[:, None, :]) ** 2).sum(-1).mean(-1).sqrt()
    d_cur_cent = (x_cur - c_U).norm(dim=-1)
    d_cur_center = (x_cur - 0.5).norm(dim=-1)
    eye = torch.eye(K, dtype=torch.bool, device=dev)
    d1 = C._pair_dist(x).masked_fill(eye[None], INF).min(-1).values
    mean_iso = torch.where(torch.isfinite(d1), d1, torch.zeros_like(d1)).mean(-1)
    out = torch.stack([n_u, n_u.log(), d_cur_first, min_d, mean_d, ratio2, spread, d_cur_cent, d_cur_center,
                       mean_iso], dim=-1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

@torch.no_grad()
def cvrp_state_features(unselected_locs, unselected_demand, current_loc, depot_loc, load, feas):
    x = unselected_locs.float()
    q = unselected_demand.float()
    B, K, _ = x.shape
    dev = x.device
    x_cur = current_loc.float()
    x_0 = depot_loc.float()
    load = load.float().view(B)
    feas = feas.bool()
    n_u = torch.full((B,), float(K), device=dev)
    tot_dem = q.sum(-1)
    frac_fit = feas.float().sum(-1) / n_u
    d_cur_depot = (x_cur - x_0).norm(dim=-1)
    d_cur = (x - x_cur[:, None, :]).norm(dim=-1)
    min_f = d_cur.masked_fill(~feas, INF).min(-1).values
    min_d = torch.where(torch.isfinite(min_f), min_f, d_cur_depot)
    mean_d = d_cur.mean(-1)
    c_U = x.mean(dim=1)
    spread = ((x - c_U[:, None, :]) ** 2).sum(-1).mean(-1).sqrt() if K >= 2 else torch.zeros(B, device=dev)
    is_depot = (x_cur == x_0).all(dim=-1).float()
    out = torch.stack([n_u, torch.log1p(n_u), load, tot_dem, frac_fit, d_cur_depot, min_d, mean_d, spread, is_depot],
                      dim=-1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

class TorchTrees:

    def __init__(self, trees, device):
        self.K = len(trees)
        n_max = max(t["n_nodes"] for t in trees)
        self.depth = max(t["depth"] for t in trees)

        def pad(key, fill, dtype):
            a = np.full((self.K, n_max), fill, dtype=dtype)
            for k, t in enumerate(trees):
                a[k, :t["n_nodes"]] = t[key]
            return torch.from_numpy(a).to(device)

        self.left = pad("left", -1, np.int64)
        self.right = pad("right", -1, np.int64)
        self.feature = pad("feature", 0, np.int64).clamp(min=0)
        self.threshold = pad("threshold", 0.0, np.float64)
        self.value = pad("value", 0.0, np.float64)
        self.device = device

    @torch.no_grad()
    def __call__(self, x):
        shp = x.shape[:-1]
        xf = x.reshape(-1, x.shape[-1]).to(self.device).float().double()
        n = xf.size(0)
        node = torch.zeros(self.K, n, dtype=torch.long, device=self.device)
        for _ in range(self.depth):
            left = self.left.gather(1, node)
            right = self.right.gather(1, node)
            f = self.feature.gather(1, node)
            thr = self.threshold.gather(1, node)
            xv = xf.gather(1, f.t()).t()
            nxt = torch.where(xv <= thr, left, right)
            node = torch.where(left < 0, node, nxt)
        out = self.value.gather(1, node)
        return out.t().float().reshape(*shp, self.K)

class MoET:

    def __init__(self, ck, device, hard=False):
        self.ck = ck
        self.problem = ck["problem"]
        self.device = device
        self.hard = hard
        self.trees = TorchTrees(ck["trees"], device)
        self.W = torch.as_tensor(ck["gate_W"], dtype=torch.float32, device=device)
        self.b = torch.as_tensor(ck["gate_b"], dtype=torch.float32, device=device)
        self.z_mean = torch.as_tensor(ck["z_mean"], dtype=torch.float32, device=device)
        self.z_std = torch.as_tensor(ck["z_std"], dtype=torch.float32, device=device)
        self.K = self.W.size(0)

    @torch.no_grad()
    def gate(self, z):
        zt = (z.to(self.device).float() - self.z_mean) / self.z_std
        logits = zt @ self.W.t() + self.b
        if self.hard:
            return F.one_hot(logits.argmax(-1), self.K).float()
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def scores(self, x, z):
        T = self.trees(x)
        g = self.gate(z)
        return (T * g[:, None, :]).sum(-1)

def load_moet(path, device, hard=False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    names = C.TSP_FEATURES if ck["problem"] == "tsp" else C.CVRP_FEATURES
    assert ck["node_features"] == names
    return MoET(ck, device, hard=hard), ck

class MoETScorerTSP:

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, unselected_locs, first_loc, current_loc):
        dev = self.model.device
        u, f, c = unselected_locs.to(dev), first_loc.to(dev), current_loc.to(dev)
        return self.model.scores(C.tsp_features(u, f, c), tsp_state_features(u, f, c))

class MoETScorerCVRP:

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, unselected_locs, unselected_demand, current_loc, depot_loc, load, feas):
        dev = self.model.device
        u, q, c, d0, ld, fe = (unselected_locs.to(dev), unselected_demand.to(dev), current_loc.to(dev),
                               depot_loc.to(dev), load.to(dev), feas.to(dev))
        s = self.model.scores(C.cvrp_features(u, q, c, d0, ld, fe), cvrp_state_features(u, q, c, d0, ld, fe))
        K = feas.size(1)
        s = s.clone()
        s[:, :K] = s[:, :K].masked_fill(~fe.bool(), float("-inf"))
        return s

def make_bank_tsp(model):
    return C.NAMBankTSP(MoETScorerTSP(model)), C.NullRouterTSP()

def make_bank_cvrp(model):
    return C.NAMBankCVRP(MoETScorerCVRP(model)), C.NullRouterCVRP()

