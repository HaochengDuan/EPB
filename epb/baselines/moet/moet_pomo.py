import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nam"))
import nam_pomo as NC

INF = float("inf")

EPS = 1e-9

TSP_STATE_NAMES = ["n_u", "log_nu", "d_cur_first", "min_d", "mean_d", "ratio2", "spread", "d_cur_cent",
                   "d_cur_center", "mean_iso"]

CVRP_STATE_NAMES = ["n_u", "log1p_nu", "load", "tot_dem", "frac_fit", "d_cur_depot", "min_d", "mean_d", "spread",
                    "is_depot"]

@torch.no_grad()
def tsp_state_features(locs, current_node, mask, visited, step_norm, first_node) -> torch.Tensor:
    B, N, _ = locs.shape
    dev = locs.device
    ar = torch.arange(B, device=dev)
    cur = current_node.long().view(B)
    first = first_node.long().view(B)
    U = mask.bool()
    n_u = U.sum(-1).clamp(min=1).float()
    x_cur = locs[ar, cur]
    x_first = locs[ar, first]
    d_cur = (locs - x_cur[:, None, :]).norm(dim=-1)
    d_cur_first = (x_cur - x_first).norm(dim=-1)
    d_u = d_cur.masked_fill(~U, INF)
    two = d_u.topk(2, dim=-1, largest=False).values
    min_d = torch.where(torch.isfinite(two[:, 0]), two[:, 0], torch.zeros_like(two[:, 0]))
    ratio2 = torch.where(torch.isfinite(two[:, 1]), two[:, 1] / min_d.clamp(min=EPS), torch.ones_like(min_d)).clamp(max=50.0)
    mean_d = (d_cur * U).sum(-1) / n_u
    c_U = (locs * U[..., None]).sum(1) / n_u[:, None]
    spread = ((((locs - c_U[:, None, :]) ** 2).sum(-1) * U).sum(-1) / n_u).sqrt()
    d_cur_cent = (x_cur - c_U).norm(dim=-1)
    d_cur_center = (x_cur - 0.5).norm(dim=-1)
    D = NC._pair_dist(locs)
    eye = torch.eye(N, dtype=torch.bool, device=dev)
    d1 = D.masked_fill(~U[:, None, :], INF).masked_fill(eye[None], INF).min(-1).values
    d1 = torch.where(torch.isfinite(d1), d1, torch.zeros_like(d1))
    mean_iso = (d1 * U).sum(-1) / n_u
    z = torch.stack([n_u, n_u.log(), d_cur_first, min_d, mean_d, ratio2, spread, d_cur_cent, d_cur_center, mean_iso], -1)
    return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

@torch.no_grad()
def cvrp_state_features(locs, demand, current_node, mask, visited, load, first_node) -> torch.Tensor:
    B, C, _ = locs.shape
    dev = locs.device
    ar = torch.arange(B, device=dev)
    cur = current_node.long().view(B)
    load = load.float().view(B)
    F_ = mask.bool()
    cust = torch.ones(C, dtype=torch.bool, device=dev)
    cust[0] = False
    U = (visited.view(B, C) == 0) & cust[None]
    n_u = U.sum(-1).float()
    n_u1 = n_u.clamp(min=1)
    x_cur = locs[ar, cur]
    d_cur = (locs - x_cur[:, None, :]).norm(dim=-1)
    d_cur_depot = d_cur[:, 0]
    tot_dem = (demand * U).sum(-1)
    frac_fit = (F_ & U).sum(-1).float() / n_u1
    d_fc = d_cur.masked_fill(~(F_ & U), INF).min(-1).values
    min_d = torch.where(torch.isfinite(d_fc), d_fc, d_cur_depot)
    mean_d = (d_cur * U).sum(-1) / n_u1
    c_U = (locs * U[..., None]).sum(1) / n_u1[:, None]
    spread = ((((locs - c_U[:, None, :]) ** 2).sum(-1) * U).sum(-1) / n_u1).sqrt()
    is_depot = (cur == 0).float()
    z = torch.stack([n_u, torch.log1p(n_u), load, tot_dem, frac_fit, d_cur_depot, min_d, mean_d, spread, is_depot], -1)
    return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

class PackedTrees:

    def __init__(self, feature, threshold, left, right, value, depth: int, n_nodes: List[int], n_leaves: List[int],
                 depths: List[int]):
        self.feature, self.threshold, self.left, self.right, self.value = feature, threshold, left, right, value
        self.depth, self.n_nodes, self.n_leaves, self.depths = int(depth), list(n_nodes), list(n_leaves), list(depths)
        self.K = feature.size(0)

    @staticmethod
    def from_sklearn(trees) -> "PackedTrees":
        K = len(trees)
        M = max(int(t.tree_.node_count) for t in trees)
        feature = torch.zeros(K, M, dtype=torch.long)
        threshold = torch.zeros(K, M, dtype=torch.float64)
        left = torch.full((K, M), -1, dtype=torch.long)
        right = torch.full((K, M), -1, dtype=torch.long)
        value = torch.zeros(K, M, dtype=torch.float32)
        n_nodes, n_leaves, depths = [], [], []
        for k, t in enumerate(trees):
            tt = t.tree_
            n = int(tt.node_count)
            f = torch.from_numpy(np.asarray(tt.feature, dtype=np.int64))
            feature[k, :n] = torch.where(f >= 0, f, torch.zeros_like(f))
            threshold[k, :n] = torch.from_numpy(np.asarray(tt.threshold, dtype=np.float64))
            left[k, :n] = torch.from_numpy(np.asarray(tt.children_left, dtype=np.int64))
            right[k, :n] = torch.from_numpy(np.asarray(tt.children_right, dtype=np.int64))
            value[k, :n] = torch.from_numpy(np.asarray(tt.value[:, 0, 0], dtype=np.float32))
            n_nodes.append(n)
            n_leaves.append(int(t.get_n_leaves()))
            depths.append(int(t.get_depth()))
        return PackedTrees(feature, threshold, left, right, value, max(depths), n_nodes, n_leaves, depths)

    def to(self, dev) -> "PackedTrees":
        return PackedTrees(self.feature.to(dev), self.threshold.to(dev), self.left.to(dev), self.right.to(dev),
                           self.value.to(dev), self.depth, self.n_nodes, self.n_leaves, self.depths)

    def state_dict(self) -> Dict:
        return dict(feature=self.feature.cpu(), threshold=self.threshold.cpu(), left=self.left.cpu(),
                    right=self.right.cpu(), value=self.value.cpu(), depth=self.depth, n_nodes=self.n_nodes,
                    n_leaves=self.n_leaves, depths=self.depths)

    @staticmethod
    def from_state_dict(d) -> "PackedTrees":
        return PackedTrees(d["feature"], d["threshold"], d["left"], d["right"], d["value"], d["depth"], d["n_nodes"],
                           d["n_leaves"], d["depths"])

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        R = x.size(0)
        xt = x.double().t()
        node = torch.zeros(self.K, R, dtype=torch.long, device=x.device)
        for _ in range(self.depth):
            f = self.feature.gather(1, node)
            thr = self.threshold.gather(1, node)
            xv = xt.gather(0, f)
            l = self.left.gather(1, node)
            r = self.right.gather(1, node)
            nxt = torch.where(xv <= thr, l, r)
            node = torch.where(l < 0, node, nxt)
        return self.value.gather(1, node).t().contiguous()

class MoET:

    def __init__(self, problem: str, trees: PackedTrees, gate_W: torch.Tensor, gate_b: torch.Tensor,
                 z_mu: torch.Tensor, z_sigma: torch.Tensor, node_names: List[str], state_names: List[str],
                 hard: bool = False):
        self.problem, self.trees = problem, trees
        self.W, self.b = gate_W.float(), gate_b.float()
        self.z_mu, self.z_sigma = z_mu.float(), z_sigma.float().clamp(min=1e-6)
        self.node_names, self.state_names = list(node_names), list(state_names)
        self.K, self.F, self.J = trees.K, len(node_names), len(state_names)
        self.hard = hard

    def to(self, dev) -> "MoET":
        m = MoET(self.problem, self.trees.to(dev), self.W.to(dev), self.b.to(dev), self.z_mu.to(dev),
                 self.z_sigma.to(dev), self.node_names, self.state_names, self.hard)
        return m

    @torch.no_grad()
    def gate(self, z: torch.Tensor) -> torch.Tensor:
        zs = (z.float() - self.z_mu) / self.z_sigma
        logits = zs @ self.W.t() + self.b
        if self.hard:
            return torch.nn.functional.one_hot(logits.argmax(-1), self.K).float()
        return torch.softmax(logits, -1)

    @torch.no_grad()
    def expert_scores(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F = x.shape
        return self.trees(x.reshape(B * C, F)).view(B, C, self.K)

    @torch.no_grad()
    def logits(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        t = self.expert_scores(x)
        g = self.gate(z)
        return (t * g[:, None, :]).sum(-1)

    def n_params(self) -> int:
        return sum(2 * (n - l) + l for n, l in zip(self.trees.n_nodes, self.trees.n_leaves)) + self.W.numel() + self.b.numel()

    def gate_table(self) -> Dict:
        return dict(state_names=self.state_names, W=self.W.cpu().tolist(), b=self.b.cpu().tolist(),
                    z_mu=self.z_mu.cpu().tolist(), z_sigma=self.z_sigma.cpu().tolist())

def moet_logits_tsp(model: MoET, locs, current_node, mask, visited, step_norm, first_node) -> torch.Tensor:
    x = NC.tsp_features(locs, current_node, mask, visited, step_norm, first_node)
    z = tsp_state_features(locs, current_node, mask, visited, step_norm, first_node)
    return model.logits(x, z)

def moet_logits_cvrp(model: MoET, locs, demand, current_node, mask, visited, load, first_node) -> torch.Tensor:
    x = NC.cvrp_features(locs, demand, current_node, mask, visited, load, first_node)
    z = cvrp_state_features(locs, demand, current_node, mask, visited, load, first_node)
    return model.logits(x, z)

def load_moet(path: str, device=None, hard: bool = False) -> Tuple[MoET, Dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = MoET(ck["problem"], PackedTrees.from_state_dict(ck["trees"]), ck["gate_W"], ck["gate_b"], ck["z_mu"],
             ck["z_sigma"], ck["node_names"], ck["state_names"], hard=hard)
    if device is not None:
        m = m.to(device)
    keys = ("problem", "trees", "gate_W", "gate_b", "z_mu", "z_sigma", "node_names", "state_names")
    meta = {k: v for k, v in ck.items() if k not in keys}
    return m, meta

