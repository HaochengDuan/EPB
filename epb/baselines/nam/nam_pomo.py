from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

EPS = 1e-9

INF = float("inf")

TSP_FEATURE_NAMES = ["d_cur", "rank_cur", "ratio_cur", "d_first", "closing", "cos_home", "cos_mass", "iso",
                     "d_cent", "cnt_r", "d_center", "cos_onward"]

CVRP_FEATURE_NAMES = ["d_cur", "rank_cur", "ratio_cur", "d_depot", "savings", "dem_ratio", "slack", "ang_depot",
                      "dem_r", "cnt_r", "iso", "n_fit", "is_depot", "cos_depot", "load_depot"]

def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    na = a.norm(dim=-1)
    nb = b.norm(dim=-1)
    c = (a * b).sum(-1) / (na * nb + EPS)
    c = torch.where((na > 0) & (nb > 0), c, torch.zeros_like(c))
    return c.clamp(-1.0, 1.0)

def _pair_dist(locs: torch.Tensor) -> torch.Tensor:
    diff = locs[:, :, None, :] - locs[:, None, :, :]
    return diff.norm(dim=-1)

@torch.no_grad()
def tsp_features(locs, current_node, mask, visited, step_norm, first_node) -> torch.Tensor:
    B, N, _ = locs.shape
    dev = locs.device
    ar = torch.arange(B, device=dev)
    cur = current_node.long().view(B)
    first = first_node.long().view(B)
    U = mask.bool()
    n_u = U.sum(-1).clamp(min=1).float()
    D = _pair_dist(locs)
    d_cur = D[ar, cur]
    d_first = D[ar, first]
    x_cur = locs[ar, cur]
    x_first = locs[ar, first]
    less = (d_cur[:, None, :] < d_cur[:, :, None]) & U[:, None, :]
    rank_cur = less.sum(-1).float()
    d_nn_cur = d_cur.masked_fill(~U, INF).min(-1).values
    d_nn_cur = torch.where(torch.isfinite(d_nn_cur), d_nn_cur, torch.ones_like(d_nn_cur))
    ratio_cur = (d_cur / d_nn_cur.clamp(min=1e-6)[:, None]).clamp(max=50.0)
    closing = d_first / n_u[:, None]
    v_j = locs - x_cur[:, None, :]
    cos_home = _cos(v_j, (x_first - x_cur)[:, None, :])
    c_U = (locs * U[..., None]).sum(1) / n_u[:, None]
    cos_mass = _cos(v_j, (c_U - x_cur)[:, None, :])
    D_u = D.masked_fill(~U[:, None, :], INF)
    eye = torch.eye(N, dtype=torch.bool, device=dev)
    D_u = D_u.masked_fill(eye[None], INF)
    d1, nn_idx = D_u.min(-1)
    has_nn = torch.isfinite(d1)
    iso = torch.where(has_nn, d1, torch.zeros_like(d1))
    d_cent = (locs - c_U[:, None, :]).norm(dim=-1)
    r = (1.5 / n_u.sqrt())[:, None, None]
    near = (D < r) & U[:, None, :]
    cnt_r = near.sum(-1).float() - U.float()
    d_center = (locs - 0.5).norm(dim=-1)
    nn_safe = torch.where(has_nn, nn_idx, torch.zeros_like(nn_idx))
    x_nn = locs.gather(1, nn_safe[..., None].expand(-1, -1, 2))
    cos_onward = _cos(x_cur[:, None, :] - locs, x_nn - locs)
    cos_onward = torch.where(has_nn, cos_onward, torch.zeros_like(cos_onward))
    x = torch.stack([d_cur, rank_cur, ratio_cur, d_first, closing, cos_home, cos_mass, iso, d_cent, cnt_r,
                     d_center, cos_onward], dim=-1)
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

@torch.no_grad()
def cvrp_features(locs, demand, current_node, mask, visited, load, first_node) -> torch.Tensor:
    B, C, _ = locs.shape
    dev = locs.device
    ar = torch.arange(B, device=dev)
    cur = current_node.long().view(B)
    load = load.float().view(B)
    F_ = mask.bool()
    cust = torch.ones(C, dtype=torch.bool, device=dev)
    cust[0] = False
    U = (visited.view(B, C) == 0) & cust[None]
    n_u = U.sum(-1).clamp(min=1).float()
    D = _pair_dist(locs)
    d_cur = D[ar, cur]
    d_depot = D[:, 0, :]
    d_cur_depot = d_cur[:, 0]
    x_cur = locs[ar, cur]
    x_0 = locs[:, 0]
    less = (d_cur[:, None, :] < d_cur[:, :, None]) & F_[:, None, :]
    rank_cur = less.sum(-1).float()
    d_nn_c = d_cur.masked_fill(~(F_ & cust[None]), INF).min(-1).values
    d_nn_f = d_cur.masked_fill(~F_, INF).min(-1).values
    d_nn = torch.where(torch.isfinite(d_nn_c), d_nn_c, d_nn_f)
    d_nn = torch.where(torch.isfinite(d_nn), d_nn, torch.ones_like(d_nn))
    ratio_cur = (d_cur / d_nn.clamp(min=1e-6)[:, None]).clamp(max=50.0)
    savings = d_cur_depot[:, None] + d_depot - d_cur
    savings[:, 0] = 0.0
    dem_ratio = (demand / load.clamp(min=1e-6)[:, None]).clamp(max=2.0)
    dem_ratio[:, 0] = 0.0
    slack = load[:, None] - demand
    slack[:, 0] = 1.0
    a_j = torch.atan2(locs[..., 1] - x_0[:, None, 1], locs[..., 0] - x_0[:, None, 0])
    a_cur = a_j[ar, cur]
    ang = (a_j - a_cur[:, None]).abs()
    ang = torch.where(ang > torch.pi, 2 * torch.pi - ang, ang)
    ang = torch.where((cur == 0)[:, None], torch.zeros_like(ang), ang)
    ang[:, 0] = 0.0
    r = (1.5 / n_u.sqrt())[:, None, None]
    near = (D < r) & U[:, None, :]
    cnt_r = near.sum(-1).float() - U.float()
    dem_r = (near.float() * demand[:, None, :]).sum(-1) - demand * U.float()
    D_u = D.masked_fill(~U[:, None, :], INF)
    eye = torch.eye(C, dtype=torch.bool, device=dev)
    d1 = D_u.masked_fill(eye[None], INF).min(-1).values
    iso = torch.where(torch.isfinite(d1), d1, torch.zeros_like(d1))
    rem = (load[:, None] - demand)
    fits = (demand[:, None, :] <= rem[:, :, None]) & U[:, None, :]
    self_fit = (demand <= rem) & U
    n_fit = fits.sum(-1).float() - self_fit.float()
    n_fit[:, 0] = n_u
    is_depot = torch.zeros(B, C, device=dev)
    is_depot[:, 0] = 1.0
    cos_depot = _cos(locs - x_cur[:, None, :], (x_0 - x_cur)[:, None, :])
    cos_depot = torch.where((cur == 0)[:, None], torch.zeros_like(cos_depot), cos_depot)
    cos_depot[:, 0] = 1.0
    load_depot = load[:, None] * is_depot
    x = torch.stack([d_cur, rank_cur, ratio_cur, d_depot, savings, dem_ratio, slack, ang, dem_r, cnt_r, iso,
                     n_fit, is_depot, cos_depot, load_depot], dim=-1)
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

class NAM(nn.Module):

    def __init__(self, K: int, hidden: int = 32, feature_names: Optional[List[str]] = None):
        super().__init__()
        self.K, self.H = K, hidden
        self.feature_names = list(feature_names) if feature_names else [f"x{k}" for k in range(K)]
        self.register_buffer("mu", torch.zeros(K))
        self.register_buffer("sigma", torch.ones(K))
        lin = lambda i, o: nn.Linear(i, o)
        w1, b1, w2, b2, w3, b3 = [], [], [], [], [], []
        for _ in range(K):
            l1, l2, l3 = lin(1, hidden), lin(hidden, hidden), lin(hidden, 1)
            w1.append(l1.weight.detach().clone().view(hidden)); b1.append(l1.bias.detach().clone())
            w2.append(l2.weight.detach().clone().t()); b2.append(l2.bias.detach().clone())
            w3.append(l3.weight.detach().clone().view(hidden)); b3.append(l3.bias.detach().clone().view(1))
        self.w1 = nn.Parameter(torch.stack(w1))
        self.b1 = nn.Parameter(torch.stack(b1))
        self.w2 = nn.Parameter(torch.stack(w2))
        self.b2 = nn.Parameter(torch.stack(b2))
        self.w3 = nn.Parameter(torch.stack(w3))
        self.b3 = nn.Parameter(torch.stack(b3).view(K))
        self.bias = nn.Parameter(torch.zeros(1))

    def set_standardisation(self, mu: torch.Tensor, sigma: torch.Tensor):
        self.mu.copy_(mu)
        self.sigma.copy_(sigma.clamp(min=1e-6))

    def contributions(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mu) / self.sigma
        lead = z.shape[:-1]
        z = z.reshape(-1, self.K)
        h = torch.relu(z[:, :, None] * self.w1[None] + self.b1[None])
        h = h.permute(1, 0, 2)
        h = torch.relu(torch.bmm(h, self.w2) + self.b2[:, None, :])
        out = (h * self.w3[:, None, :]).sum(-1) + self.b3[:, None]
        return out.t().reshape(*lead, self.K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.contributions(x).sum(-1) + self.bias

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

def nam_logits_tsp(model: NAM, locs, current_node, mask, visited, step_norm, first_node) -> torch.Tensor:
    x = tsp_features(locs, current_node, mask, visited, step_norm, first_node)
    return model(x)

def nam_logits_cvrp(model: NAM, locs, demand, current_node, mask, visited, load, first_node) -> torch.Tensor:
    x = cvrp_features(locs, demand, current_node, mask, visited, load, first_node)
    return model(x)

def load_nam(path: str, device=None) -> Tuple[NAM, Dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = NAM(ck["K"], ck["hidden"], ck["feature_names"])
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    if device is not None:
        m = m.to(device)
    meta = {k: v for k, v in ck.items() if k not in ("model_state_dict",)}
    return m, meta

