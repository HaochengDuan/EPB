import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9

TSP_FEATURES = ["d_cur", "rank_cur", "ratio_cur", "d_first", "closing", "cos_home", "cos_mass", "iso", "d_cent",
                "cnt_r", "d_center", "cos_onward"]

CVRP_FEATURES = ["d_cur", "rank_cur", "ratio_cur", "d_depot", "savings", "dem_ratio", "slack", "ang_depot", "dem_r",
                 "cnt_r", "iso", "n_fit", "is_depot", "cos_depot", "load_depot"]

INF = float("inf")

def _cos(a, b):
    na = a.norm(dim=-1)
    nb = b.norm(dim=-1)
    c = (a * b).sum(-1) / (na * nb + EPS)
    c = torch.where((na > 0) & (nb > 0), c, torch.zeros_like(c))
    return c.clamp(-1.0, 1.0)

def _pair_dist(x):
    return (x[:, :, None, :] - x[:, None, :, :]).norm(dim=-1)

@torch.no_grad()
def tsp_features(unselected_locs, first_loc, current_loc):
    x = unselected_locs.float()
    B, K, _ = x.shape
    dev = x.device
    x_cur = current_loc.float()
    x_first = first_loc.float()
    n_u = torch.full((B,), float(K), device=dev)
    D = _pair_dist(x)
    d_cur = (x - x_cur[:, None, :]).norm(dim=-1)
    d_first = (x - x_first[:, None, :]).norm(dim=-1)
    rank_cur = (d_cur[:, None, :] < d_cur[:, :, None]).sum(-1).float()
    d_nn_cur = d_cur.min(-1).values
    ratio_cur = (d_cur / d_nn_cur.clamp(min=1e-6)[:, None]).clamp(max=50.0)
    closing = d_first / n_u[:, None]
    v_j = x - x_cur[:, None, :]
    cos_home = _cos(v_j, (x_first - x_cur)[:, None, :])
    c_U = x.mean(dim=1)
    cos_mass = _cos(v_j, (c_U - x_cur)[:, None, :])
    eye = torch.eye(K, dtype=torch.bool, device=dev)
    D_u = D.masked_fill(eye[None], INF)
    d1, nn_idx = D_u.min(-1)
    has_nn = torch.isfinite(d1)
    iso = torch.where(has_nn, d1, torch.zeros_like(d1))
    d_cent = (x - c_U[:, None, :]).norm(dim=-1)
    r = (1.5 / n_u.sqrt())[:, None, None]
    cnt_r = (D < r).sum(-1).float() - 1.0
    d_center = (x - 0.5).norm(dim=-1)
    nn_safe = torch.where(has_nn, nn_idx, torch.zeros_like(nn_idx))
    x_nn = x.gather(1, nn_safe[..., None].expand(-1, -1, 2))
    cos_onward = _cos(x_cur[:, None, :] - x, x_nn - x)
    cos_onward = torch.where(has_nn, cos_onward, torch.zeros_like(cos_onward))
    out = torch.stack([d_cur, rank_cur, ratio_cur, d_first, closing, cos_home, cos_mass, iso, d_cent, cnt_r,
                       d_center, cos_onward], dim=-1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

@torch.no_grad()
def cvrp_features(unselected_locs, unselected_demand, current_loc, depot_loc, load, feas):
    x = unselected_locs.float()
    q = unselected_demand.float()
    B, K, _ = x.shape
    dev = x.device
    x_cur = current_loc.float()
    x_0 = depot_loc.float()
    load = load.float().view(B)
    feas = feas.bool()
    at_depot = (x_cur == x_0).all(dim=-1)
    n_u = torch.full((B,), float(K), device=dev)
    xa = torch.cat([x, x_0[:, None, :]], dim=1)
    d_cur = (xa - x_cur[:, None, :]).norm(dim=-1)
    d_depot = (xa - x_0[:, None, :]).norm(dim=-1)
    d_cur_depot = d_cur[:, K]
    F_ = torch.cat([feas, (~at_depot)[:, None]], dim=1)
    cust = torch.ones(K + 1, dtype=torch.bool, device=dev); cust[K] = False
    rank_cur = ((d_cur[:, None, :] < d_cur[:, :, None]) & F_[:, None, :]).sum(-1).float()
    d_nn_c = d_cur.masked_fill(~(F_ & cust[None]), INF).min(-1).values
    d_nn_f = d_cur.masked_fill(~F_, INF).min(-1).values
    d_nn = torch.where(torch.isfinite(d_nn_c), d_nn_c, d_nn_f)
    d_nn = torch.where(torch.isfinite(d_nn), d_nn, torch.ones_like(d_nn))
    ratio_cur = (d_cur / d_nn.clamp(min=1e-6)[:, None]).clamp(max=50.0)
    savings = d_cur_depot[:, None] + d_depot - d_cur
    savings[:, K] = 0.0
    qa = torch.cat([q, torch.zeros(B, 1, device=dev)], dim=1)
    dem_ratio = (qa / load.clamp(min=1e-6)[:, None]).clamp(max=2.0)
    dem_ratio[:, K] = 0.0
    slack = load[:, None] - qa
    slack[:, K] = 1.0
    a_j = torch.atan2(xa[..., 1] - x_0[:, None, 1], xa[..., 0] - x_0[:, None, 0])
    a_cur = torch.atan2(x_cur[:, 1] - x_0[:, 1], x_cur[:, 0] - x_0[:, 0])
    ang = (a_j - a_cur[:, None]).abs()
    ang = torch.where(ang > torch.pi, 2 * torch.pi - ang, ang)
    ang = torch.where(at_depot[:, None], torch.zeros_like(ang), ang)
    ang[:, K] = 0.0
    Da = _pair_dist(xa)
    r = (1.5 / n_u.sqrt())[:, None, None]
    U = cust[None].expand(B, K + 1)
    near = (Da < r) & U[:, None, :]
    cnt_r = near.sum(-1).float() - U.float()
    dem_r = (near.float() * qa[:, None, :]).sum(-1) - qa * U.float()
    eye = torch.eye(K + 1, dtype=torch.bool, device=dev)
    d1 = Da.masked_fill(~U[:, None, :], INF).masked_fill(eye[None], INF).min(-1).values
    iso = torch.where(torch.isfinite(d1), d1, torch.zeros_like(d1))
    rem = load[:, None] - qa
    fits = (qa[:, None, :] <= rem[:, :, None]) & U[:, None, :]
    self_fit = (qa <= rem) & U
    n_fit = fits.sum(-1).float() - self_fit.float()
    n_fit[:, K] = n_u
    is_depot = torch.zeros(B, K + 1, device=dev)
    is_depot[:, K] = 1.0
    cos_depot = _cos(xa - x_cur[:, None, :], (x_0 - x_cur)[:, None, :])
    cos_depot = torch.where(at_depot[:, None], torch.zeros_like(cos_depot), cos_depot)
    cos_depot[:, K] = 1.0
    load_depot = load[:, None] * is_depot
    out = torch.stack([d_cur, rank_cur, ratio_cur, d_depot, savings, dem_ratio, slack, ang, dem_r, cnt_r, iso, n_fit,
                       is_depot, cos_depot, load_depot], dim=-1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

class NAM(nn.Module):

    def __init__(self, n_feat, hidden=32, mean=None, std=None):
        super().__init__()
        self.n_feat, self.hidden = n_feat, hidden
        Fh = n_feat
        self.W1 = nn.Parameter(torch.empty(Fh, hidden))
        self.b1 = nn.Parameter(torch.empty(Fh, hidden))
        self.W2 = nn.Parameter(torch.empty(Fh, hidden, hidden))
        self.b2 = nn.Parameter(torch.empty(Fh, hidden))
        self.W3 = nn.Parameter(torch.empty(Fh, hidden))
        self.b3 = nn.Parameter(torch.zeros(Fh))
        self.bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("mean", torch.zeros(Fh) if mean is None else mean.clone().float())
        self.register_buffer("std", torch.ones(Fh) if std is None else std.clone().float())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.W1, -1.0, 1.0)
        nn.init.uniform_(self.b1, -1.0, 1.0)
        k = 1.0 / math.sqrt(self.hidden)
        nn.init.uniform_(self.W2, -k, k)
        nn.init.uniform_(self.b2, -k, k)
        nn.init.uniform_(self.W3, -k, k)
        nn.init.zeros_(self.b3)
        nn.init.zeros_(self.bias)

    def shape(self, x):
        z = (x - self.mean) / self.std
        h = F.relu(z.unsqueeze(-1) * self.W1 + self.b1)
        h = F.relu(torch.einsum("...fi,fij->...fj", h, self.W2) + self.b2)
        return (h * self.W3).sum(-1) + self.b3

    def forward(self, x):
        return self.shape(x).sum(-1) + self.bias

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

def load_nam(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = NAM(ck["n_feat"], hidden=ck["hidden"])
    m.load_state_dict(ck["model_state_dict"], strict=True)
    names = TSP_FEATURES if ck["problem"] == "tsp" else CVRP_FEATURES
    assert ck["features"] == names, "feature list of the checkpoint differs from nam_lehd"
    return m.to(device).eval(), ck

class NAMScorerTSP:

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, unselected_locs, first_loc, current_loc):
        dev = next(self.model.parameters()).device
        x = tsp_features(unselected_locs.to(dev), first_loc.to(dev), current_loc.to(dev))
        return self.model(x)

class NAMScorerCVRP:

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, unselected_locs, unselected_demand, current_loc, depot_loc, load, feas):
        dev = next(self.model.parameters()).device
        x = cvrp_features(unselected_locs.to(dev), unselected_demand.to(dev), current_loc.to(dev), depot_loc.to(dev),
                          load.to(dev), feas.to(dev))
        s = self.model(x)
        K = feas.size(1)
        s = s.clone()
        s[:, :K] = s[:, :K].masked_fill(~feas.to(dev).bool(), float("-inf"))
        return s

class NAMBankTSP:

    def __init__(self, scorer):
        self.scorer = scorer
        self.heuristics = [{"index": 0, "fn": scorer}]
        self.M = 1
        self.heuristic_temperature = 1.0

    @torch.no_grad()
    def compute_probs(self, batch, device):
        mask = batch["mask"].bool().to(device)
        unsel_idx = batch["unselected_idx"].long().to(device)
        B, N = mask.shape
        heu_K = self.scorer(batch["unselected_locs"].float().to(device), batch["first_loc"].float().to(device),
                            batch["current_loc"].float().to(device)).float()
        assert heu_K.shape == unsel_idx.shape and torch.isfinite(heu_K).all()
        V_K = F.softmax(heu_K, dim=-1)
        full = torch.full((B, N), float("-inf"), device=device)
        full.scatter_(1, unsel_idx, heu_K)
        V_full = F.softmax(full.masked_fill(~mask, float("-inf")), dim=-1).masked_fill(~mask, 0.0)
        return V_full.unsqueeze(1), V_K.unsqueeze(1)

class NullRouterTSP(nn.Module):

    def __init__(self):
        super().__init__()
        self.routing_temperature = 1.0
        self.num_heuristics = 1

    def encode_subgraph(self, unsel, first, current):
        B, K, _ = unsel.shape
        z = torch.zeros(B, 1, device=unsel.device)
        return z, torch.zeros(B, K, 1, device=unsel.device), z

    def compute_query(self, h_first, h_unsel, h_current):
        return torch.zeros(h_first.size(0), 1, device=h_first.device)

class NAMBankCVRP:

    def __init__(self, scorer):
        self.scorer = scorer
        self.heuristics = [{"index": 0, "fn": scorer}]
        self.M = 1
        self.num_heuristics = 1
        self.heuristic_temperature = 1.0
        self.N_problem = None

    def set_N(self, N):
        self.N_problem = N

    @torch.no_grad()
    def compute_probs(self, batch, device):
        unsel_locs = batch["unselected_locs"].float().to(device)
        unsel_dem = batch["unselected_demand"].float().to(device)
        cur_loc = batch["current_loc"].float().to(device)
        dep_loc = batch["depot_loc"].float().to(device)
        load = batch["load"].float().to(device)
        feas = batch["feasibility_mask"].bool().to(device)
        unsel_idx = batch["unselected_idx"].long().to(device)
        B, K = feas.shape
        N1 = self.N_problem + 1
        at_depot = (cur_loc == dep_loc).all(dim=-1)
        heu = self.scorer(unsel_locs, unsel_dem, cur_loc, dep_loc, load, feas).float()
        assert heu.shape == (B, K + 1) and torch.isfinite(heu[:, K]).all() and torch.isfinite(heu[:, :K][feas]).all()
        logits = torch.full((B, N1), float("-inf"), device=device)
        logits[:, 0] = heu[:, K]
        logits.scatter_(1, unsel_idx, heu[:, :K])
        logits[:, 0] = torch.where(at_depot, torch.full_like(logits[:, 0], float("-inf")), logits[:, 0])
        V_full = F.softmax(logits, dim=-1).unsqueeze(1)
        V_K = V_full.gather(2, unsel_idx.unsqueeze(1))
        return V_full, V_K

class NullRouterCVRP(nn.Module):

    def __init__(self):
        super().__init__()
        self.routing_temperature = 1.0
        self.num_heuristics = 1

    def forward(self, batch, V_full, V_K, return_scores=False):
        B = V_full.size(0)
        w = torch.ones(B, 1, device=V_full.device)
        if return_scores:
            return V_full[:, 0], w, torch.zeros(B, 1, device=V_full.device)
        return V_full[:, 0], w

