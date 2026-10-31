import math
import torch

def heuristic(locs, current_node, mask, visited, step_norm, first_node):
    """The two-lane band rule as one finite penalty in units of the median nearest-neighbour spacing: A, a node on
    the return side of the band on the outward leg; B, a node behind the direction of travel; C, a node whose choice
    skips unvisited nodes on the way back to the start. Gated by the anisotropy of the instance (0 on isotropic
    instances) and switched off in the endgame. The paper gives this rule to the teacher (POMO + same rule)."""
    B, N, _ = locs.shape
    _c = locs - locs.mean(1, keepdim=True); _cxx = (_c[..., 0] ** 2).mean(1); _cyy = (_c[..., 1] ** 2).mean(1); _cxy = (_c[..., 0] * _c[..., 1]).mean(1)
    _th = 0.5 * torch.atan2(2.0 * _cxy, _cxx - _cyy); _u = torch.stack([torch.cos(_th), torch.sin(_th)], -1); _v = torch.stack([-_u[:, 1], _u[:, 0]], -1)
    _tr = _cxx + _cyy; _disc = (_tr ** 2 / 4.0 - (_cxx * _cyy - _cxy ** 2)).clamp(min=0).sqrt()
    _aniso = torch.clamp((2.0 * _disc / _tr.clamp(min=1e-9) - 0.3) / 0.4, 0.0, 1.0).unsqueeze(1)                     # 0 isotropic .. 1 thin band
    _s = (locs * _u.unsqueeze(1)).sum(-1); _t = (locs * _v.unsqueeze(1)).sum(-1); _t = _t - _t.mean(1, keepdim=True)  # along / across the band
    _s_cur = _s.gather(1, current_node.view(B, 1)); _s_start = _s.gather(1, first_node.view(B, 1))
    _dd = torch.norm(locs.unsqueeze(2) - locs.unsqueeze(1), dim=-1) + torch.eye(N, device=locs.device).unsqueeze(0) * 9.0
    _med = _dd.min(-1).values.median(-1, keepdim=True).values.clamp(min=1e-4)                                       # median NN spacing
    _head = torch.sign(_s_cur - _s_start + 1e-6); _unv = mask.float()
    _out = torch.clamp((_unv * (((_s - _s_cur) * _head) > 0.5 * _med).float()).sum(-1, keepdim=True) / 3.0, 0.0, 1.0)   # 1 outward leg, 0 return leg
    _late = torch.sigmoid(20.0 * (step_norm.view(B, 1) - 0.85))
    _ruleA = _aniso * (1.0 - _late) * _out * torch.relu(-_head * _t) / _med
    _dir = _head * (2.0 * _out - 1.0)
    _ruleB = _aniso * (1.0 - _late) * torch.clamp(torch.relu(-(_s - _s_cur) * _dir) / _med, max=4.0)
    _ahead = ((_s - _s_cur) * (-_head) > 0.5 * _med).float() * _unv
    _skip = (_ahead.unsqueeze(1) * (((_s.unsqueeze(1) - _s.unsqueeze(2)) * (-_head).unsqueeze(1)) < -0.5 * _med.unsqueeze(1)).float()).sum(-1)
    _ruleC = _aniso * (1.0 - _late) * (1.0 - _out) * torch.clamp(_skip, max=4.0)
    penalty = _ruleA + _ruleB + _ruleC
    return (-penalty).masked_fill(~mask.bool(), 0.0)
