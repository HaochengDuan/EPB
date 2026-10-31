"""The start and route rules of the CVRP-50 band edit in one function (the rules-alone policy uses it with the
no-early-return rule). CVRP program signature (index 0 = depot), closed form, no search, no state.

Band frame of the customers: principal axis u and normal v (2x2 PCA), along-axis coordinate s, across offset t, depot
projection s0, depot offset h = |t_depot|, spacing sp = band length / N. Gate: PCA variance ratio < 0.05 and largest
gap along the axis < 15 x the median gap; otherwise every row is 0.
START (rows at the depot): far side = the side (sign of s - s0) of the unvisited customer farthest from the depot;
   depth = far side x (s - s0); the next stretch = the unvisited customers of the far side whose cumulative demand,
   counted from the outer end, fits one vehicle; with outer end b and inner end a, target = a if going out along one
   side and back along the other is cheaper (sqrt(h^2 + a^2) + LOOPC (b - a) < sqrt(h^2 + b^2)), else b;
   score_j = -|depth_j - target| / sp.
ROUTE (rows at a customer c with remaining load L): side = sign(s_c - s0), depth = side (s - s0); members = the
   unvisited customers of that side whose cumulative demand from the outer end fits L; outbound if a member lies more
   than sp / 2 farther out than c, else return.
   outbound: score_j = -d(c, j) + PROG clamp(depth_j - depth_c, -4 sp, 4 sp) + LANE sp [j on c's lane]
   return:   score_j = depth_j - KAPPA d(c, j)   (outermost first)
   non-members: - MEMB (outbound) / - MEMB_RET (return). Depot: the customers' minimum (outbound) or -KAPPA d(c, 0).
"""
import math, torch


def heuristic(locs, demand, current_node, mask, visited, load, first_node):
    START = 1; ROUTE = 1; LOOPC = 0.74; KAPPA = 0.5; PROG = 1.0; LANE = 1.0; MEMB = 1.0; MEMB_RET = 0.0; DEPOT = 1
    B, N1, _ = locs.shape; N = N1 - 1; device = locs.device
    cust = locs[:, 1:, :]
    m = cust.mean(1, keepdim=True); c = cust - m
    cxx = (c[..., 0] ** 2).mean(1); cyy = (c[..., 1] ** 2).mean(1); cxy = (c[..., 0] * c[..., 1]).mean(1)
    th = 0.5 * torch.atan2(2.0 * cxy, cxx - cyy)
    u = torch.stack([torch.cos(th), torch.sin(th)], -1); vn = torch.stack([-u[:, 1], u[:, 0]], -1)
    tr = cxx + cyy; disc = (tr ** 2 / 4.0 - (cxx * cyy - cxy ** 2)).clamp(min=0.0).sqrt()
    ratio = (tr / 2.0 - disc).clamp(min=0.0) / (tr / 2.0 + disc).clamp(min=1e-12)
    s = ((locs - m) * u.unsqueeze(1)).sum(-1); t = ((locs - m) * vn.unsqueeze(1)).sum(-1)
    ss = s[:, 1:].sort(dim=-1).values; gp = ss[:, 1:] - ss[:, :-1]
    gap_ok = gp.max(dim=-1).values < 15.0 * gp.median(dim=-1).values
    g = ((ratio < 0.05) & gap_ok).float()
    sp = ((ss[:, -1] - ss[:, 0]) / N).clamp(min=1e-6).unsqueeze(1)
    is_c = (torch.arange(N1, device=device) > 0)[None, :]
    feas = (mask > 0.5) & is_c
    unv = (visited < 0.5) & is_c
    cur = locs.gather(1, current_node.view(-1, 1, 1).expand(-1, 1, 2))
    d_c = torch.norm(locs - cur, dim=-1); d_0 = torch.norm(locs - locs[:, :1, :], dim=-1)
    at_dep = (current_node == 0)
    s0 = s[:, :1]; q = demand * unv.float()
    score = torch.zeros_like(d_0)
    if START:
        far_j = d_0.masked_fill(~unv, -1.0).argmax(1, keepdim=True)
        fside = torch.sign(s.gather(1, far_j) - s0); fside = torch.where(fside == 0, torch.ones_like(fside), fside)
        dep = fside * (s - s0)
        cum = ((dep.unsqueeze(1) >= dep.unsqueeze(2)).float() * (q * (dep > 0).float()).unsqueeze(1)).sum(-1)
        mem = unv & (dep > 0) & (cum <= 1.0 + 1e-6)
        b_out = dep.masked_fill(~mem, -1e9).amax(1, keepdim=True); a_in = dep.masked_fill(~mem, 1e9).amin(1, keepdim=True)
        has = mem.any(1, keepdim=True)
        b_out = torch.where(has, b_out, dep.masked_fill(~unv, -1e9).amax(1, keepdim=True)); a_in = torch.where(has, a_in, b_out)
        hh = t[:, :1].abs()
        loop = (torch.sqrt(hh ** 2 + a_in ** 2) + LOOPC * (b_out - a_in)) < torch.sqrt(hh ** 2 + b_out ** 2)
        target = torch.where(loop, a_in, b_out)
        st = -(dep - target).abs() / sp
        score = torch.where(at_dep[:, None] & feas, st, score)
    if ROUTE:
        side = torch.sign(s.gather(1, current_node.view(-1, 1)) - s0); side = torch.where(side == 0, torch.ones_like(side), side)
        depth = side * (s - s0); dcur = depth.gather(1, current_node.view(-1, 1))
        cum = ((depth.unsqueeze(1) >= depth.unsqueeze(2)).float() * (q * (depth > 0).float()).unsqueeze(1)).sum(-1)
        mem = unv & (depth > 0) & (cum <= load.float().unsqueeze(1) + 1e-6)
        outb = (mem & (depth > dcur + 0.5 * sp)).any(1, keepdim=True)
        lane_c = torch.sign(t.gather(1, current_node.view(-1, 1)))
        same = (torch.sign(t) == lane_c).float()
        o_out = -d_c + PROG * torch.maximum(torch.minimum(depth - dcur, 4.0 * sp), -4.0 * sp) + LANE * sp * same
        o_ret = depth - KAPPA * d_c
        o = torch.where(outb, o_out - MEMB * (~mem).float(), o_ret - MEMB_RET * (~mem).float())
        on = (~at_dep)[:, None] & feas
        nf = on.float().sum(1).clamp(min=1.0)
        omean = (o * on.float()).sum(1) / nf
        omin = o.masked_fill(~on, float("inf")).amin(1)
        dep_v = omean if DEPOT == 0 else torch.where(outb[:, 0], omin, -KAPPA * d_c[:, 0])
        o = torch.where(on, o, torch.zeros_like(o))
        o[:, 0] = torch.where((~at_dep) & on.any(1), dep_v, torch.zeros_like(dep_v))
        score = torch.where(at_dep[:, None], score, o)
    return score * g.unsqueeze(1)
