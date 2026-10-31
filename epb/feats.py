"""Feature primitives of the program-complexity budget (program_budget.py): one namespace per setting.

Under a budget tier a program may touch its raw inputs only by passing them to these functions, so the number of
distinct primitives it calls is exactly the number of concepts it uses. Every primitive is a batched torch op returning
a finite float tensor ([B, N] for pomo_tsp, [B, K] for any K >= 1 for lehd_tsp) on the device of its inputs.

Programs import the primitives by module name: `from tsp_feats import feat` (pomo_tsp) and `from cand_feats import feat`
(lehd_tsp; a name that does not reveal the teacher, used in the prompts). `lehd_feats`, the lehd_tsp name used by the
original trainers, is an alias of cand_feats, so banks written with it still load. Importing this module registers
all three names in sys.modules, so budget-tier banks compile without any other file on the path; anything that
compiles bank programs imports it first.
"""
import sys
import types

import torch


# ---- pomo_tsp: heuristic(locs, current_node, mask, visited, step_norm, first_node) -> [B, N] ---------------------

def _idx(locs, node):
    B = locs.shape[0]
    return locs[torch.arange(B, device=locs.device), node.long().view(B)]        # [B, 2]


def _pair_dist(locs):
    return torch.cdist(locs, locs)                                               # [B, N, N]


class PomoTspFeat:
    """Namespace of primitives. Each returns [B, N] float32."""

    @staticmethod
    def dist_cur(locs, current_node):
        """Euclidean distance from the current city to every city."""
        return torch.norm(locs - _idx(locs, current_node).unsqueeze(1), dim=-1)

    @staticmethod
    def dist_start(locs, first_node):
        """Euclidean distance from the tour's starting city to every city (cost of closing the tour there)."""
        return torch.norm(locs - _idx(locs, first_node).unsqueeze(1), dim=-1)

    @staticmethod
    def nn_dist(locs, mask):
        """Distance from every city to its nearest OTHER unvisited city (isolation / lookahead). 0 where no other unvisited city exists."""
        m = mask.bool()
        d = _pair_dist(locs)
        big = torch.full_like(d, 4.0)
        d = torch.where(m.unsqueeze(1), d, big)                                  # only unvisited columns count
        d = d + torch.eye(locs.shape[1], device=locs.device).unsqueeze(0) * 4.0  # exclude self
        out = d.min(dim=-1).values
        return torch.where(out >= 4.0, torch.zeros_like(out), out)

    @staticmethod
    def density(locs, mask, radius=0.1):
        """Number of OTHER unvisited cities within `radius` of every city."""
        m = mask.bool()
        d = _pair_dist(locs)
        within = (d < radius) & m.unsqueeze(1)
        within = within & ~torch.eye(locs.shape[1], dtype=torch.bool, device=locs.device).unsqueeze(0)
        return within.sum(dim=-1).float()

    @staticmethod
    def heading_cos(locs, current_node, visited):
        """Cosine of the angle between (city - current) and the current heading, where the heading is
        (current - centroid of visited cities). 0 when the heading is undefined (nothing visited yet)."""
        cur = _idx(locs, current_node)                                           # [B, 2]
        v = visited.float()
        n = v.sum(dim=1, keepdim=True).clamp_min(1.0)
        cen = (locs * v.unsqueeze(-1)).sum(dim=1) / n                            # [B, 2]
        head = cur - cen
        to = locs - cur.unsqueeze(1)                                             # [B, N, 2]
        hn = head.norm(dim=-1, keepdim=True)
        tn = to.norm(dim=-1)
        cos = (to * head.unsqueeze(1)).sum(-1) / (hn * tn).clamp_min(1e-9)
        return torch.where((hn > 1e-6) & (tn > 1e-9), cos, torch.zeros_like(cos))

    @staticmethod
    def outward(locs, mask):
        """Distance from every city to the centroid of the unvisited cities (large = on the outskirts)."""
        m = mask.float()
        n = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        cen = (locs * m.unsqueeze(-1)).sum(dim=1) / n
        return torch.norm(locs - cen.unsqueeze(1), dim=-1)

    @staticmethod
    def progress(step_norm, mask):
        """Fraction of the tour already built, in [0, 1], broadcast to [B, N]."""
        return step_norm.float().view(-1, 1).expand(-1, mask.shape[1]).clone()

    @staticmethod
    def n_unvisited(mask):
        """Number of unvisited cities, broadcast to [B, N]."""
        return mask.float().sum(dim=1, keepdim=True).expand(-1, mask.shape[1]).clone()


POMO_TSP_PRIMITIVES = {
    "dist_cur": "feat.dist_cur(locs, current_node) -> [B,N]  distance from the current city",
    "dist_start": "feat.dist_start(locs, first_node) -> [B,N]  distance from the tour's start city",
    "nn_dist": "feat.nn_dist(locs, mask) -> [B,N]  distance to the nearest other unvisited city (isolation)",
    "density": "feat.density(locs, mask, radius=0.1) -> [B,N]  number of other unvisited cities within radius",
    "heading_cos": "feat.heading_cos(locs, current_node, visited) -> [B,N]  cosine to the current heading (+1 straight ahead, -1 backwards)",
    "outward": "feat.outward(locs, mask) -> [B,N]  distance to the centroid of the unvisited cities",
    "progress": "feat.progress(step_norm, mask) -> [B,N]  fraction of the tour completed",
    "n_unvisited": "feat.n_unvisited(mask) -> [B,N]  number of unvisited cities",
}


# ---- lehd_tsp: heuristic(unselected_locs [B,K,2], first_loc [B,2], current_loc [B,2]) -> [B, K] ------------------
# No visited set and no step index: heading_cos / progress are replaced by homeward_cos (alignment with the direction
# back to the start city), and n_unvisited is K.

class LehdTspFeat:
    """Namespace of primitives. Each returns [B, K] float32."""

    @staticmethod
    def dist_cur(unselected_locs, current_loc):
        """Euclidean distance from the current city to every candidate."""
        return torch.norm(unselected_locs - current_loc.unsqueeze(1), dim=-1)

    @staticmethod
    def dist_start(unselected_locs, first_loc):
        """Euclidean distance from the tour's starting city to every candidate (cost of closing the tour there)."""
        return torch.norm(unselected_locs - first_loc.unsqueeze(1), dim=-1)

    @staticmethod
    def nn_dist(unselected_locs):
        """Distance from every candidate to its nearest OTHER candidate (isolation / lookahead). 0 when K = 1."""
        K = unselected_locs.shape[1]
        d = torch.cdist(unselected_locs, unselected_locs)
        d = d + torch.eye(K, device=unselected_locs.device).unsqueeze(0) * 4.0     # exclude self
        out = d.min(dim=-1).values
        return torch.where(out >= 4.0, torch.zeros_like(out), out)

    @staticmethod
    def density(unselected_locs, radius=0.1):
        """Number of OTHER candidates within `radius` of every candidate."""
        K = unselected_locs.shape[1]
        d = torch.cdist(unselected_locs, unselected_locs)
        within = (d < radius) & ~torch.eye(K, dtype=torch.bool, device=unselected_locs.device).unsqueeze(0)
        return within.sum(dim=-1).float()

    @staticmethod
    def homeward_cos(unselected_locs, first_loc, current_loc):
        """Cosine of the angle between (candidate - current) and (start - current): +1 = straight back
        toward the start city, -1 = directly away from it. 0 when current == start."""
        home = first_loc - current_loc                                             # [B, 2]
        to = unselected_locs - current_loc.unsqueeze(1)                             # [B, K, 2]
        hn = home.norm(dim=-1, keepdim=True)
        tn = to.norm(dim=-1)
        cos = (to * home.unsqueeze(1)).sum(-1) / (hn * tn).clamp_min(1e-9)
        return torch.where((hn > 1e-6) & (tn > 1e-9), cos, torch.zeros_like(cos))

    @staticmethod
    def outward(unselected_locs):
        """Distance from every candidate to the centroid of the candidates (large = on the outskirts)."""
        cen = unselected_locs.mean(dim=1, keepdim=True)
        return torch.norm(unselected_locs - cen, dim=-1)

    @staticmethod
    def n_unvisited(unselected_locs):
        """Number of candidates K, broadcast to [B, K]."""
        B, K = unselected_locs.shape[:2]
        return torch.full((B, K), float(K), device=unselected_locs.device)


LEHD_TSP_PRIMITIVES = {
    "dist_cur": "feat.dist_cur(unselected_locs, current_loc) -> [B,K]  distance from the current city",
    "dist_start": "feat.dist_start(unselected_locs, first_loc) -> [B,K]  distance from the tour's start city",
    "nn_dist": "feat.nn_dist(unselected_locs) -> [B,K]  distance to the nearest other candidate (isolation)",
    "density": "feat.density(unselected_locs, radius=0.1) -> [B,K]  number of other candidates within radius",
    "homeward_cos": "feat.homeward_cos(unselected_locs, first_loc, current_loc) -> [B,K]  cosine to the direction back to the start (+1 toward, -1 away)",
    "outward": "feat.outward(unselected_locs) -> [B,K]  distance to the centroid of the candidates",
    "n_unvisited": "feat.n_unvisited(unselected_locs) -> [B,K]  number of candidates left",
}


# ---- registry ---------------------------------------------------------------------------------------------------
MODULES = {"pomo_tsp": "tsp_feats", "lehd_tsp": "cand_feats"}          # setting -> module name of the prompts
ALIASES = {"lehd_tsp": ("lehd_feats",)}                                  # setting -> older names programs may import
FEAT = {"pomo_tsp": PomoTspFeat, "lehd_tsp": LehdTspFeat}
PRIMITIVES = {"pomo_tsp": POMO_TSP_PRIMITIVES, "lehd_tsp": LEHD_TSP_PRIMITIVES}


def import_names(setting: str) -> tuple:
    """The module names a program of `setting` may import the primitives from (the prompts' name first)."""
    return (MODULES[setting],) + ALIASES.get(setting, ())


def _register() -> None:
    for setting in MODULES:
        for name in import_names(setting):
            if name not in sys.modules:
                mod = types.ModuleType(name, f"Budget feature primitives of {setting} (epb.feats).")
                mod.feat, mod.PRIMITIVES = FEAT[setting], PRIMITIVES[setting]
                sys.modules[name] = mod


_register()
