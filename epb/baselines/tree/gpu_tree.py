from __future__ import annotations

import numpy as np
import torch

class GPUTree:

    def __init__(self, sk_tree, device):
        t = sk_tree.tree_
        self.left = torch.as_tensor(t.children_left, dtype=torch.long, device=device)
        self.right = torch.as_tensor(t.children_right, dtype=torch.long, device=device)
        self.feature = torch.as_tensor(t.feature, dtype=torch.long, device=device)
        self.threshold = torch.as_tensor(t.threshold, dtype=torch.float32, device=device)
        val = t.value
        if val.shape[-1] == 2:
            p1 = val[:, 0, 1] / np.clip(val[:, 0, :].sum(-1), 1e-12, None)
        else:
            cls = sk_tree.classes_
            p1 = np.full(val.shape[0], 1.0 if cls[0] == 1 else 0.0)
        self.leaf_p = torch.as_tensor(p1, dtype=torch.float32, device=device)
        self.is_leaf = self.feature < 0
        self.max_depth = int(t.max_depth)
        self.feat_safe = self.feature.clamp(min=0)

    @torch.no_grad()
    def predict_proba1(self, X: torch.Tensor) -> torch.Tensor:
        node = torch.zeros(X.size(0), dtype=torch.long, device=X.device)
        for _ in range(self.max_depth):
            leaf = self.is_leaf[node]
            if bool(leaf.all()):
                break
            f = self.feat_safe[node]
            x = X.gather(1, f.unsqueeze(1)).squeeze(1)
            go_left = x <= self.threshold[node]
            nxt = torch.where(go_left, self.left[node], self.right[node])
            node = torch.where(leaf, node, nxt)
        return self.leaf_p[node]

class TreePolicy:

    def __init__(self, name, trees, device, stage_feat_idx=None, stage_thresholds=None,
                 stage_desc=""):
        self.name = name
        self.trees = [GPUTree(t, device) for t in trees]
        self.stage_feat_idx = stage_feat_idx
        self.stage_thresholds = stage_thresholds
        self.stage_desc = stage_desc
        self.staged = len(self.trees) > 1

    @torch.no_grad()
    def score(self, feats: torch.Tensor) -> torch.Tensor:
        B, C, F = feats.shape
        X = feats.reshape(-1, F)
        if not self.staged:
            return self.trees[0].predict_proba1(X).view(B, C)
        v = X[:, self.stage_feat_idx]
        lo, hi = self.stage_thresholds
        stage = torch.where(v <= lo, 0, torch.where(v <= hi, 1, 2))
        out = torch.empty(X.size(0), dtype=torch.float32, device=X.device)
        for s, tree in enumerate(self.trees):
            m = stage == s
            if bool(m.any()):
                out[m] = tree.predict_proba1(X[m])
        return out.view(B, C)

@torch.no_grad()
def masked_argmax(scores: torch.Tensor, feas: torch.Tensor) -> torch.Tensor:
    s = scores.masked_fill(~feas, -float("inf"))
    return s.argmax(dim=-1)
