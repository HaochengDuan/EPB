"""LEHD teacher on the CVRP: the model loader, the K-bucketed decision-step generator and the teacher's decoding for
the evaluation. The model code is NCO_code's LEHD.CVRP.VRPModel, imported from a clone of
https://github.com/CIAM-Group/NCO_code (--nco-root, or NCO_ROOT) and patched to be device-aware (`import_vrp_model`).

  python -m epb.teachers.lehd_cvrp --checkpoint <LEHD_CVRP_CKPT> --nco-root <NCO_code clone> --problem-size 100 \
      --num-instances 275000 --batch-size 512 --target-per-bucket 4096 --seed 201 --teacher-capacity-rule \
      --output lehd_cvrp100_steps

LEHD's decoder output is [B, 2N]: positions 0..N-1 are the probabilities of going to a customer directly, N..2N-1 of
going there via the depot (return, refill, then visit). The generator maps it to the action space [depot,
customers] (A = N + 1): p(depot) = the via-depot mass, p(customer i) = the direct probability. When the teacher's
argmax is a via-depot move, it writes two rows: the state at the current node (target: depot) and the state at the
depot (target: the customer, with the renormalised via-depot probabilities). Rows with no feasible customer (a forced
return) are skipped. Output: <output>/manifest.pt ({"format": "lehd_cvrp_kbucket_v1", "N", "n_instances",
"total_rows", "per_K", "history"}) and <output>/bucket_K{K:03d}_shard_{S:04d}.pt, one bucket per number K of
unvisited customers.

Capacity rule (--teacher-capacity-rule): LEHD's decoder does not mask by capacity; the teacher's environment refills
only when the remaining capacity is smaller than the demand, so with its integer demands an exact fill is feasible.
With demands k / capacity in float32 the remaining load of an exact fill can round just below the demand. The flag
applies the teacher's rule with the tolerance CAPACITY_TOL (demand <= load + 1e-6 fits, the rule of the evaluation)
to the stored feasibility mask, the forced-return test and the transition. Without it (the data of the paper's runs)
the mask and the transition use no tolerance, so some rows mark an exact-fill customer infeasible.

Evaluation (epb.settings.lehd_cvrp): lehd_cvrp_greedy (one-pass greedy cost), lehd_greedy_order (the customer order
of the greedy rollout), lehd_constrained_replay (an order replayed with the teacher deciding when to return to the
depot) and lehd_restricted_greedy (the greedy order of a sub-problem, the repair of the random re-construct).
"""

from __future__ import annotations

import argparse
import datetime
import os
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

from . import add_nco_path

MODEL_PARAMS = {
    "mode": "test", "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** 0.5,
    "decoder_layer_num": 6, "qkv_dim": 16,
    "head_num": 8, "ff_hidden_dim": 512,
}
CAPACITY_TOL = 1e-6   # --teacher-capacity-rule: demand <= load + CAPACITY_TOL fits (the evaluation's rule)


# ─────────────────────────────────────────────────────────────────────────────
# Model: NCO_code's VRPModel with two device fixes of CVRP_Decoder
# ─────────────────────────────────────────────────────────────────────────────

def _patched_decoder_forward(self, data, selected_node_list, capacity, remaining_capacity):
    """CVRP_Decoder.forward with `new_props` created on the device of its inputs (upstream: on the CPU)."""
    data_ = data[:, 1:, :].clone().detach()
    selected_node_list_ = selected_node_list.clone().detach() - 1
    device = data.device

    batch_size_V = data_.shape[0]
    problem_size = data_.shape[1]
    new_data = data_.clone().detach()

    left_encoded_node = self._get_new_data(new_data, selected_node_list_, problem_size, batch_size_V)

    embedded_first_node = data[:, [0], :]
    if selected_node_list_.shape[1] == 0:
        embedded_last_node = data[:, [0], :]
    else:
        embedded_last_node = self._get_encoding(new_data, selected_node_list_[:, [-1]])

    remaining_capacity = remaining_capacity.reshape(batch_size_V, 1, 1) / capacity
    first_node_cat = torch.cat((embedded_first_node, remaining_capacity), dim=2)
    last_node_cat = torch.cat((embedded_last_node, remaining_capacity), dim=2)

    embedded_first_node_ = self.embedding_first_node(first_node_cat)
    embedded_last_node_ = self.embedding_last_node(last_node_cat)

    embeded_all = torch.cat((embedded_first_node_, left_encoded_node, embedded_last_node_), dim=1)
    out = embeded_all
    for layer in self.layers:
        out = layer(out)

    out = self.Linear_final(out)
    out[:, [0, -1], :] = out[:, [0, -1], :] + float("-inf")
    out = torch.cat((out[:, :, 0], out[:, :, 1]), dim=1)
    props = F.softmax(out, dim=-1)
    customer_num = left_encoded_node.shape[1]
    props = torch.cat((props[:, 1:customer_num + 1], props[:, customer_num + 1 + 1 + 1:-1]), dim=1)

    index_small = torch.le(props, 1e-5)
    props_clone = props.clone()
    props_clone[index_small] = props_clone[index_small] + torch.tensor(
        1e-7, dtype=props_clone[index_small].dtype, device=device)
    props = props_clone

    new_props = torch.zeros(batch_size_V, 2 * (problem_size), device=device)

    index_1_ = torch.arange(batch_size_V, dtype=torch.long, device=device)[:, None].repeat(
        1, selected_node_list_.shape[1] * 2)
    index_2_ = torch.cat(
        ((selected_node_list_).type(torch.long),
         (problem_size) + (selected_node_list_).type(torch.long)), dim=-1
    )
    new_props[index_1_, index_2_,] = -2
    index = torch.gt(new_props, -1).view(batch_size_V, -1)
    new_props[index] = props.ravel()

    return new_props


def _patched_get_new_data(self, data, selected_node_list, prob_size, B_V):
    """CVRP_Decoder._get_new_data with its index tensors on the device of `data`."""
    device = data.device
    new_list = torch.arange(prob_size, device=device)[None, :].repeat(B_V, 1)
    new_list_len = prob_size - selected_node_list.shape[1]
    index_2 = selected_node_list.type(torch.long)
    index_1 = torch.arange(B_V, dtype=torch.long, device=device)[:, None].expand(
        B_V, index_2.shape[1])
    new_list[index_1, index_2] = -2
    unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)

    new_data = data
    emb_dim = data.shape[-1]
    new_data_len = new_list_len
    index_2_ = unselect_list.repeat_interleave(repeats=emb_dim, dim=1)
    index_1_ = torch.arange(B_V, dtype=torch.long, device=device)[:, None].expand(
        B_V, index_2_.shape[1])
    index_3_ = torch.arange(emb_dim, device=device)[None, :].repeat(repeats=(B_V, new_data_len))
    new_data_ = new_data[index_1_, index_2_, index_3_].view(B_V, new_data_len, emb_dim)
    return new_data_


def import_vrp_model(nco_root: Optional[str] = None):
    """NCO_code's VRPModel class, with CVRP_Decoder.forward and ._get_new_data replaced by the device-aware versions
    above (the class is patched in place, once)."""
    add_nco_path(nco_root)
    from LEHD.CVRP.VRPModel import VRPModel, CVRP_Decoder
    CVRP_Decoder.forward = _patched_decoder_forward
    CVRP_Decoder._get_new_data = _patched_get_new_data
    return VRPModel


def load_lehd_cvrp(ckpt_path: str, device, nco_root: Optional[str] = None):
    """The LEHD-CVRP teacher (test mode) with the weights of ckpt_path, in eval mode."""
    VRPModel = import_vrp_model(nco_root)
    model = VRPModel(**MODEL_PARAMS).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Teacher decoding for the evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def lehd_cvrp_greedy(model, depot_xy, node_xy, node_demand,
                     capacity: float = 1.0, batch_size: int = 32):
    """LEHD-CVRP greedy rollout, one customer per step in lockstep over the batch; returns the cost of the physical
    sequences [B]. selected_node_list holds the customers only (1-indexed); a depot return is recorded in the physical
    sequence. If the argmax is a direct move but the demand exceeds the load (+1e-6), the move goes via the depot."""
    device = node_xy.device
    B, N, _ = node_xy.shape
    costs = torch.zeros(B, device=device)
    for s in range(0, B, batch_size):
        e = min(s + batch_size, B)
        d  = depot_xy[s:e].to(device)
        nx = node_xy[s:e].to(device)
        nd = node_demand[s:e].to(device)
        bs = e - s
        all_locs   = torch.cat([d, nx], dim=1)                                 # [bs, N+1, 2]
        all_demand = torch.cat([torch.zeros(bs, 1, device=device), nd], dim=1)
        cap_col = torch.full((bs, N + 1), capacity, device=device)
        problems = torch.cat([all_locs, all_demand.unsqueeze(-1),
                              cap_col.unsqueeze(-1)], dim=-1)                  # [bs, N+1, 4]

        model.capacity = capacity
        encoded = model.encoder(problems, capacity)                            # [bs, N+1, D]
        selected_node_list = torch.zeros(bs, 0, dtype=torch.long, device=device)
        load = torch.full((bs,), capacity, device=device)
        seqs: List[List[int]] = [[0] for _ in range(bs)]

        for step in range(N):
            problems[:, :, 3] = load.unsqueeze(-1)
            remaining = problems[:, 1, 3]                                      # [bs]
            probs_2N = model.decoder(encoded, selected_node_list, capacity, remaining)
            argmax_2N = probs_2N.argmax(dim=-1)                                # [bs]
            sel_cust = argmax_2N % N + 1                                       # 1-indexed
            is_via   = (argmax_2N >= N)
            sel_dem  = nd.gather(1, (sel_cust - 1).unsqueeze(-1)).squeeze(-1)
            need_refill = (~is_via) & (sel_dem > load + 1e-6)
            is_via = is_via | need_refill

            # per-instance update of the physical sequence and the load
            sel_cust_cpu = sel_cust.tolist()
            is_via_cpu   = is_via.tolist()
            sel_dem_cpu  = sel_dem.tolist()
            for i in range(bs):
                if is_via_cpu[i]:
                    seqs[i].append(0)
                    load[i] = capacity
                seqs[i].append(int(sel_cust_cpu[i]))
                load[i] = load[i] - sel_dem[i]

            selected_node_list = torch.cat(
                [selected_node_list, sel_cust.unsqueeze(-1)], dim=1)

        for i in range(bs):
            if seqs[i][-1] != 0:
                seqs[i].append(0)
        for i in range(bs):
            idx_t = torch.tensor(seqs[i], device=device, dtype=torch.long)
            pts = all_locs[i].index_select(0, idx_t)
            costs[s + i] = (pts[1:] - pts[:-1]).norm(dim=-1).sum()
    return costs


@torch.no_grad()
def lehd_greedy_order(
    model, depot_xy, node_xy, node_demand,
    capacity: float = 1.0, batch_size: int = 64, device=None,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
    """The greedy rollout of lehd_cvrp_greedy, also returning the order: (order [B, M] customer ids, 1-indexed in the
    given problem; costs [B] of the physical sequences; the physical sequences, 0 = depot)."""
    if device is None:
        device = node_xy.device
    B, M, _ = node_xy.shape
    costs = torch.zeros(B, device=device)
    order = torch.zeros(B, M, dtype=torch.long, device=device)
    seqs_all: List[List[int]] = [None] * B  # type: ignore
    for s in range(0, B, batch_size):
        e = min(s + batch_size, B)
        bs = e - s
        d = depot_xy[s:e].to(device)
        nx = node_xy[s:e].to(device)
        nd = node_demand[s:e].to(device)
        all_locs = torch.cat([d, nx], dim=1)                                   # [bs, M+1, 2]
        all_demand = torch.cat([torch.zeros(bs, 1, device=device), nd], dim=1)
        cap_col = torch.full((bs, M + 1), capacity, device=device)
        problems = torch.cat([all_locs, all_demand.unsqueeze(-1),
                              cap_col.unsqueeze(-1)], dim=-1)                   # [bs, M+1, 4]
        model.capacity = capacity
        encoded = model.encoder(problems, capacity)
        selected = torch.zeros(bs, 0, dtype=torch.long, device=device)
        load = torch.full((bs,), capacity, device=device)
        seqs: List[List[int]] = [[0] for _ in range(bs)]
        for step in range(M):
            problems[:, :, 3] = load.unsqueeze(-1)
            remaining = problems[:, 1, 3]
            probs = model.decoder(encoded, selected, capacity, remaining)      # [bs, 2M]
            probs = probs.to(device)
            argmax = probs.argmax(dim=-1)
            sel_cust = argmax % M + 1                                          # 1-indexed
            is_via = argmax >= M
            sel_dem = nd.gather(1, (sel_cust - 1).unsqueeze(-1)).squeeze(-1)
            need_refill = (~is_via) & (sel_dem > load + 1e-6)
            is_via = is_via | need_refill
            # vectorised load update (no per-element device writes in the Python loop)
            load = torch.where(is_via, torch.full_like(load, capacity), load) - sel_dem
            sc = sel_cust.tolist(); iv = is_via.tolist()
            for i in range(bs):
                if iv[i]:
                    seqs[i].append(0)
                seqs[i].append(int(sc[i]))
            order[s:e, step] = sel_cust
            selected = torch.cat([selected, sel_cust.unsqueeze(-1)], dim=1)
        for i in range(bs):
            if seqs[i][-1] != 0:
                seqs[i].append(0)
            idx_t = torch.tensor(seqs[i], device=device, dtype=torch.long)
            pts = all_locs[i].index_select(0, idx_t)
            costs[s + i] = (pts[1:] - pts[:-1]).norm(dim=-1).sum()
            seqs_all[s + i] = seqs[i]
    return order, costs, seqs_all


@torch.no_grad()
def lehd_restricted_greedy(
    model, locs_full: torch.Tensor, dem_full: torch.Tensor,
    restrict_to: torch.Tensor, capacity: float = 1.0,
    batch_size: int = 64, device=None,
) -> torch.Tensor:
    """Re-order the customers restrict_to [B, L] by solving them as an L-customer sub-problem (same depot) with the
    teacher's greedy rollout. Returns the new order [B, L] in full-problem customer ids."""
    if device is None:
        device = locs_full.device
    B, L = restrict_to.shape
    bidx = torch.arange(B, device=device).unsqueeze(1)
    sub_depot = locs_full[:, 0:1, :]                                           # [B,1,2]
    sub_nodes = locs_full[bidx, restrict_to]                                   # [B,L,2]
    sub_dem = dem_full[bidx, restrict_to]                                      # [B,L]
    order_local, _, _ = lehd_greedy_order(
        model, sub_depot, sub_nodes, sub_dem,
        capacity=capacity, batch_size=batch_size, device=device)              # [B,L] in 1..L
    new_sub = restrict_to.gather(1, order_local - 1)                          # back to full ids
    return new_sub


@torch.no_grad()
def lehd_constrained_replay(
    model, locs_full: torch.Tensor, dem_full: torch.Tensor,
    perm: torch.Tensor, capacity: float = 1.0,
    batch_size: int = 64, device=None,
) -> Tuple[torch.Tensor, List[List[int]]]:
    """Replay the customer order perm [B, N]; at every step the teacher decides whether to go via the depot (via
    probability > direct probability, or forced when demand > load + 1e-6). Returns (costs [B] of the physical
    sequences, the physical sequences)."""
    if device is None:
        device = locs_full.device
    B, N1, _ = locs_full.shape
    N = N1 - 1
    costs = torch.zeros(B, device=device)
    seqs_all: List[List[int]] = [None] * B  # type: ignore
    for s in range(0, B, batch_size):
        e = min(s + batch_size, B)
        bs = e - s
        d = locs_full[s:e, 0:1, :].to(device)
        nx = locs_full[s:e, 1:, :].to(device)
        nd = dem_full[s:e, 1:].to(device)
        pm = perm[s:e].to(device)                                             # [bs, N] 1-indexed
        all_locs = torch.cat([d, nx], dim=1)
        all_demand = torch.cat([torch.zeros(bs, 1, device=device), nd], dim=1)
        cap_col = torch.full((bs, N + 1), capacity, device=device)
        problems = torch.cat([all_locs, all_demand.unsqueeze(-1),
                              cap_col.unsqueeze(-1)], dim=-1)
        model.capacity = capacity
        encoded = model.encoder(problems, capacity)
        selected = torch.zeros(bs, 0, dtype=torch.long, device=device)
        load = torch.full((bs,), capacity, device=device)
        seqs: List[List[int]] = [[0] for _ in range(bs)]
        for step in range(N):
            problems[:, :, 3] = load.unsqueeze(-1)
            remaining = problems[:, 1, 3]
            probs = model.decoder(encoded, selected, capacity, remaining).to(device)  # [bs,2N]
            tgt = pm[:, step]                                                 # [bs] 1..N
            direct = probs.gather(1, (tgt - 1).unsqueeze(-1)).squeeze(-1)
            via = probs.gather(1, (N + tgt - 1).unsqueeze(-1)).squeeze(-1)
            sel_dem = nd.gather(1, (tgt - 1).unsqueeze(-1)).squeeze(-1)
            need_refill = sel_dem > load + 1e-6
            is_via = (via > direct) | need_refill
            load = torch.where(is_via, torch.full_like(load, capacity), load) - sel_dem
            iv = is_via.tolist(); tg = tgt.tolist()
            for i in range(bs):
                if iv[i]:
                    seqs[i].append(0)
                seqs[i].append(int(tg[i]))
            selected = torch.cat([selected, tgt.unsqueeze(-1)], dim=1)
        for i in range(bs):
            if seqs[i][-1] != 0:
                seqs[i].append(0)
            idx_t = torch.tensor(seqs[i], device=device, dtype=torch.long)
            pts = all_locs[i].index_select(0, idx_t)
            costs[s + i] = (pts[1:] - pts[:-1]).norm(dim=-1).sum()
            seqs_all[s + i] = seqs[i]
    return costs, seqs_all


# ─────────────────────────────────────────────────────────────────────────────
# Decision steps
# ─────────────────────────────────────────────────────────────────────────────

class KBucketWriter:
    """Rows go to per-K in-memory buffers; a buffer is written as bucket_K{K:03d}_shard_{S:04d}.pt when it holds
    target_per_bucket rows (and by flush_all)."""

    def __init__(self, output_dir: str, target_per_bucket: int):
        self.output_dir = output_dir
        self.target_per_bucket = target_per_bucket
        os.makedirs(output_dir, exist_ok=True)
        self._buf: Dict[int, List[Dict[str, torch.Tensor]]] = {}
        self._shard_idx: Dict[int, int] = {}
        self._n_rows: Dict[int, int] = {}
        self._shard_files: Dict[int, List[str]] = {}

    def add_row(self, K: int, row: Dict[str, torch.Tensor]) -> None:
        if K not in self._buf:
            self._buf[K] = []
            self._shard_idx[K] = 0
            self._n_rows[K] = 0
            self._shard_files[K] = []
        self._buf[K].append(row)
        if len(self._buf[K]) >= self.target_per_bucket:
            self._flush_K(K)

    def _flush_K(self, K: int) -> None:
        rows = self._buf[K]
        if not rows:
            return
        n = len(rows)
        stacked: Dict[str, torch.Tensor] = {}
        for key in rows[0]:
            stacked[key] = torch.stack([r[key] for r in rows], dim=0)

        s = self._shard_idx[K]
        fname = f"bucket_K{K:03d}_shard_{s:04d}.pt"
        path = os.path.join(self.output_dir, fname)
        torch.save(stacked, path)
        self._shard_files[K].append(fname)
        self._n_rows[K] += n
        self._shard_idx[K] += 1
        self._buf[K] = []

    def flush_all(self) -> None:
        for K in list(self._buf.keys()):
            self._flush_K(K)

    def manifest(self, *, N: int, n_instances: int, pomo_size: int,
                 history_entry: Dict) -> Dict:
        per_K = {
            K: {"shards": self._shard_files[K], "n_rows": self._n_rows[K]}
            for K in sorted(self._n_rows.keys())
        }
        total_rows = sum(v["n_rows"] for v in per_K.values())
        return {
            "format": "lehd_cvrp_kbucket_v1",
            "N": N,
            "n_instances": n_instances,
            "pomo_size": pomo_size,
            "total_rows": total_rows,
            "per_K": per_K,
            "history": [history_entry],
        }


def get_random_problems(batch_size: int, problem_size: int):
    """Uniform depot and customers, integer demands 1..9 / capacity (POMO-CVRP's distribution), from the global
    torch stream."""
    depot_xy = torch.rand(batch_size, 1, 2)
    node_xy  = torch.rand(batch_size, problem_size, 2)
    demand_scaler = {20: 30, 50: 40, 100: 50}.get(problem_size)
    if demand_scaler is None:
        raise NotImplementedError(f"No demand_scaler for problem_size={problem_size}")
    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    return depot_xy, node_xy, node_demand


def _build_problems(depot_xy, node_xy, node_demand, capacity: float = 1.0):
    """LEHD's [B, N+1, 4] problems tensor: x, y, demand, capacity."""
    B = depot_xy.size(0)
    all_locs = torch.cat([depot_xy, node_xy], dim=1)                       # [B, N+1, 2]
    all_demand = torch.cat(
        [torch.zeros(B, 1, device=depot_xy.device), node_demand], dim=1
    )                                                                       # [B, N+1]
    cap_col = torch.full((B, all_locs.size(1)), capacity, device=depot_xy.device)
    problems = torch.cat([all_locs, all_demand.unsqueeze(-1),
                            cap_col.unsqueeze(-1)], dim=-1)                # [B, N+1, 4]
    return problems


@torch.no_grad()
def rollout_and_bucket(
    model,
    num_instances: int,
    problem_size: int,
    batch_size: int,
    device: torch.device,
    writer: KBucketWriter,
    instance_id_start: int = 0,
    teacher_capacity_rule: bool = False,
) -> Dict:
    """Greedy LEHD-CVRP rollouts on fresh instances, one row per physical decision step (two for a via-depot move:
    state at the current node -> depot, state at the depot -> customer), written to `writer` by K.
    teacher_capacity_rule: a customer fits iff demand <= load + CAPACITY_TOL in the stored mask, the forced-return
    test and the transition (otherwise no tolerance)."""
    tol = CAPACITY_TOL if teacher_capacity_rule else 0.0
    model.eval()
    capacity = 1.0
    N = problem_size
    # decoder output [B, 2N]: 0..N-1 direct to customer 1..N, N..2N-1 via the depot (upstream VRPModel.forward splits
    # at N - 1, which is off by one)
    split_line = N

    total_rows = 0
    total_skipped_forced_depot = 0
    total_steps_simulated = 0
    total_generated = 0
    next_iid = instance_id_start

    while total_generated < num_instances:
        bs = min(batch_size, num_instances - total_generated)

        depot_xy, node_xy, node_demand = get_random_problems(bs, N)
        depot_xy = depot_xy.to(device); node_xy = node_xy.to(device); node_demand = node_demand.to(device)
        problems = _build_problems(depot_xy, node_xy, node_demand, capacity=capacity)
        all_locs_cpu = problems[:, :, :2].cpu()
        all_demand_cpu = problems[:, :, 2].cpu()

        model.capacity = capacity
        encoded_nodes = model.encoder(problems, capacity)                  # [B, N+1, D]

        selected_node_list = torch.zeros(bs, 0, dtype=torch.long, device=device)  # customers (1-indexed), no depot
        load = torch.full((bs,), capacity, device=device)
        visited_cust = torch.zeros(bs, N, dtype=torch.bool, device=device)  # over customers, 0-indexed
        instance_ids = torch.arange(next_iid, next_iid + bs, dtype=torch.int32)
        next_iid += bs

        current_node = torch.zeros(bs, dtype=torch.long, device=device)    # 0 = depot, 1..N = customer

        for step in range(N):  # exactly N customer visits per trajectory
            problems[:, :, 3] = load.unsqueeze(-1)
            remaining_capacity = problems[:, 1, 3]   # [B]
            probs_2N = model.decoder(encoded_nodes, selected_node_list, capacity, remaining_capacity)
            # one device-to-host copy per step; the per-instance loop below indexes CPU tensors only
            probs_2N_cpu     = probs_2N.cpu()                              # [B, 2N]
            direct_probs     = probs_2N_cpu[:, :N]                         # [B, N]
            via_depot_probs  = probs_2N_cpu[:, N:]                         # [B, N]
            argmax_full_cpu  = probs_2N_cpu.argmax(dim=-1)                 # [B]
            current_node_cpu = current_node.cpu()                         # [B]
            load_cpu         = load.cpu()                                 # [B]
            visited_cust_cpu = visited_cust.cpu()                         # [B, N]

            for b in range(bs):
                cur_b = int(current_node_cpu[b])
                unv_b = ~visited_cust_cpu[b]                                # [N]
                K_b = int(unv_b.sum())
                if K_b == 0:
                    continue
                unsel_idx_K = (unv_b.nonzero(as_tuple=True)[0] + 1).to(torch.int32)  # [K], 1-indexed, ascending
                unsel_locs_K = all_locs_cpu[b, unsel_idx_K.long()]         # [K, 2]
                unsel_dem_K = all_demand_cpu[b, unsel_idx_K.long()]        # [K]
                cur_loc_b = all_locs_cpu[b, cur_b]                          # [2]
                depot_loc_b = all_locs_cpu[b, 0]                            # [2]
                load_b = float(load_cpu[b])

                feas_K = (unsel_dem_K <= load_b + tol)                      # [K]

                direct_unsel = direct_probs[b, unv_b]                       # [K]
                via_unsel    = via_depot_probs[b, unv_b]                    # [K]

                teacher_depot_prob_b = float(via_unsel.sum())
                teacher_probs_K_t = direct_unsel.to(torch.float16)

                if not feas_K.any():
                    # forced return: no learnable signal
                    total_skipped_forced_depot += 1
                    continue

                argmax_2N = int(argmax_full_cpu[b])
                is_via_depot = (argmax_2N >= split_line)
                if is_via_depot:
                    sel_global = argmax_2N - split_line + 1  # 1-indexed customer in [1, N]
                else:
                    sel_global = argmax_2N + 1               # 1-indexed customer in [1, N]

                # a direct move whose demand exceeds the load is a refill via the depot (the teacher's environment)
                sel_demand_b = float(all_demand_cpu[b, sel_global].item())
                if (not is_via_depot) and (load_b + 1e-6 < sel_demand_b):
                    is_via_depot = True

                if is_via_depot:
                    next_local = -1
                else:
                    eq = (unsel_idx_K == sel_global).nonzero(as_tuple=True)[0]
                    if eq.numel() != 1:
                        continue
                    next_local = int(eq.item())

                # row 1: the state at the current node
                row1 = {
                    "instance_id":         torch.tensor(int(instance_ids[b].item()), dtype=torch.int32),
                    "pomo_idx":            torch.tensor(0, dtype=torch.int16),
                    "depot_loc":           depot_loc_b.clone().float(),
                    "current_loc":         cur_loc_b.clone().float(),
                    "load":                torch.tensor(load_b, dtype=torch.float32),
                    "unselected_idx":      unsel_idx_K,
                    "unselected_locs":     unsel_locs_K.float(),
                    "unselected_demand":   unsel_dem_K.float(),
                    "feasibility_mask":    feas_K.bool(),
                    "teacher_probs_K":     teacher_probs_K_t,
                    "teacher_depot_prob":  torch.tensor(teacher_depot_prob_b, dtype=torch.float16),
                    "teacher_logits_K":    teacher_probs_K_t.clone(),  # LEHD exposes no raw logits: the probabilities
                    "teacher_depot_logit": torch.tensor(teacher_depot_prob_b, dtype=torch.float16),
                    "next_node_local":     torch.tensor(next_local, dtype=torch.int16),
                }
                writer.add_row(K_b, row1)
                total_rows += 1

                # row 2: the state at the depot (via-depot moves only)
                if is_via_depot and float(via_unsel.sum()) > 1e-9:
                    via_norm = via_unsel / via_unsel.sum().clamp_min(1e-9)
                    eq = (unsel_idx_K == sel_global).nonzero(as_tuple=True)[0]
                    if eq.numel() == 1:
                        depot_next_local = int(eq.item())
                        row2 = {
                            "instance_id":         row1["instance_id"].clone(),
                            "pomo_idx":            torch.tensor(0, dtype=torch.int16),
                            "depot_loc":           depot_loc_b.clone().float(),
                            "current_loc":         depot_loc_b.clone().float(),  # at the depot
                            "load":                torch.tensor(capacity, dtype=torch.float32),
                            "unselected_idx":      unsel_idx_K,
                            "unselected_locs":     unsel_locs_K.float(),
                            "unselected_demand":   unsel_dem_K.float(),
                            "feasibility_mask":    (unsel_dem_K <= capacity).bool(),
                            "teacher_probs_K":     via_norm.to(torch.float16),
                            "teacher_depot_prob":  torch.tensor(0.0, dtype=torch.float16),
                            "teacher_logits_K":    via_norm.to(torch.float16),
                            "teacher_depot_logit": torch.tensor(float("-inf"), dtype=torch.float16),
                            "next_node_local":     torch.tensor(depot_next_local, dtype=torch.int16),
                        }
                        writer.add_row(K_b, row2)
                        total_rows += 1

            # transition of the whole batch
            argmax_full = probs_2N.argmax(dim=-1)                          # [B]
            is_via_argmax = (argmax_full >= split_line)
            sel_customer_global = torch.where(
                is_via_argmax, argmax_full - split_line + 1, argmax_full + 1
            ).long()                                                        # [B]

            sel_demand = torch.gather(
                problems[:, :, 2], 1, sel_customer_global.unsqueeze(-1)
            ).squeeze(-1)                                                   # [B]
            # via the depot: the argmax says so, or the demand does not fit
            is_via = is_via_argmax | (load + tol < sel_demand)
            load = torch.where(is_via, torch.full_like(load, capacity), load)
            load = load - sel_demand

            cust_zero_idx = sel_customer_global - 1
            visited_cust.scatter_(1, cust_zero_idx.unsqueeze(-1), True)
            current_node = sel_customer_global

            selected_node_list = torch.cat(
                [selected_node_list, sel_customer_global.unsqueeze(-1)], dim=-1
            )
            total_steps_simulated += bs

        total_generated += bs
        print(f"  Generated {total_generated:>6}/{num_instances} | "
              f"rows={total_rows:>9,} | forced_skipped={total_skipped_forced_depot:>6,} | "
              f"steps_simulated={total_steps_simulated:,}", flush=True)

    return {
        "rows_written": total_rows,
        "rows_skipped_forced_depot": total_skipped_forced_depot,
        "steps_simulated": total_steps_simulated,
        "next_instance_id": next_iid,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="K-bucketed decision-step data of a LEHD-CVRP teacher (greedy rollouts).")
    p.add_argument("--checkpoint", required=True, help="LEHD-CVRP checkpoint (NCO_code format)")
    p.add_argument("--nco-root", default=None,
                   help="clone of CIAM-Group/NCO_code or its single_objective/ (default: $NCO_ROOT)")
    p.add_argument("--problem-size", type=int, default=100)
    p.add_argument("--num-instances", type=int, default=25000)
    p.add_argument("--batch-size", type=int, default=128, help="instances per encoder call")
    p.add_argument("--target-per-bucket", type=int, default=4096, help="rows per shard of a K-bucket")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--instance-id-start", type=int, default=0,
                   help="id of the first instance (give parallel runs disjoint id ranges)")
    p.add_argument("--teacher-capacity-rule", action="store_true",
                   help="store the teacher's capacity rule (an exact fill fits, tolerance CAPACITY_TOL) in the "
                        "feasibility mask and apply it in the transition; default: no tolerance (the paper's data)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if os.path.exists(os.path.join(args.output, "manifest.pt")):
        raise FileExistsError(f"{args.output} already holds a dataset (manifest.pt)")

    print(f"Loading the LEHD-CVRP teacher from {args.checkpoint}")
    VRPModel = import_vrp_model(args.nco_root)
    model = VRPModel(**MODEL_PARAMS).to(device)        # built after the seed: its init draws from the same stream
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()
    print(f"  params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    writer = KBucketWriter(args.output, target_per_bucket=args.target_per_bucket)

    print(f"Generating LEHD-CVRP per-step data: num_instances={args.num_instances} problem_size={args.problem_size} "
          f"batch={args.batch_size} target_per_bucket={args.target_per_bucket} output={args.output}")

    t0 = time.perf_counter()
    stats = rollout_and_bucket(
        model=model,
        num_instances=args.num_instances,
        problem_size=args.problem_size,
        batch_size=args.batch_size,
        device=device,
        writer=writer,
        instance_id_start=args.instance_id_start,
        teacher_capacity_rule=args.teacher_capacity_rule,
    )
    rollout_sec = time.perf_counter() - t0

    writer.flush_all()

    history_entry = {
        "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
        "n_instances": args.num_instances,
        "pomo_size":   1,
        "seed":        args.seed,
        "rollout_sec": rollout_sec,
        "teacher":     "LEHD-CVRP",
        "checkpoint":  args.checkpoint,
        "teacher_capacity_rule": bool(args.teacher_capacity_rule),
        **stats,
    }
    manifest = writer.manifest(
        N=args.problem_size,
        n_instances=args.num_instances,
        pomo_size=1,
        history_entry=history_entry,
    )
    manifest_path = os.path.join(args.output, "manifest.pt")
    torch.save(manifest, manifest_path)

    print(f"Done in {rollout_sec:.1f}s: {manifest['total_rows']:,} rows")
    for K in sorted(manifest["per_K"].keys()):
        info = manifest["per_K"][K]
        marker = " (undersized)" if info["n_rows"] < args.target_per_bucket else ""
        print(f"  K={K:3d}: {info['n_rows']:>8,} rows in {len(info['shards']):>3} shard(s){marker}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
