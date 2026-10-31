"""POMO teacher on the CVRP: the model (from yd-kwon/POMO, NEW_py_ver/CVRP), POMO's instance generator and the
decision-step generator.

  python -m epb.teachers.pomo_cvrp --checkpoint <POMO_CVRP_CKPT> --problem-size 100 --pomo-size 20 \
      --num-instances 100000 --batch-size 512 --seed 123 --no-merge --output pomo_cvrp100_steps

Instances: uniform depot and customer coordinates, integer demands 1..9 divided by the capacity of the problem size
(30 / 40 / 50 / 80 / 85 / 100 for N = 20 / 50 / 100 / 200 / 250 / 500), drawn per batch from the global torch
stream, so --batch-size is part of the recipe. --pomo-size P records the P greedy trajectories that start at
customers 1..P; each contributes one row per move until every customer is served: teacher_logits, teacher_probs,
mask (1 = feasible), locs (depot first), demand (depot 0), current_node, first_node, visited, load (remaining
capacity), step_i, step_max. Output: one .pt file, or with --no-merge a sharded directory (manifest.pt +
shard_NNNN.pt, about 500k rows per shard). The checkpoint is a POMO-CVRP checkpoint in the yd-kwon/POMO format.
"""

import argparse
import datetime
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# POMO-CVRP model (yd-kwon/POMO NEW_py_ver/CVRP)
# ─────────────────────────────────────────────────────────────────────────────

def reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    return q_reshaped.transpose(1, 2)


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / math.sqrt(key_dim)
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(
            batch_s, head_num, n, input_s
        )
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(
            batch_s, head_num, n, input_s
        )

    weights = F.softmax(score_scaled, dim=3)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    return out_transposed.reshape(batch_s, n, head_num * key_dim)


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        return normalized.transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)
        return out3


class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        encoder_layer_num = model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand):
        # depot_xy (batch, 1, 2), node_xy_demand (batch, problem, 3) -> (batch, problem + 1, embedding)
        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_xy_demand)
        out = torch.cat((embedded_depot, embedded_node), dim=1)
        for layer in self.layers:
            out = layer(out)
        return out


class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']

        self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None
        self.v = None
        self.single_head_key = None

    def set_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(self, encoded_last_node, load, ninf_mask):
        """Returns (probs, logits)."""
        head_num = self.model_params['head_num']

        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        q = q_last

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        score = torch.matmul(mh_atten_out, self.single_head_key)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        return probs, score_masked


def _get_encoding(encoded_nodes, node_index_to_pick):
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    return encoded_nodes.gather(dim=1, index=gathering_index)


class CVRPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None

    def pre_forward(self, depot_xy, node_xy, node_demand):
        """Encode depot_xy (B, 1, 2), node_xy (B, N, 2), node_demand (B, N)."""
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, current_node, load, ninf_mask):
        """One decoding step after pre_forward: current_node (batch, pomo), load (batch, pomo) remaining capacity,
        ninf_mask (batch, pomo, problem + 1). Returns (selected = argmax (batch, pomo), probs, logits)."""
        encoded_last_node = _get_encoding(self.encoded_nodes, current_node)
        probs, logits = self.decoder(encoded_last_node, load, ninf_mask)
        selected = probs.argmax(dim=2)
        return selected, probs, logits


# ─────────────────────────────────────────────────────────────────────────────
# Instances (yd-kwon/POMO)
# ─────────────────────────────────────────────────────────────────────────────

def get_random_problems(batch_size, problem_size):
    depot_xy = torch.rand(size=(batch_size, 1, 2))
    node_xy = torch.rand(size=(batch_size, problem_size, 2))

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    elif problem_size == 200:
        demand_scaler = 80
    elif problem_size == 250:
        demand_scaler = 85  # interpolated between N = 200 (80) and N = 500 (100)
    elif problem_size == 500:
        demand_scaler = 100
    else:
        raise NotImplementedError(f"No demand_scaler for problem_size={problem_size}")

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    return depot_xy, node_xy, node_demand


# ─────────────────────────────────────────────────────────────────────────────
# Decision steps
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_decision_steps(
    model: CVRPModel,
    num_instances: int,
    problem_size: int,
    pomo_size: int,
    batch_size: int,
    device: torch.device,
    output_path: str,
    shard_size: int = 500_000,
    no_merge: bool = False,
    seed: Optional[int] = None,
) -> str:
    """Greedy POMO-CVRP rollouts (start customers 1..pomo_size), every decision step of an unfinished trajectory
    recorded. The depot can be revisited; the mask combines the visited flags and load vs demand (a customer fits
    if demand <= load + 1e-5); a trajectory ends when every customer is served. no_merge: output_path is the shard
    directory (with manifest.pt); otherwise the shards are merged into the file output_path."""
    model.eval()
    num_nodes = problem_size + 1  # depot + customers

    shard_dir = output_path if no_merge else output_path + ".shards"
    if no_merge and os.path.exists(os.path.join(shard_dir, "manifest.pt")):
        raise FileExistsError(f"{shard_dir} already holds a dataset (manifest.pt)")
    os.makedirs(shard_dir, exist_ok=True)
    shard_idx = 0
    total_steps = 0

    buf_keys = [
        "teacher_logits", "teacher_probs", "mask",
        "locs", "demand", "current_node", "first_node",
        "visited", "load", "step_i",
    ]
    buf = {k: [] for k in buf_keys}
    buf_count = 0

    def _flush():
        nonlocal shard_idx, buf_count
        if buf_count == 0:
            return
        shard = {k: torch.cat(v, dim=0) for k, v in buf.items()}
        shard_path = os.path.join(shard_dir, f"shard_{shard_idx:04d}.pt")
        torch.save(shard, shard_path)
        print(f"    Flushed shard {shard_idx} ({buf_count} steps) -> {shard_path}")
        shard_idx += 1
        buf_count = 0
        for k in buf:
            buf[k] = []

    total_generated = 0
    while total_generated < num_instances:
        bs = min(batch_size, num_instances - total_generated)

        depot_xy, node_xy, node_demand = get_random_problems(bs, problem_size)
        depot_xy = depot_xy.to(device)
        node_xy = node_xy.to(device)
        node_demand = node_demand.to(device)

        all_locs = torch.cat((depot_xy, node_xy), dim=1)  # (bs, N+1, 2)
        all_demand = torch.cat(
            (torch.zeros(bs, 1, device=device), node_demand), dim=1
        )  # (bs, N+1)

        all_locs_cpu = all_locs.cpu()
        all_demand_cpu = all_demand.cpu()

        model.pre_forward(depot_xy, node_xy, node_demand)

        # step 0: every trajectory is at the depot
        current_node = torch.zeros(bs, pomo_size, dtype=torch.long, device=device)

        load = torch.ones(bs, pomo_size, device=device)
        visited_flag = torch.zeros(bs, pomo_size, num_nodes, device=device)
        ninf_mask = torch.zeros(bs, pomo_size, num_nodes, device=device)
        finished = torch.zeros(bs, pomo_size, dtype=torch.bool, device=device)

        batch_idx = torch.arange(bs, device=device)[:, None].expand(bs, pomo_size)
        pomo_idx = torch.arange(pomo_size, device=device)[None, :].expand(bs, pomo_size)

        visited_flag[:, :, 0] = 0  # the depot can be revisited
        at_the_depot = torch.ones(bs, pomo_size, dtype=torch.bool, device=device)

        # step 1 (POMO start): trajectory p goes to customer p + 1
        first_customer = torch.arange(1, pomo_size + 1, device=device)[None, :].expand(bs, pomo_size)
        current_node = first_customer.clone()
        first_node_tracker = first_customer.clone()

        demand_list = all_demand[:, None, :].expand(bs, pomo_size, num_nodes)
        selected_demand = demand_list.gather(dim=2, index=current_node[:, :, None]).squeeze(2)
        load -= selected_demand
        at_the_depot = (current_node == 0)

        visited_flag[batch_idx, pomo_idx, current_node] = 1.0

        visited_ninf = torch.zeros(bs, pomo_size, num_nodes, device=device)
        visited_ninf[batch_idx, pomo_idx, current_node] = float('-inf')
        visited_ninf[:, :, 0][~at_the_depot] = 0  # the depot is open unless the vehicle is there

        ninf_mask = visited_ninf.clone()
        round_error_epsilon = 0.00001
        demand_too_large = load[:, :, None] + round_error_epsilon < demand_list
        ninf_mask[demand_too_large] = float('-inf')

        newly_finished = (visited_ninf == float('-inf')).all(dim=2)
        finished = finished | newly_finished
        ninf_mask[:, :, 0][finished] = 0

        step = 2
        max_steps = problem_size * 3  # safety limit (depot revisits)

        while not finished.all() and step < max_steps:
            selected, probs, logits = model(current_node, load, ninf_mask)

            feasible_mask = (ninf_mask == 0).float()

            # rows of the unfinished trajectories only
            active = ~finished  # (bs, pomo)
            n_active = active.sum().item()

            if n_active > 0:
                flat_active = active.reshape(-1)  # (bs*pomo,)

                flat_logits = logits.reshape(-1, num_nodes)[flat_active]
                flat_probs = probs.reshape(-1, num_nodes)[flat_active]
                flat_mask = feasible_mask.reshape(-1, num_nodes)[flat_active]
                flat_locs = all_locs_cpu.unsqueeze(1).expand(bs, pomo_size, num_nodes, 2)\
                    .reshape(-1, num_nodes, 2)[flat_active.cpu()]
                flat_demand = all_demand_cpu.unsqueeze(1).expand(bs, pomo_size, num_nodes)\
                    .reshape(-1, num_nodes)[flat_active.cpu()]
                flat_cur = current_node.reshape(-1)[flat_active]
                flat_first = first_node_tracker.reshape(-1)[flat_active]
                flat_visited = visited_flag.reshape(-1, num_nodes)[flat_active]
                flat_load = load.reshape(-1)[flat_active]

                buf["teacher_logits"].append(flat_logits.cpu())
                buf["teacher_probs"].append(flat_probs.cpu())
                buf["mask"].append(flat_mask.cpu())
                buf["locs"].append(flat_locs)
                buf["demand"].append(flat_demand)
                buf["current_node"].append(flat_cur.cpu())
                buf["first_node"].append(flat_first.cpu())
                buf["visited"].append(flat_visited.cpu())
                buf["load"].append(flat_load.cpu())
                buf["step_i"].append(torch.full((n_active,), step, dtype=torch.float32))

                buf_count += n_active
                total_steps += n_active

                if buf_count >= shard_size:
                    _flush()

            # transition
            current_node = selected
            at_the_depot = (selected == 0)

            selected_demand = demand_list.gather(dim=2, index=selected[:, :, None]).squeeze(2)
            load -= selected_demand
            load[at_the_depot] = 1  # refill at the depot

            visited_ninf[batch_idx, pomo_idx, selected] = float('-inf')
            visited_ninf[:, :, 0][~at_the_depot] = 0

            visited_flag[batch_idx, pomo_idx, selected] = 1.0
            visited_flag[:, :, 0] = 0  # the depot is never marked visited

            ninf_mask = visited_ninf.clone()
            demand_too_large = load[:, :, None] + round_error_epsilon < demand_list
            ninf_mask[demand_too_large] = float('-inf')

            newly_finished = (visited_ninf == float('-inf')).all(dim=2)
            finished = finished | newly_finished
            ninf_mask[:, :, 0][finished] = 0

            step += 1

        total_generated += bs
        print(f"  Generated {total_generated}/{num_instances} instances "
              f"({total_steps} decision steps, last episode {step} steps)")

    _flush()

    if no_merge:
        manifest_path = os.path.join(shard_dir, "manifest.pt")
        shard_files = [f"shard_{i:04d}.pt" for i in range(shard_idx)]
        manifest = {
            "format":      "sharded_v1",
            "total_steps": total_steps,
            "shard_files": shard_files,
            "step_max":    float(step - 1),
            "history":     [{
                "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
                "n_instances": num_instances,
                "seed":        seed,
                "shard_files": shard_files,
                "n_steps":     total_steps,
            }],
        }
        torch.save(manifest, manifest_path)
        print(f"Sharded mode: wrote {shard_idx} shards + manifest to {shard_dir} ({total_steps:,} steps)")
        return shard_dir

    print(f"Merging {shard_idx} shards into {output_path}...")
    shard_files = sorted(
        [os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".pt")]
    )
    merged = {k: [] for k in buf_keys}
    for sf in shard_files:
        shard = torch.load(sf, map_location="cpu")
        for k in merged:
            merged[k].append(shard[k])
        del shard

    result = {k: torch.cat(v, dim=0) for k, v in merged.items()}
    del merged

    # step_max: the last step index of the last batch
    result["step_max"] = torch.full((total_steps,), float(step - 1))
    torch.save(result, output_path)

    for sf in shard_files:
        os.remove(sf)
    os.rmdir(shard_dir)

    print(f"Total decision steps: {total_steps}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Decision-step dataset of a POMO-CVRP teacher (greedy rollouts).")
    p.add_argument("--checkpoint", required=True, help="POMO-CVRP checkpoint (yd-kwon/POMO format)")
    p.add_argument("--problem-size", type=int, default=100, help="customers (20, 50, 100, 200, 250 or 500)")
    p.add_argument("--pomo-size", type=int, default=5, help="trajectories per instance (start customers 1..P)")
    p.add_argument("--num-instances", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64, help="instances per rollout batch (part of the recipe)")
    p.add_argument("--output", required=True, help="output .pt file, or the shard directory with --no-merge")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-merge", action="store_true",
                   help="write a sharded directory (manifest.pt + shard_NNNN.pt) at --output instead of one file")
    # architecture (must match the checkpoint)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--encoder-layers", type=int, default=6)
    p.add_argument("--head-num", type=int, default=8)
    p.add_argument("--qkv-dim", type=int, default=16)
    p.add_argument("--ff-hidden-dim", type=int, default=512)
    p.add_argument("--logit-clipping", type=float, default=10)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pomo_size = args.pomo_size if args.pomo_size is not None else args.problem_size

    model_params = {
        'embedding_dim': args.embedding_dim,
        'sqrt_embedding_dim': args.embedding_dim ** 0.5,
        'encoder_layer_num': args.encoder_layers,
        'qkv_dim': args.qkv_dim,
        'head_num': args.head_num,
        'logit_clipping': args.logit_clipping,
        'ff_hidden_dim': args.ff_hidden_dim,
        'eval_type': 'argmax',
    }

    print(f"Loading the POMO-CVRP teacher from {args.checkpoint}")
    model = CVRPModel(**model_params).to(device)       # built after the seed: its init draws from the same stream
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    print(f"Generating CVRP decision steps: {args.num_instances} instances, "
          f"problem_size={args.problem_size}, pomo_size={pomo_size}")
    out_path = generate_decision_steps(
        model=model,
        num_instances=args.num_instances,
        problem_size=args.problem_size,
        pomo_size=pomo_size,
        batch_size=args.batch_size,
        device=device,
        output_path=args.output,
        no_merge=args.no_merge,
        seed=args.seed,
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
