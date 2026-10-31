"""POMO teacher on the TSP: the model (from yd-kwon/POMO, NEW_py_ver) and the decision-step generator.

  python -m epb.teachers.pomo_tsp --checkpoint <POMO_TSP_CKPT> --problem-size 100 --num-instances 100000 \
      --batch-size 256 --pomo-size 1 --seed 42 --output pomo_tsp100_steps.pt

The generator runs greedy POMO rollouts from start nodes 0 .. pomo_size-1 and records every decision step t = 1 ..
N-1 into one .pt file: teacher_logits, teacher_probs, mask (1 = unvisited), locs, current_node, first_node, visited,
step_i (= t, the number of visited nodes) and step_max (= N - 1). Rows are batch-major: the rows of all rollouts of a
batch at step 1, then at step 2, and so on, so --batch-size is part of the recipe. The checkpoint is a POMO-TSP
checkpoint in the yd-kwon/POMO format (`model_state_dict`).
"""

import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# POMO model (yd-kwon/POMO NEW_py_ver)
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
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = F.softmax(score_scaled, dim=3)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    return out_transposed.reshape(batch_s, n, head_num * key_dim)


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        return normalized.transpose(1, 2)


class Feed_Forward_Module(nn.Module):
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

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)
        return out3


class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        encoder_layer_num = model_params['encoder_layer_num']
        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data):
        out = self.embedding(data)
        for layer in self.layers:
            out = layer(out)
        return out


class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None
        self.v = None
        self.single_head_key = None
        self.q_first = None

    def set_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def set_q1(self, encoded_q1):
        head_num = self.model_params['head_num']
        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)

    def forward(self, encoded_last_node, ninf_mask):
        """Returns (probs, logits); logits are the masked pre-softmax scores."""
        head_num = self.model_params['head_num']

        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        q = self.q_first + q_last
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


class TSPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None

    def pre_forward(self, problems):
        """Encode the node coordinates problems (batch, problem, 2)."""
        self.encoded_nodes = self.encoder(problems)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, current_node, ninf_mask, first_node=None):
        """One decoding step: current_node (batch, pomo) or None at step 0, ninf_mask (batch, pomo, problem).
        Returns (selected (batch, pomo), probs, logits (batch, pomo, problem)); step 0 selects the start nodes
        0 .. pomo-1, later steps the argmax."""
        batch_size = ninf_mask.size(0)
        pomo_size = ninf_mask.size(1)

        if current_node is None:
            selected = torch.arange(pomo_size, device=ninf_mask.device)[None, :].expand(batch_size, pomo_size)
            probs = torch.ones(batch_size, pomo_size, ninf_mask.size(2), device=ninf_mask.device) / ninf_mask.size(2)
            logits = torch.zeros_like(probs)

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            self.decoder.set_q1(encoded_first_node)
        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, current_node)
            probs, logits = self.decoder(encoded_last_node, ninf_mask=ninf_mask)

            selected = probs.argmax(dim=2)

        return selected, probs, logits


# ─────────────────────────────────────────────────────────────────────────────
# Decision steps
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_decision_steps(
    model: TSPModel,
    num_instances: int,
    problem_size: int,
    pomo_size: int,
    batch_size: int,
    device: torch.device,
    output_path: str,
    shard_size: int = 500_000,
) -> str:
    """Greedy POMO rollouts on fresh uniform instances (torch.rand per batch), every decision step recorded.
    Steps are flushed to temporary shards every `shard_size` rows and merged into output_path at the end."""
    model.eval()
    step_max = float(problem_size - 1)

    shard_dir = output_path + ".shards"
    os.makedirs(shard_dir, exist_ok=True)
    shard_idx = 0
    total_steps = 0

    buf = {
        "teacher_logits": [], "teacher_probs": [], "mask": [],
        "locs": [], "current_node": [], "first_node": [],
        "visited": [], "step_i": [],
    }
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

        problems = torch.rand(bs, problem_size, 2, device=device)
        problems_cpu = problems.cpu()

        model.pre_forward(problems)

        current_node = None
        first_node_tracker = None
        ninf_mask = torch.zeros(bs, pomo_size, problem_size, device=device)
        visited = torch.zeros(bs, pomo_size, problem_size, device=device)

        batch_idx = torch.arange(bs, device=device)[:, None].expand(bs, pomo_size)
        pomo_idx = torch.arange(pomo_size, device=device)[None, :].expand(bs, pomo_size)

        for step in range(problem_size):
            selected, probs, logits = model(current_node, ninf_mask, first_node_tracker)

            if step == 0:
                first_node_tracker = selected.clone()
                current_node = selected
                ninf_mask[batch_idx, pomo_idx, selected] = float('-inf')
                visited[batch_idx, pomo_idx, selected] = 1.0
                continue

            feasible_mask = (ninf_mask == 0).float()

            # (bs, pomo, ...) -> (bs * pomo, ...)
            n_new = bs * pomo_size
            buf["teacher_logits"].append(logits.cpu().reshape(n_new, problem_size))
            buf["teacher_probs"].append(probs.cpu().reshape(n_new, problem_size))
            buf["mask"].append(feasible_mask.cpu().reshape(n_new, problem_size))
            buf["locs"].append(problems_cpu.unsqueeze(1).expand(bs, pomo_size, problem_size, 2)
                               .reshape(n_new, problem_size, 2))
            buf["current_node"].append(current_node.cpu().reshape(n_new))
            buf["first_node"].append(first_node_tracker.cpu().reshape(n_new))
            buf["visited"].append(visited.cpu().reshape(n_new, problem_size))
            buf["step_i"].append(torch.full((n_new,), step, dtype=torch.float32))

            buf_count += n_new
            total_steps += n_new

            if buf_count >= shard_size:
                _flush()

            current_node = selected
            ninf_mask[batch_idx, pomo_idx, selected] = float('-inf')
            visited[batch_idx, pomo_idx, selected] = 1.0

        total_generated += bs
        print(f"  Generated {total_generated}/{num_instances} instances "
              f"({total_steps} decision steps so far)")

    _flush()

    print(f"Merging {shard_idx} shards into {output_path}...")
    shard_files = sorted(
        [os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".pt")]
    )
    merged = {k: [] for k in buf}
    for sf in shard_files:
        shard = torch.load(sf, map_location="cpu")
        for k in merged:
            merged[k].append(shard[k])
        del shard
    result = {k: torch.cat(v, dim=0) for k, v in merged.items()}
    del merged
    result["step_max"] = torch.full((total_steps,), step_max)
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
    p = argparse.ArgumentParser(description="Decision-step dataset of a POMO-TSP teacher (greedy rollouts).")
    p.add_argument("--checkpoint", required=True, help="POMO-TSP checkpoint (yd-kwon/POMO format)")
    p.add_argument("--problem-size", type=int, default=100)
    p.add_argument("--pomo-size", type=int, default=None,
                   help="start nodes 0 .. pomo_size-1 per instance (default: problem_size; 1 = one greedy rollout)")
    p.add_argument("--num-instances", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=100, help="instances per rollout batch (part of the recipe)")
    p.add_argument("--output", required=True, help="output .pt file")
    p.add_argument("--seed", type=int, default=42)
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

    print(f"Loading the POMO-TSP teacher from {args.checkpoint}")
    model = TSPModel(**model_params).to(device)        # built after the seed: its init draws from the same stream
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    print(f"Generating decision steps: {args.num_instances} instances, "
          f"problem_size={args.problem_size}, pomo_size={pomo_size}")
    generate_decision_steps(
        model=model,
        num_instances=args.num_instances,
        problem_size=args.problem_size,
        pomo_size=pomo_size,
        batch_size=args.batch_size,
        device=device,
        output_path=args.output,
    )

    print(f"Saved to {args.output}")
    data = torch.load(args.output, map_location="cpu")
    for k, v in data.items():
        print(f"    {k}: {tuple(v.shape)}")
    del data


if __name__ == "__main__":
    main()
