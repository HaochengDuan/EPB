"""LEHD teacher on the TSP: per-step decision data of greedy LEHD rollouts. The model code is NCO_code's
LEHD.TSP.TSPModel, imported from a clone of https://github.com/CIAM-Group/NCO_code (--nco-root, or NCO_ROOT).

  python -m epb.teachers.lehd_tsp --ckpt <LEHD_TSP_CKPT> --nco-root <NCO_code clone> --problem-size 100 \
      --instance-source pomo --pomo-file <POMO_TSP100_STEPS> --first-node 0 --start-step 2 --batch-size 1024 \
      --save-dtype float16 --output lehd_tsp100_steps.pt

Output (one .pt file):

  meta            {N, n_instances, teacher, ckpt, instance_source, first_node_idx, saved_dtype, start_step}
  instance_locs   [I, N, 2] float32
  tour            [I, N]    int64     the teacher's greedy tour (first .. last)
  per_step        {step: {unselected_locs [I, K, 2], first_loc [I, 2], current_loc [I, 2], teacher_logits [I, K],
                   teacher_probs [I, K], next_node_local [I] int32, first_node_idx [I], current_node_idx [I],
                   unselected_idx [I, K]}} for step = start_step .. N-1, K = N - step

step follows the POMO convention (step 1 = the first decision, one node visited). Step 1 is symmetry-degenerate on
the TSP (every first move is equivalent), so --start-step 2 rolls it forward without storing it. unselected_idx is in
ascending node order (LEHD's own convention), so teacher_logits[k] belongs to node unselected_idx[k];
next_node_local is the argmax over the K candidates. Instances: those of a POMO decision-step file (its step_i == 1
rows, --instance-source pomo) or fresh uniform ones from a torch.Generator seeded with --seed.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn.functional as F

from . import add_nco_path

MODEL_PARAMS = {
    "mode": "test",
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** 0.5,
    "decoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "ff_hidden_dim": 512,
}


def build_lehd_model(ckpt_path: str, device: torch.device, nco_root: str | None = None):
    """NCO_code's LEHD TSPModel (test mode) with the weights of ckpt_path, in eval mode."""
    add_nco_path(nco_root)
    from LEHD.TSP.TSPModel import TSPModel

    model = TSPModel(**MODEL_PARAMS).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    print(f"[LEHD] loaded the teacher ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params)")
    return model


@torch.no_grad()
def lehd_step(model, encoded_nodes: torch.Tensor, selected_node_list: torch.Tensor):
    """One decoder step on the encoder output encoded_nodes [B, N, D] with the visited nodes selected_node_list
    [B, T] (global ids, first .. last). Returns, over the K = N - T unselected nodes in ascending order:
    logits_unsel [B, K] (raw scores), probs_unsel [B, K] (LEHD's softmax with the first / last anchors masked),
    next_node_local [B] (argmax) and unselected_idx [B, K] (global ids)."""
    decoder = model.decoder
    data = encoded_nodes
    B, N, D = data.shape
    T = selected_node_list.size(1)
    K = N - T

    # decoder._get_new_data, device-safe
    new_list = torch.arange(N, device=data.device).unsqueeze(0).repeat(B, 1)
    idx_b = torch.arange(B, device=data.device).unsqueeze(1).expand(B, T)
    new_list[idx_b, selected_node_list] = -2
    unselected_idx = new_list[new_list >= 0].view(B, K)  # [B, K], ascending

    unsel_embs = data.gather(1, unselected_idx.unsqueeze(-1).expand(B, K, D))  # [B, K, D]

    first_idx = selected_node_list[:, 0:1]   # [B, 1]
    last_idx  = selected_node_list[:, -1:]   # [B, 1]
    first_emb = data.gather(1, first_idx.unsqueeze(-1).expand(B, 1, D)).squeeze(1)  # [B, D]
    last_emb  = data.gather(1, last_idx.unsqueeze(-1).expand(B, 1, D)).squeeze(1)   # [B, D]
    first_emb = decoder.embedding_first_node(first_emb).unsqueeze(1)  # [B, 1, D]
    last_emb  = decoder.embedding_last_node(last_emb).unsqueeze(1)    # [B, 1, D]

    out = torch.cat([first_emb, unsel_embs, last_emb], dim=1)  # [B, K+2, D]
    for layer in decoder.layers:
        out = layer(out)
    out = decoder.Linear_final(out).squeeze(-1)  # [B, K+2]

    logits_unsel = out[:, 1:-1].clone()

    out_masked = out.clone()
    out_masked[:, 0] = float("-inf")
    out_masked[:, -1] = float("-inf")
    probs_full = F.softmax(out_masked, dim=-1)
    probs_unsel = probs_full[:, 1:-1]

    next_node_local = probs_unsel.argmax(dim=-1)  # [B]
    return logits_unsel, probs_unsel, next_node_local, unselected_idx


# ─────────────────────────────────────────────────────────────────────────────
# Instances: those of a POMO decision-step file, or fresh
# ─────────────────────────────────────────────────────────────────────────────

def load_pomo_instances(pomo_path: str, n_instances: int | None = None) -> torch.Tensor:
    """The instances of a POMO-TSP decision-step file (its rows with step_i == 1, in file order)."""
    print(f"[instance] loading the POMO decision-step file {pomo_path}")
    t0 = time.perf_counter()
    data = torch.load(pomo_path, map_location="cpu", weights_only=False)
    step_i = data["step_i"]
    locs = data["locs"]
    print(f"[instance] loaded in {time.perf_counter() - t0:.1f}s, {len(step_i):,} rows")

    step1_idx = (step_i == 1.0).nonzero(as_tuple=True)[0]
    instances = locs[step1_idx]
    print(f"[instance] {instances.size(0):,} instances (rows with step_i == 1)")

    if n_instances is not None and n_instances < instances.size(0):
        instances = instances[:n_instances].contiguous()
        print(f"[instance] truncated to {n_instances:,}")

    del data
    return instances.float().contiguous()


def generate_fresh_instances(n_instances: int, N: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n_instances, N, 2, generator=g)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout_lehd_greedy(
    model,
    instances: torch.Tensor,            # [I, N, 2] on CPU
    first_node_idx: int,
    batch_size: int,
    device: torch.device,
    save_dtype: torch.dtype = torch.float16,
    start_step: int = 2,
):
    """Greedy LEHD rollout of every instance from first_node_idx; returns (per_step buffers, tours [I, N]). Steps
    before start_step are executed (the trajectory needs them) but not stored."""
    I, N, _ = instances.shape
    print(f"[rollout] {I:,} instances, N={N}, batch_size={batch_size}, "
          f"first_node={first_node_idx}, start_step={start_step}, device={device}")

    per_step = {}
    for step in range(start_step, N):
        K = N - step
        per_step[step] = {
            "unselected_locs":  torch.empty(I, K, 2, dtype=torch.float32),
            "first_loc":        torch.empty(I, 2, dtype=torch.float32),
            "current_loc":      torch.empty(I, 2, dtype=torch.float32),
            "teacher_logits":   torch.empty(I, K, dtype=save_dtype),
            "teacher_probs":    torch.empty(I, K, dtype=save_dtype),
            "next_node_local":  torch.empty(I, dtype=torch.int32),
            "first_node_idx":   torch.empty(I, dtype=torch.int64),
            "current_node_idx": torch.empty(I, dtype=torch.int64),
            "unselected_idx":   torch.empty(I, K, dtype=torch.int64),
        }

    full_tours = torch.empty(I, N, dtype=torch.int64)

    n_chunks = (I + batch_size - 1) // batch_size
    t_start = time.perf_counter()

    for chunk_idx in range(n_chunks):
        b0 = chunk_idx * batch_size
        b1 = min(b0 + batch_size, I)
        Bc = b1 - b0
        locs_b = instances[b0:b1].to(device)        # [Bc, N, 2]

        encoded = model.encoder(locs_b)              # [Bc, N, D]

        selected = torch.full((Bc, 1), first_node_idx, dtype=torch.long, device=device)

        for step in range(1, N):
            logits_unsel, probs_unsel, next_local, unsel_idx = lehd_step(
                model, encoded, selected
            )
            if step >= start_step:
                K = unsel_idx.size(1)
                unsel_locs = locs_b.gather(
                    1, unsel_idx.unsqueeze(-1).expand(Bc, K, 2)
                )  # [Bc, K, 2]
                first_idx = selected[:, 0]    # [Bc]
                curr_idx  = selected[:, -1]   # [Bc]
                first_loc = locs_b.gather(1, first_idx.view(Bc, 1, 1).expand(Bc, 1, 2)).squeeze(1)
                current_loc = locs_b.gather(1, curr_idx.view(Bc, 1, 1).expand(Bc, 1, 2)).squeeze(1)

                buf = per_step[step]
                buf["unselected_locs"][b0:b1]  = unsel_locs.cpu().float()
                buf["first_loc"][b0:b1]        = first_loc.cpu().float()
                buf["current_loc"][b0:b1]      = current_loc.cpu().float()
                buf["teacher_logits"][b0:b1]   = logits_unsel.cpu().to(save_dtype)
                buf["teacher_probs"][b0:b1]    = probs_unsel.cpu().to(save_dtype)
                buf["next_node_local"][b0:b1]  = next_local.cpu().to(torch.int32)
                buf["first_node_idx"][b0:b1]   = first_idx.cpu()
                buf["current_node_idx"][b0:b1] = curr_idx.cpu()
                buf["unselected_idx"][b0:b1]   = unsel_idx.cpu()

            next_global = unsel_idx.gather(1, next_local.unsqueeze(1))  # [Bc, 1]
            selected = torch.cat([selected, next_global], dim=1)

        full_tours[b0:b1] = selected.cpu()

        elapsed = time.perf_counter() - t_start
        eta = elapsed / (chunk_idx + 1) * (n_chunks - chunk_idx - 1)
        print(
            f"[rollout] chunk {chunk_idx + 1}/{n_chunks} "
            f"({b1:,}/{I:,})  elapsed={elapsed:.1f}s  ETA={eta:.1f}s",
            flush=True,
        )

    return per_step, full_tours


def tour_length(locs: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    """locs [I, N, 2], tour [I, N] -> closed-tour lengths [I]."""
    I, N, _ = locs.shape
    ordered = locs.gather(1, tour.unsqueeze(-1).expand(I, N, 2))  # [I, N, 2]
    rolled = torch.roll(ordered, shifts=-1, dims=1)
    seg = ((ordered - rolled) ** 2).sum(-1).sqrt()
    return seg.sum(-1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Per-step decision data of a LEHD-TSP teacher (greedy rollouts).")
    p.add_argument("--ckpt", type=str, required=True, help="LEHD-TSP checkpoint (NCO_code format)")
    p.add_argument("--nco-root", type=str, default=None,
                   help="clone of CIAM-Group/NCO_code or its single_objective/ (default: $NCO_ROOT)")
    p.add_argument("--problem-size", type=int, default=100, help="N (number of cities)")
    p.add_argument("--n-instances", type=int, default=None,
                   help="number of instances (default: all of the POMO file; 100000 for fresh instances)")
    p.add_argument("--instance-source", choices=["pomo", "fresh"], default="pomo",
                   help="the instances of a POMO decision-step file, or fresh uniform instances")
    p.add_argument("--pomo-file", type=str, default=None, help="POMO-TSP decision-step file (--instance-source pomo)")
    p.add_argument("--first-node", type=int, default=0, help="start node of the rollouts")
    p.add_argument("--start-step", type=int, default=2,
                   help="first step stored (earlier steps are rolled forward, not stored)")
    p.add_argument("--batch-size", type=int, default=1024, help="instances per rollout batch")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--output", type=str, required=True, help="output .pt file")
    p.add_argument("--save-dtype", choices=["float16", "float32"], default="float16",
                   help="dtype of the stored logits / probs (coordinates are always float32)")
    p.add_argument("--seed", type=int, default=42, help="seed of the fresh instances")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32

    model = build_lehd_model(args.ckpt, device, args.nco_root)

    if args.instance_source == "pomo":
        if args.pomo_file is None:
            raise ValueError("--pomo-file is required when --instance-source=pomo")
        instances = load_pomo_instances(args.pomo_file, args.n_instances)
        if instances.size(1) != args.problem_size:
            raise ValueError(
                f"POMO instance N={instances.size(1)} does not match "
                f"--problem-size={args.problem_size}. Use --instance-source=fresh "
                f"if you want a different N."
            )
    else:
        n = args.n_instances or 100_000
        instances = generate_fresh_instances(n, args.problem_size, seed=args.seed)
        print(f"[instance] generated {n:,} fresh instances, N={args.problem_size}")

    per_step, full_tours = rollout_lehd_greedy(
        model, instances,
        first_node_idx=args.first_node,
        batch_size=args.batch_size,
        device=device,
        save_dtype=save_dtype,
        start_step=args.start_step,
    )
    lengths = tour_length(instances, full_tours)
    print(f"[rollout] tour length mean={lengths.mean().item():.4f}  min={lengths.min().item():.4f}  "
          f"max={lengths.max().item():.4f}")

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    print(f"[save] writing {out_path} ...")
    t0 = time.perf_counter()
    torch.save({
        "meta": {
            "N": args.problem_size,
            "n_instances": instances.size(0),
            "teacher": "LEHD",
            "ckpt": args.ckpt,
            "instance_source": args.instance_source,
            "first_node_idx": args.first_node,
            "saved_dtype": args.save_dtype,
            "start_step": args.start_step,
        },
        "instance_locs": instances,
        "tour": full_tours,
        "per_step": per_step,
    }, out_path)
    print(f"[save] done ({os.path.getsize(out_path) / 1e9:.2f} GB, {time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
