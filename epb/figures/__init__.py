import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .. import eval as E
from .. import instances as I
from .. import settings as S

DATA = os.path.dirname(os.path.abspath(__file__))
STAGE_GROUPS = os.path.join(DATA, "cvrp50_stages.json")
STAGE_INSTANCES = "file:" + os.path.join(DATA, "cvrp50_stage_instances.pt")
TOUR_LEGEND = os.path.join(DATA, "tsp100_legend.json")

STAGE_STARTS = 8
STAGE_BATCH = 64
STAGE_BINS = 20
STAGE_SIGMA = 1.0
STAGE_LIGHTEN = 0.35
STAGE_FS = dict(legend=34, label=38, tick=30)
TOUR_DECODE = ("onehot", "onehot")
TOUR_FS = dict(legend=30, title=34, tick=24)
RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
      "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _student_args(setting_name: str, ckpt: str, device: str, extra: List[str] = ()):
    return S.get(setting_name), E.build_args(["--setting", setting_name, "--method", "epb", "--ckpt", ckpt,
                                              "--device", device, *extra])


def _problem_size(ckpt_file: str) -> Optional[int]:
    args = torch.load(ckpt_file, map_location="cpu", weights_only=False).get("args") or {}
    args = args if isinstance(args, dict) else vars(args)
    return args.get("problem_size")


def _name(ckpt: str) -> str:
    p = os.path.normpath(ckpt)
    return os.path.basename(p if os.path.isdir(p) else os.path.dirname(p))


def _lighten(color, f: float):
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def _fit_to_legend(fig, ax, leg, pad: float = 0.05, gap: float = 0.02) -> None:
    fig.canvas.draw()
    r, dpi = fig.canvas.get_renderer(), fig.dpi
    leg_w = leg.get_window_extent(r).width / dpi
    leg_h = leg.get_window_extent(r).height / dpi
    xax_h = ax.xaxis.get_tightbbox(r).height / dpi
    yax_w = ax.yaxis.get_tightbbox(r).width / dpi
    side = leg_h
    W = yax_w + side + gap * side + leg_w + pad
    H = xax_h + side + pad
    fig.set_size_inches(W, H)
    fig.subplots_adjust(left=yax_w / W, right=(yax_w + side) / W, bottom=xax_h / H, top=1 - pad / H)


def _save(fig, out_pdf: str) -> None:
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf[:-4] + ".png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    print(f"wrote {out_pdf} (+ .png)", flush=True)


class _StageCounter:

    def __init__(self, model, num_programs: int, bins: int, device):
        self.model, self.bins = model, bins
        self.counts = torch.zeros(num_programs, bins, dtype=torch.long, device=device)

    def __call__(self, flat: Dict, V: torch.Tensor, **kw):
        y_hat, w = self.model(flat, V, **kw)
        done = (flat["visited"][:, 1:] > 0).all(-1) & (flat["current_node"] == 0)
        load = flat["load"][~done].clamp(0.0, 1.0)
        idx = w.argmax(-1)[~done] * self.bins + (load * self.bins).long().clamp(max=self.bins - 1)
        self.counts.view(-1).scatter_add_(0, idx, torch.ones_like(idx))
        return y_hat, w


def stage_counts(ckpt: str, instances: str, device: str) -> Dict:
    from ..settings.pomo_cvrp import rollout
    setting, a = _student_args("pomo_cvrp", ckpt, device)
    inst, sha = I.load(instances, "pomo_cvrp", _problem_size(a.ckpt))
    policy = setting.eval_policy("epb", a)
    counter = _StageCounter(policy.model, len(policy.bank), STAGE_BINS, policy.device)
    policy.model = counter
    d, x, q = inst["depot_xy"], inst["node_xy"], inst["node_demand"]
    n = q.size(0)
    with torch.no_grad():
        for s in range(0, n, STAGE_BATCH):
            e = min(s + STAGE_BATCH, n)
            rollout(policy, d[s:e].to(policy.device), x[s:e].to(policy.device), q[s:e].to(policy.device),
                    STAGE_STARTS)
            print(f"  {e}/{n} instances, {int(counter.counts.sum())} decisions", flush=True)
    name = os.path.basename(instances[5:]) if instances.startswith("file:") else instances
    return {"student": _name(ckpt), "instances": name, "instances_sha256": sha, "n": n, "starts": STAGE_STARTS,
            "bins": STAGE_BINS, "counts": counter.counts.cpu().tolist()}


def _p_used(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M, bins = counts.shape
    radius = int(np.ceil(3.0 * STAGE_SIGMA))
    k = np.exp(-(np.arange(-radius, radius + 1) ** 2) / (2.0 * STAGE_SIGMA ** 2))
    k /= k.sum()
    pad = np.concatenate([counts[:, :radius][:, ::-1], counts, counts[:, -radius:][:, ::-1]], axis=1)
    smooth = np.stack([np.convolve(pad[i], k, mode="valid") for i in range(M)])
    p_load = smooth / np.maximum(smooth.sum(axis=0, keepdims=True), 1e-12)
    centres = (np.arange(bins) + 0.5) / bins
    return p_load[:, ::-1], 1.0 - centres[::-1]


def _shares(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    share = counts.sum(axis=1) / max(counts.sum(), 1)
    rank = np.empty(len(share), dtype=int)
    rank[np.argsort(-share, kind="stable")] = np.arange(len(share))
    return share, rank


def stage_stats(counts: np.ndarray) -> None:
    share, rank = _shares(counts)
    p_used, used = _p_used(counts)
    raw = counts[:, ::-1]
    print(f"{'rank':>4} {'slot':>4} {'share%':>7} {'mean_used':>9} {'peak_used':>9}")
    for b in np.argsort(rank):
        mean_used = (raw[b] * used).sum() / max(raw[b].sum(), 1)
        print(f"H{rank[b]:02d}  {b:>4} {100 * share[b]:7.2f} {mean_used:9.3f} {used[p_used[b].argmax()]:9.3f}")


def stage_plot(counts: np.ndarray, groups: Dict, out_pdf: str) -> List[Tuple[str, float]]:
    import matplotlib as mpl
    from matplotlib.patches import Patch
    plt = _plt()
    M = counts.shape[0]
    sets = groups["sets"]
    if sorted(s for g in sets for s in g["slots"]) != list(range(M)):
        raise ValueError(f"the sets of --groups must partition the bank slots 0..{M - 1}")
    p_used, used = _p_used(counts)
    share, rank = _shares(counts)
    slots, colours = [], []
    for g in sets:
        for j, s in enumerate(g["slots"]):
            slots.append(s)
            colours.append(_lighten(g["colour"], STAGE_LIGHTEN * j / max(len(g["slots"]) - 1, 1)))
    x = np.r_[0.0, used, 1.0]
    P = p_used[slots]
    P = np.c_[P[:, :1], P, P[:, -1:]]
    out = []
    with mpl.rc_context(RC):
        fig, ax = plt.subplots(figsize=(30, 9))
        ax.stackplot(x, P, colors=colours, alpha=0.95, edgecolor="white", linewidth=0.6)
        top = 0
        for g in sets[:-1]:
            top += len(g["slots"])
            ax.plot(x, P[:top].sum(axis=0), color="white", linewidth=2.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.margins(0)
        ax.set_xlabel("used capacity", fontsize=STAGE_FS["label"])
        ax.set_ylabel("P(heuristic | used)", fontsize=STAGE_FS["label"])
        ax.tick_params(axis="both", labelsize=STAGE_FS["tick"])
        handles, labels = [], []
        for g in sets:
            ids = ", ".join(f"H{rank[s]:02d}" for s in g["slots"])
            tot = float(share[g["slots"]].sum())
            handles.append(Patch(facecolor=g["colour"], edgecolor="none"))
            labels.append(f"{g['stage']} · {g['name']}  [{ids}]  {100 * tot:.1f}%")
            out.append((labels[-1], tot))
        leg = ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=STAGE_FS["legend"],
                        frameon=False, ncol=1, handlelength=1.6, handletextpad=0.6, labelspacing=0.7, borderpad=0,
                        borderaxespad=0)
        _fit_to_legend(fig, ax, leg)
        _save(fig, out_pdf)
        plt.close(fig)
    return out


def stage(args) -> None:
    if args.hist:
        rec = json.load(open(args.hist))
    else:
        if not args.ckpt:
            raise SystemExit("stage needs --ckpt (or --hist)")
        rec = stage_counts(args.ckpt, args.instances or STAGE_INSTANCES, args.device)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"stage_{rec['student']}.json")
        json.dump(rec, open(path, "w"))
        print(f"wrote {path}")
    counts = np.asarray(rec["counts"], dtype=np.int64)
    print(f"{rec['student']}: {int(counts.sum())} decisions ({rec['n']} instances x {rec['starts']} starts)")
    stage_stats(counts)
    groups = json.load(open(args.groups or STAGE_GROUPS))
    for label, _ in stage_plot(counts, groups, os.path.join(args.out, f"stage_{rec['student']}.pdf")):
        print(" ", label)


def _traced(policy):
    from ..settings.pomo_tsp import RouterPolicy

    class Traced(RouterPolicy):
        def begin(self, locs, first):
            super().begin(locs, first)
            self.w = []

        def pick(self, vis, cur, step):
            nxt = super().pick(vis, cur, step)
            self.w.append(self.weights)
            return nxt

    traced = Traced.__new__(Traced)
    traced.__dict__.update(vars(policy))
    return traced


def tour_data(ckpt: str, i: int, instances: str, device: str, k_mode: str, v_mode: str,
              teacher_ckpt: Optional[str] = None) -> Tuple[Dict, np.ndarray, torch.Tensor]:
    from ..settings.pomo_tsp import construct, tour_costs
    setting, a = _student_args("pomo_tsp", ckpt, device, ["--k-mode", k_mode, "--v-mode", v_mode])
    inst, sha = I.load(instances, "pomo_tsp", _problem_size(a.ckpt))
    locs = inst["locs"]
    n, N, _ = locs.shape
    if not 0 <= i < n:
        raise ValueError(f"--instance {i}: the set has {n} instances")
    policy = _traced(setting.eval_policy("epb", a))
    _, chunk, S = setting.eval_layout(policy, a, N)
    s = i // chunk * chunk
    x = locs[s:s + chunk].to(policy.device)
    best = (float("inf"), None, None, None)
    for s0 in range(0, N, S):
        first = torch.arange(s0, min(s0 + S, N), device=x.device).expand(x.size(0), -1).contiguous()
        tours, _ = construct(policy, x, first)
        c, j = tour_costs(x, tours).min(1)
        c, j = float(c[i - s]), int(j[i - s])
        if c < best[0]:
            w = torch.stack(policy.w, 1).view(x.size(0), first.size(1), N - 1, -1)[i - s, j]
            best = (c, tours[i - s, j].cpu(), w.cpu(), s0 + j)
    cost, tour, w, start = best
    xy = locs[i].numpy()
    D = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    t = tour.numpy()
    visited = np.zeros(N, dtype=bool)
    visited[t[0]] = True
    nn = np.zeros(N - 1, dtype=bool)
    for k in range(N - 1):
        d = D[t[k]].copy()
        d[visited] = np.inf
        nn[k] = int(d.argmin()) == t[k + 1]
        visited[t[k + 1]] = True
    rec = {"student": _name(ckpt), "instances": instances, "instances_sha256": sha, "instance": i, "N": N,
           "k_mode": k_mode, "v_mode": v_mode, "start": start, "cost": cost, "tour": t.tolist(),
           "dominant": w.argmax(-1).tolist(), "nn_edge": nn.tolist(), "teacher_cost": None}
    if teacher_ckpt:
        ta = E.build_args(["--setting", "pomo_tsp", "--method", "teacher", "--teacher-ckpt", teacher_ckpt,
                           "--device", device])
        teacher = setting.eval_policy("teacher", ta)
        tchunk = setting.eval_layout(teacher, ta, N)[1]
        ts = i // tchunk * tchunk
        rec["teacher_cost"] = float(setting.eval_search(teacher, {"locs": locs[ts:ts + tchunk]}, ta)[i - ts])
    return rec, xy, w


def tour_plot(rec: Dict, xy: np.ndarray, names: Dict[str, str], M: int, out_pdf: str) -> None:
    import matplotlib as mpl
    from matplotlib.patches import FancyArrowPatch
    plt = _plt()
    t, top, nn = rec["tour"], rec["dominant"], rec["nn_edge"]
    with mpl.rc_context(RC):
        fig, ax = plt.subplots(figsize=(22, 13))
        cmap = plt.get_cmap("tab20", M)
        ax.scatter(xy[:, 0], xy[:, 1], c="black", s=22, zorder=2.5)
        ax.scatter(*xy[t[0]], c="tab:green", marker="s", s=380, zorder=5)
        for k in range(len(t) - 1):
            p0, p1 = tuple(xy[t[k]]), tuple(xy[t[k + 1]])
            if nn[k]:
                ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, color="black", lw=1.2,
                                             alpha=.85, zorder=2))
            else:
                ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=24, color=cmap(top[k]),
                                             lw=4.0, alpha=.95, zorder=3))
        ax.add_patch(FancyArrowPatch(tuple(xy[t[-1]]), tuple(xy[t[0]]), arrowstyle="-|>", mutation_scale=15,
                                     color="black", ls="--", lw=1.8, alpha=.7, zorder=2))
        present = sorted({top[k] for k in range(len(nn)) if not nn[k]})
        handles = [plt.Line2D([0], [0], color=cmap(m), lw=7, label=f"H{m:02d}  {names.get(str(m), '')}".rstrip())
                   for m in present]
        handles.append(plt.Line2D([0], [0], color="black", lw=2, label="NN edge"))
        handles.append(plt.Line2D([0], [0], color="black", ls="--", lw=3, label="return edge"))
        handles.append(plt.Line2D([0], [0], color="tab:green", marker="s", lw=0, markersize=20, label="start city"))
        title = f"cost = {rec['cost']:.3f}"
        if rec["teacher_cost"] is not None:
            title += f"    POMO = {rec['teacher_cost']:.3f}"
        leg = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=TOUR_FS["legend"],
                        frameon=False, title=title, title_fontsize=TOUR_FS["title"], alignment="left",
                        labelspacing=0.7, handlelength=3.0, borderaxespad=0, borderpad=0)
        ax.set_aspect("equal")
        ax.set_xlim(-.02, 1.02)
        ax.set_ylim(-.02, 1.02)
        ax.tick_params(labelsize=TOUR_FS["tick"])
        _fit_to_legend(fig, ax, leg)
        _save(fig, out_pdf)
        plt.close(fig)


def tour(args) -> None:
    rec, xy, w = tour_data(args.ckpt, args.instance, args.instances, args.device, args.k_mode, args.v_mode,
                           args.teacher_ckpt)
    nn = np.asarray(rec["nn_edge"])
    print(f"{rec['student']} instance {rec['instance']}: start {rec['start']}, cost {rec['cost']:.4f}"
          + (f", teacher {rec['teacher_cost']:.4f}" if rec["teacher_cost"] is not None else "")
          + f"; nearest-neighbour edges {int(nn.sum())}/{nn.size}")
    names = json.load(open(args.legend or TOUR_LEGEND, encoding="utf-8"))["names"]
    base = os.path.join(args.out, f"tour_{rec['student']}_inst{rec['instance']}")
    os.makedirs(args.out, exist_ok=True)
    json.dump(rec, open(base + ".json", "w"))
    tour_plot(rec, xy, names, w.size(-1), base + ".pdf")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m epb.figures", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="figure", required=True)
    st = sub.add_parser("stage", help="CVRP: P(dominant program | used capacity), stacked by program set")
    st.add_argument("--ckpt", help="pomo_cvrp student directory (best_model.pt, bank.json [, decode.json])")
    st.add_argument("--hist", help="a stage_<student>.json written before: re-plot without the rollout")
    st.add_argument("--groups", help="program sets (default: cvrp50_stages.json next to this module)")
    st.add_argument("--instances", help="main | file:<path> (default: the figure's 200 instances)")
    to = sub.add_parser("tour", help="TSP: the best multi-start tour, edges coloured by the dominant program")
    to.add_argument("--ckpt", required=True, help="pomo_tsp student directory (best_model.pt, bank.json, decode.json)")
    to.add_argument("--instance", type=int, default=9)
    to.add_argument("--instances", default="main", help="main | file:<path>")
    to.add_argument("--teacher-ckpt", help="POMO-TSP checkpoint: adds the teacher's multi-start cost to the title")
    to.add_argument("--legend", help="program names by bank slot (default: tsp100_legend.json next to this module)")
    to.add_argument("--k-mode", choices=("soft", "onehot"), default=TOUR_DECODE[0])
    to.add_argument("--v-mode", choices=("soft", "onehot"), default=TOUR_DECODE[1])
    for p in (st, to):
        p.add_argument("--out", default=os.path.join("results", "figures"))
        p.add_argument("--device", default="cuda")
        p.add_argument("--threads", type=int, help="torch CPU threads")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    {"stage": stage, "tour": tour}[args.figure](args)
