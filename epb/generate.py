"""Write the decision-step dataset of an experiment: run the teacher (generate.teacher_ckpt) on generate.num_instances
instances and store every decision step at data.path, in the format the setting's loader reads.

  python -m epb.generate --config epb/configs/pomo_cvrp50.yaml [--set generate.num_instances=1000]
"""
import argparse

from . import settings as S
from .train import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if not cfg.generate.teacher_ckpt:
        raise ValueError("generate.teacher_ckpt is not set (the teacher checkpoint is not part of this package)")
    S.get(cfg.setting.name).generate(cfg)
    no_effect = [k for k in cfg.unread() if k.startswith("generate.") and k in cfg.explicit]
    if no_effect:
        print(f"[config] set but not used by the {cfg.setting.name} generator: {no_effect}")


if __name__ == "__main__":
    main()
