import argparse
import json

from . import TESTS, get


def _common_args(p):
    p.add_argument("--teacher-ckpt", help="teacher checkpoint (POMO: yd-kwon/POMO format; LEHD: NCO_code format)")
    p.add_argument("--nco-root", help="clone of CIAM-Group/NCO_code (LEHD tests; default $NCO_ROOT)")
    p.add_argument("--student", help="student directory (best_model.pt + bank.json) or a name under epb/students "
                                     "(default: the test's student)")
    p.add_argument("--bank", help="bank JSON (default: <student>/bank.json)")
    p.add_argument("--config", help="experiment parameters (default: the test's <test>.json)")
    p.add_argument("--out", help="output directory (default results/counterfactual/<test>)")
    p.add_argument("--inst-lo", type=int, help="first instance (see the test)")
    p.add_argument("--inst-hi", type=int, help="end of the instance range (exclusive)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-unverified", action="store_true",
                   help="accept shards / batch outputs / caches written without an experiment fingerprint (older "
                        "versions) without checking them")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="test", required=True, metavar="{" + ",".join(TESTS) + "}")
    for name in TESTS:
        p = sub.add_parser(name)
        _common_args(p)
        get(name).add_args(p)
    a = ap.parse_args()
    test = get(a.test)
    a.student = a.student or test.STUDENT
    a.out = a.out or f"results/counterfactual/{test.NAME}"
    cfg = json.load(open(a.config or test.CONFIG))
    test.run(a, cfg)


if __name__ == "__main__":
    main()
