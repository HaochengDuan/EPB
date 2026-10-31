import argparse

from . import CASES, get
from . import common as C


def main():
    ap = argparse.ArgumentParser(prog="python -m epb.interventions")
    ap.add_argument("case", choices=CASES)
    ap.add_argument("--teacher-ckpt", help="POMO teacher checkpoint (yd-kwon/POMO format), for teacher_rule")
    ap.add_argument("--parts", help="comma list of teacher_rule, rules_alone (band_cvrp50), divergence "
                                    "(default: all of the case)")
    ap.add_argument("--gammas", help="teacher_rule: comma list of cells, one weight per rule joined by '/'")
    ap.add_argument("--teacher-rule", help="teacher_rule: recipe | injected (band_tsp100), same (band_cvrp50)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    case = get(a.case)
    a.parts = a.parts or ",".join(case.PARTS)
    a.gammas = a.gammas or case.GAMMAS
    a.teacher_rule = a.teacher_rule or case.DEFAULT_TEACHER_RULE
    C.run(case, a)


if __name__ == "__main__":
    main()
