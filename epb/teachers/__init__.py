"""Teacher code: the POMO model definitions (from yd-kwon/POMO), the loaders of the LEHD models (CIAM-Group/NCO_code,
imported from a clone of that repository) and the decision-step generators that `python -m epb.generate` runs.

  pomo_tsp    POMO TSP model; single-file decision-step generator      python -m epb.teachers.pomo_tsp --help
  pomo_cvrp   POMO CVRP model; sharded decision-step generator         python -m epb.teachers.pomo_cvrp --help
  lehd_tsp    LEHD TSP per-step generator                              python -m epb.teachers.lehd_tsp --help
  lehd_cvrp   LEHD CVRP loader (with device patches), K-bucketed generator   python -m epb.teachers.lehd_cvrp --help
              and the teacher's greedy order, constrained replay and sub-problem greedy used by the evaluation

A generator runs in a process of its own (`command`), so its random stream starts from its seed exactly as in a fresh
run. Teacher checkpoints are not part of the package.
"""
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple


def command(module: str, args: Iterable) -> Tuple[List[str], Dict[str, str]]:
    """(argv, env) that run `python -m <this package>.<module> <args>` in a fresh process with this package
    importable from any working directory."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return [sys.executable, "-m", f"{__name__}.{module}", *[str(x) for x in args]], env


def nco_dir(nco_root: Optional[str]) -> str:
    """The directory that holds NCO_code's `LEHD` package: <clone>/single_objective, or nco_root itself (env NCO_ROOT
    if nco_root is not given)."""
    root = nco_root or os.environ.get("NCO_ROOT")
    if not root:
        raise ValueError("the LEHD teacher needs --nco-root (a clone of CIAM-Group/NCO_code) or NCO_ROOT")
    root = os.path.abspath(os.path.expanduser(str(root)))
    sub = os.path.join(root, "single_objective")
    d = sub if os.path.isdir(os.path.join(sub, "LEHD")) else root
    if not os.path.isdir(os.path.join(d, "LEHD")):
        raise FileNotFoundError(f"no LEHD package under {root}: clone https://github.com/CIAM-Group/NCO_code and pass "
                                f"the clone (or its single_objective/ directory)")
    return d


def add_nco_path(nco_root: Optional[str]) -> str:
    """Put NCO_code's single_objective/ first on sys.path (so `import LEHD...` works) and return it."""
    d = nco_dir(nco_root)
    if d not in sys.path:
        sys.path.insert(0, d)
    return d
