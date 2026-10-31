import json, os

HERE = os.path.dirname(os.path.abspath(__file__))

ROOT = os.path.dirname(HERE)
from policies.tsp_policy import TSPParams
from policies.cvrp_policy import CVRPParams

FROZEN = os.path.join(ROOT, "policies", "frozen_params.json")

def load_frozen(need="tsp"):
    with open(FROZEN) as f:
        d = json.load(f)
    return TSPParams(**d["tsp"]), CVRPParams(**d["cvrp"]), d

