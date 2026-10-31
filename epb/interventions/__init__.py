import importlib

CASES = ("band_tsp100", "band_cvrp50")


def get(name: str):
    if name not in CASES:
        raise ValueError(f"unknown case {name!r}; one of {CASES}")
    return importlib.import_module(f"{__name__}.{name}")
