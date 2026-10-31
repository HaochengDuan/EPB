import importlib

TESTS = ("hull", "isolation", "corridor", "demand_density")


def get(name: str):
    if name not in TESTS:
        raise ValueError(f"unknown test {name!r}; one of {TESTS}")
    return importlib.import_module(f"{__name__}.{name}")
