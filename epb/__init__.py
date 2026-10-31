"""EPB: distil a neural routing policy into a bank of LLM-written heuristic programs and an attention router.

  python -m epb.generate --config epb/configs/pomo_cvrp50.yaml
  python -m epb.train    --config epb/configs/pomo_cvrp50.yaml [--set train.lr=3e-4] [--resume RUN_DIR]
"""
