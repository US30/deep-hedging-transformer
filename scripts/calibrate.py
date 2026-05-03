"""
Calibrate rBergomi to SPX/VIX and save params to JSON.

Usage:
    python scripts/calibrate.py --train_end 2021-12-31 --out calibration/params.json
"""

import sys, os, argparse, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from deephedge.calibration.data import load_spx, realized_roughness
from deephedge.calibration.rbergomi_fit import calibrate


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SPX: 2000-01-01 → {args.train_end}")
    spx = load_spx(start="2000-01-01", end=args.train_end)

    print("Roughness estimate:")
    r = realized_roughness(spx)
    for q, v in r.items():
        print(f"  q={q}: H={v['H_estimate']:.4f}")

    params = calibrate(spx, n_paths=100_000, n_steps=252, device=device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved to {args.out}: {params}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_end", default="2021-12-31")
    p.add_argument("--out",       default="calibration/params.json")
    main(p.parse_args())
