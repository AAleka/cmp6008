#!/usr/bin/env python
"""
NeRF vs 3D Gaussian Splatting comparison runner (nerfstudio).

Trains each method on the SAME scene with identical data settings, records
wall-clock training time, evaluates (PSNR/SSIM/LPIPS/FPS), then writes a
comparison table (Markdown + CSV) for the report.

Run inside the `nerfstudio` conda env:

    conda activate nerfstudio
    python scripts/run_comparison.py                       # flowers, nerfacto + splatfacto
    python scripts/run_comparison.py --scene treehill
    python scripts/run_comparison.py --methods nerfacto splatfacto instant-ngp
    python scripts/run_comparison.py --iters 30000 --downscale 4

    # Re-build the table without retraining (reuses existing eval JSONs):
    python scripts/run_comparison.py --summarize-only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
TIMESTAMP = "run"  # fixed -> deterministic output paths


def env_with_wsl_cuda() -> dict:
    """nerfstudio on WSL needs libcuda from /usr/lib/wsl/lib on the loader path."""
    env = os.environ.copy()
    wsl_lib = "/usr/lib/wsl/lib"
    env["LD_LIBRARY_PATH"] = f"{wsl_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def run(cmd: list[str], env: dict) -> None:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, env=env)


def train(method: str, scene: str, data_dir: Path, out_dir: Path,
          iters: int, downscale: int, env: dict) -> float:
    """Train one method; return wall-clock seconds."""
    run_dir = out_dir / scene / method / TIMESTAMP
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)  # clean previous run for reproducibility

    cmd = [
        "ns-train", method,
        "--data", str(data_dir),
        "--output-dir", str(out_dir),
        "--timestamp", TIMESTAMP,
        "--max-num-iterations", str(iters),
        "--vis", "tensorboard",
        "--viewer.quit-on-train-completion", "True",
        "colmap", "--colmap-path", "sparse/0",
        "--downscale-factor", str(downscale),
    ]
    start = time.time()
    run(cmd, env)
    return time.time() - start


def evaluate(method: str, scene: str, out_dir: Path, env: dict) -> Path:
    config = out_dir / scene / method / TIMESTAMP / "config.yml"
    metrics = out_dir / scene / f"{method}_metrics.json"
    run(["ns-eval", "--load-config", str(config),
         "--output-path", str(metrics)], env)
    return metrics


def checkpoint_size_mb(method: str, scene: str, out_dir: Path) -> float | None:
    ckpt_dir = out_dir / scene / method / TIMESTAMP / "nerfstudio_models"
    if not ckpt_dir.exists():
        return None
    total = sum(f.stat().st_size for f in ckpt_dir.glob("*.ckpt"))
    return round(total / (1024 * 1024), 1) if total else None


def load_metrics(path: Path) -> dict:
    """ns-eval writes {... 'results': {psnr, ssim, lpips, fps, ...}}."""
    with open(path) as f:
        data = json.load(f)
    return data.get("results", data)


def summarize(scene: str, methods: list[str], out_dir: Path) -> None:
    timings = {}
    timings_csv = out_dir / scene / "timings.csv"
    if timings_csv.exists():
        with open(timings_csv) as f:
            for row in csv.DictReader(f):
                timings[row["method"]] = float(row["train_seconds"])

    rows = []
    for method in methods:
        metrics_path = out_dir / scene / f"{method}_metrics.json"
        if not metrics_path.exists():
            print(f"  (skip {method}: no metrics at {metrics_path})")
            continue
        m = load_metrics(metrics_path)
        secs = timings.get(method)
        rows.append({
            "method": method,
            "train_min": round(secs / 60, 1) if secs else None,
            "psnr": round(m.get("psnr"), 2) if m.get("psnr") is not None else None,
            "ssim": round(m.get("ssim"), 4) if m.get("ssim") is not None else None,
            "lpips": round(m.get("lpips"), 4) if m.get("lpips") is not None else None,
            "fps": round(m.get("fps"), 1) if m.get("fps") is not None else None,
            "size_mb": checkpoint_size_mb(method, scene, out_dir),
        })

    if not rows:
        print("No metrics found to summarize. Run training first.")
        return

    # CSV
    csv_path = out_dir / scene / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Markdown
    headers = ["Method", "Train (min)", "PSNR ↑", "SSIM ↑",
               "LPIPS ↓", "FPS ↑", "Size (MB) ↓"]
    keys = ["method", "train_min", "psnr", "ssim", "lpips", "fps", "size_mb"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        cells = ["-" if r[k] is None else str(r[k]) for k in keys]
        lines.append("| " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    md_path = out_dir / scene / "comparison.md"
    md_path.write_text(md + "\n")

    print(f"\n### NeRF vs 3DGS — scene: {scene}\n")
    print(md)
    print(f"\nWrote: {md_path}\n       {csv_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="NeRF vs 3DGS comparison runner")
    p.add_argument("--scene", default="flowers")
    p.add_argument("--methods", nargs="+", default=["nerfacto", "splatfacto"])
    p.add_argument("--iters", type=int, default=30000)
    p.add_argument("--downscale", type=int, default=4)
    p.add_argument("--summarize-only", action="store_true",
                   help="Skip training/eval; rebuild table from existing JSONs.")
    args = p.parse_args()

    out_dir = PROJ_ROOT / "outputs"
    data_dir = PROJ_ROOT / "data" / "mipnerf360" / args.scene

    if args.summarize_only:
        summarize(args.scene, args.methods, out_dir)
        return 0

    if not data_dir.exists():
        print(f"ERROR: scene not found: {data_dir}", file=sys.stderr)
        avail = PROJ_ROOT / "data" / "mipnerf360"
        if avail.exists():
            print(f"Available: {[d.name for d in avail.iterdir() if d.is_dir()]}",
                  file=sys.stderr)
        return 1

    env = env_with_wsl_cuda()
    (out_dir / args.scene).mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print(f" Scene:      {args.scene}")
    print(f" Methods:    {args.methods}")
    print(f" Iterations: {args.iters}    Downscale: {args.downscale}")
    print("=" * 62)

    timings_csv = out_dir / args.scene / "timings.csv"
    with open(timings_csv, "w", newline="") as f:
        csv.writer(f).writerow(["method", "train_seconds"])

    for method in args.methods:
        print(f"\n>>> Training [{method}] on [{args.scene}] ...")
        secs = train(method, args.scene, data_dir, out_dir,
                     args.iters, args.downscale, env)
        with open(timings_csv, "a", newline="") as f:
            csv.writer(f).writerow([method, round(secs, 1)])
        print(f">>> [{method}] trained in {secs/60:.1f} min")

        print(f">>> Evaluating [{method}] ...")
        evaluate(method, args.scene, out_dir, env)

    print("\n>>> Building comparison table ...")
    summarize(args.scene, args.methods, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
