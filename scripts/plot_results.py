#!/usr/bin/env python3
"""Render benchmark plots from the CSVs produced by ./build/benchmark.

Outputs:
  results/crossover.png     end-to-end lookup throughput, CPU vs GPU, vs batch size
  results/loadfactor.png    lookup throughput vs load factor, linear vs bucketed

Pure stdlib + matplotlib. If matplotlib is missing the script exits cleanly with
a hint instead of failing the build.
"""
import csv
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib not installed; run: pip install matplotlib")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")


def read_csv(name):
    with open(os.path.join(RESULTS, name)) as f:
        return list(csv.DictReader(f))


def plot_crossover():
    rows = [r for r in read_csv("benchmark.csv")
            if r["op"] == "lookup" and r["distribution"] == "uniform"]
    series = {}
    for r in rows:
        series.setdefault(r["impl"], []).append(
            (int(r["n"]), float(r["mkeys_per_s"])))
    plt.figure(figsize=(8, 5))
    for impl, pts in sorted(series.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=impl)
    plt.xscale("log")
    plt.xlabel("batch size (keys)")
    plt.ylabel("throughput (Mkeys/s, end-to-end incl. transfers)")
    plt.title("Lookup throughput: CPU vs GPU (uniform)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    out = os.path.join(RESULTS, "crossover.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print("wrote", out)


def plot_loadfactor():
    if not os.path.exists(os.path.join(RESULTS, "loadfactor.csv")):
        return
    rows = read_csv("loadfactor.csv")
    series = {}
    for r in rows:
        series.setdefault(r["impl"], []).append(
            (float(r["load_factor"]), float(r["mkeys_per_s"])))
    plt.figure(figsize=(8, 5))
    for impl, pts in sorted(series.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="s", label=impl)
    plt.xlabel("load factor")
    plt.ylabel("lookup throughput (Mkeys/s, kernel-only)")
    plt.title("Probing efficiency vs load factor")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(RESULTS, "loadfactor.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    plot_crossover()
    plot_loadfactor()
