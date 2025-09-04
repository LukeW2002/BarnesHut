#!/usr/bin/env python3
"""
Make two bar charts from morton_bench.csv:
1) Grouped horizontal bars: total_ms, ensure_ms, encode_ms, sort_ms vs N
2) Horizontal bars: ratio_sum_over_total vs N
"""

import argparse
import math
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="morton_bench.csv",
                    help="Path to CSV (default: morton_bench.csv)")
    ap.add_argument("-o", "--out-prefix", default="morton",
                    help="Output filename prefix (default: morton)")
    args = ap.parse_args()

    # Load and sanity-check
    df = pd.read_csv(args.input)
    expected = ["N", "total_ms", "ensure_ms", "encode_ms", "sort_ms", "ratio_sum_over_total"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        sys.exit(f"CSV is missing columns: {missing}")

    df = df.sort_values("N").reset_index(drop=True)

    # ------------ Figure 1: timing grouped bars ------------
    cats = ["total_ms", "ensure_ms", "encode_ms", "sort_ms"]
    n = len(df)
    k = len(cats)
    y = np.arange(n)
    group_height = 0.8
    bar_h = group_height / k
    offsets = (np.arange(k) - (k - 1) / 2.0) * bar_h

    plt.figure(figsize=(10, max(4, n * 0.35)))
    for i, cat in enumerate(cats):
        plt.barh(y + offsets[i], df[cat].values, height=bar_h, label=cat)

    plt.yticks(y, df["N"].astype(str).tolist())
    plt.xlabel("Milliseconds")
    plt.ylabel("N")
    plt.title("Morton Pipeline Timing by N (grouped bars)")
    plt.legend(loc="best")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    bars_path = f"{args.out_prefix}_bars.png"
    plt.savefig(bars_path, dpi=150, bbox_inches="tight")

    # ------------ Figure 2: ratio bars ------------
    plt.figure(figsize=(8, max(4, n * 0.35)))
    plt.barh(y, df["ratio_sum_over_total"].values, height=min(0.7 * group_height / k + 0.1, 0.8))
    plt.yticks(y, df["N"].astype(str).tolist())
    plt.xlabel("Ratio (sum of parts / total)")
    plt.ylabel("N")
    plt.title("Stage Sum vs Whole Ratio by N")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    ratio_path = f"{args.out_prefix}_ratio.png"
    plt.savefig(ratio_path, dpi=150, bbox_inches="tight")

    print(f"Saved:\n  {os.path.abspath(bars_path)}\n  {os.path.abspath(ratio_path)}")


if __name__ == "__main__":
    main()

