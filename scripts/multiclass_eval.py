import argparse
import glob
import time
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Multiclass classification metrics evaluation and visualization"
    )
    p.add_argument("--keyword", type=str, default = "hw02")
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--stamp", type=str, default=None)
    p.add_argument("--recursive", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)

    pattern = (
        f"**/*{args.keyword}*metrics.csv" if args.recursive
        else f"*{args.keyword}*metrics.csv"
    )
    paths = sorted(glob.glob(str(outdir / pattern), recursive=args.recursive))

    dfs = []
    for p in paths:
        df = pl.read_csv(p)
        df = df.with_columns(pl.lit(Path(p).name).alias("run_file"))
        dfs.append(df)


    all_df = pl.concat(dfs, how="vertical")

    # save aggregated CSV
    agg_csv = outdir / f"{args.keyword}_aggregated_{args.stamp}.csv"
    all_df.write_csv(agg_csv)

    # -------- boxplot --------
    metrics = [
        ("Accuracy", "train_accuracy", "test_accuracy"),
        ("Precision", "train_precision", "test_precision"),
        ("Recall", "train_recall", "test_recall"),
        ("F1", "train_f1", "test_f1"),
    ]

    data = []
    positions = []
    labels = []

    # set the gap between train and test boxplots, and between different metrics
    pos = 1
    gap = 2

    for name, train, test in metrics:
        data.append(all_df[train].drop_nulls().to_numpy())
        data.append(all_df[test].drop_nulls().to_numpy())
        positions.extend([pos, pos + 1])
        labels.extend([f"Train\n{name}", f"Test\n{name}"])
        pos += 2 + gap

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, positions=positions, widths=0.6, showfliers=True)
    plt.xticks(positions, labels)
    plt.ylabel("Score")
    plt.title(
        f"Boxplot keyword={args.keyword}, runs={all_df.height}"
    )
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    plot_path = outdir / f"{args.keyword}_boxplot_{args.stamp}.pdf"
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    main()
