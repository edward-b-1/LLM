import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_names", nargs="+",
                        help="One or more run names to plot")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--no-val",  action="store_true",
                        help="Only plot train loss")
    parser.add_argument("--no-train", action="store_true",
                        help="Only plot val loss")
    args = parser.parse_args()

    fig, ax = plt.subplots()

    for run_name in args.run_names:
        path = os.path.join(args.log_dir, f"{run_name}.csv")
        if not os.path.exists(path):
            print(f"Log file not found: {path}")
            continue

        df = pd.read_csv(path)
        train = df[df["train_loss"].notna() & (df["train_loss"] != "")]
        val   = df[df["val_loss"].notna()   & (df["val_loss"]   != "")]

        if not args.no_train:
            loss = train["train_loss"].astype(float)
            steps = train["step"]
            bin_width = max(1, int(steps.max()) // 50)
            bin_edges = range(0, int(steps.max()) + bin_width + 1, bin_width)
            bins = pd.cut(steps, bins=bin_edges, include_lowest=True)
            mean  = loss.groupby(bins).mean()
            lo    = loss.groupby(bins).quantile(0.05)
            hi    = loss.groupby(bins).quantile(0.95)
            midpoints = mean.index.map(lambda b: b.mid)
            ax.plot(midpoints, mean.values, label=f"{run_name} train")
            ax.fill_between(midpoints, lo.values, hi.values, alpha=0.2)
        if not args.no_val:
            ax.plot(val["step"], val["val_loss"].astype(float),
                    label=f"{run_name} val", linewidth=2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    #ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
