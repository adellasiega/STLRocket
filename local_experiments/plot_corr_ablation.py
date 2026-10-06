import glob
import matplotlib.pyplot as plt
import pandas as pd

df = pd.concat(pd.read_csv(f) for f in glob.glob("results/ablation/corr/*.csv"))
df = df[df.status == "ok"].astype({"depth_max": int})
df["corr_threshold"] = df.corr_threshold.map(lambda t: "none" if pd.isna(t) else f"{t:g}")
order = ["none", "0.99", "0.95"]
acc = "val_balanced_accuracy"


def heatmap(ax, mean, std, title, vmin=0, vmax=1):
    mean, std = mean[order], std[order]
    im = ax.imshow(mean.values, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            light = im.norm(mean.iat[i, j]) > 0.6
            ax.text(j, i, f"{mean.iat[i, j]:.3f}\n±{std.iat[i, j]:.3f}", ha="center", va="center",
                    color="black" if light else "white", fontsize=8)
    ax.set(xticks=range(len(order)), xticklabels=order, yticks=range(mean.shape[0]),
           yticklabels=mean.index, xlabel="corr threshold", ylabel="max_depth", title=title)
    return im


# Per dataset: mean ± std over runs
g = df.groupby(["dataset", "depth_max", "corr_threshold"])[acc]
mean, std = g.mean().unstack(), g.std().unstack()
datasets = mean.index.get_level_values(0).unique()
ncols = 5
nrows = -(-len(datasets) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), constrained_layout=True)
for ax, ds in zip(axes.flat, datasets):
    im = heatmap(ax, mean.loc[ds], std.loc[ds], ds)
for ax in axes.flat[len(datasets):]:
    ax.axis("off")
fig.colorbar(im, ax=axes, label="balanced accuracy", shrink=0.5)
fig.savefig("results/ablation/corr_heatmaps_per_dataset.pdf")

# Average over datasets: mean ± std across per-dataset means
per_ds = g.mean().groupby(["depth_max", "corr_threshold"])
m = per_ds.mean().unstack()
fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
im = heatmap(ax, m, per_ds.std().unstack(), f"Average over {len(datasets)} datasets", m.values.min(), m.values.max())
fig.colorbar(im, ax=ax, label="balanced accuracy")
fig.savefig("results/ablation/corr_heatmap_average.pdf")
