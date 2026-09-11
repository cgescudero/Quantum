from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing the 10 summary CSV files
RESULTS_DIR = Path(
    r"C:\Users\flor\projects\github\Quantum\codigos\results_ibm_quantum_test\zz\seeds_varied"
)

# Put here the exact 10 data seeds used
SEEDS = [1, 5, 10, 20, 30, 42, 50, 60, 80, 100]

# Models as they appear in the CSV
MODELS = [
    "qksvc_zz",
    "svc_rbf",
    "svc_linear",
    "random_forest",
]

LABELS = {
    "qksvc_zz": "QK-SVC ZZ",
    "svc_rbf": "SVC RBF",
    "svc_linear": "SVC linear",
    "random_forest": "Random Forest",
}

METRICS = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "bal_acc": "Balanced accuracy",
    "f1": "F1 score",
}


# ============================================================
# LOAD FILES
# ============================================================

files = sorted(
    RESULTS_DIR.glob("test_ibm_quantum_summary_*.csv")
)

print(f"Found {len(files)} CSV files")

if len(files) != 10:
    print("WARNING: expected 10 files.")


all_runs = []

for seed, file in zip(SEEDS, files):

    df = pd.read_csv(file)

    df["data_seed"] = seed
    df["source_file"] = file.name

    all_runs.append(df)

results = pd.concat(all_runs, ignore_index=True)

print("\nModels found:")
print(results["model"].unique())


# ============================================================
# KEEP ONLY THE MODELS OF INTEREST
# ============================================================

results = results[
    results["model"].isin(MODELS)
].copy()


# ============================================================
# SUMMARY
# ============================================================

summary = (
    results
    .groupby("model")[list(METRICS.keys())]
    .agg(["mean", "std"])
)

print("\nMean ± standard deviation:")
print(summary)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9),
    sharey=True
)

axes = axes.flatten()

x = np.arange(len(MODELS))

rng = np.random.default_rng(1234)


for ax, (metric, metric_label) in zip(
    axes,
    METRICS.items()
):

    for i, model in enumerate(MODELS):

        values = results.loc[
            results["model"] == model,
            metric
        ].to_numpy()

        # Small horizontal jitter so individual seeds can be seen
        jitter = rng.uniform(
            -0.08,
            0.08,
            size=len(values)
        )

        # Individual seed results
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            alpha=0.55,
            s=35,
            zorder=2
        )

        # Mean and sample standard deviation
        mean = np.mean(values)
        std = np.std(values, ddof=1)

        ax.errorbar(
            i + 0.15,
            mean,
            yerr=std,
            fmt="D",
            markersize=7,
            capsize=5,
            linewidth=2,
            zorder=3
        )

    ax.set_title(
        metric_label,
        fontsize=14
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [LABELS[m] for m in MODELS],
        rotation=25,
        ha="right"
    )

    ax.set_ylim(0.5, 1.0)

    ax.grid(
        axis="y",
        alpha=0.25
    )


axes[0].set_ylabel("Score")
axes[2].set_ylabel("Score")

fig.suptitle(
    "Performance across 10 matched train/test splits",
    fontsize=16
)

fig.text(
    0.5,
    0.01,
    "Points: individual data seeds   |   Diamond: mean ± standard deviation",
    ha="center",
    fontsize=11
)

plt.tight_layout(
    rect=[0, 0.04, 1, 0.95]
)

plt.savefig(
    "ten_seed_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "ten_seed_performance.pdf",
    bbox_inches="tight"
)

plt.show()