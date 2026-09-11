import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# DATA: ZZ kernel/reuploading, seed 42, 40 train + 20 test
# ============================================================

environments = ["Aer ideal", "FakeSherbrooke", "IBM Kingston"]

# Use the values from the run you want to report as definitive.
#zz
#pr_auc  = [0.9188, 0.9408, 0.9124]
#roc_auc = [0.9100, 0.9300, 0.9200]
#bal_acc = [0.9000, 0.8000, 0.9000]
#f1      = [0.9000, 0.8000, 0.9000]

#reuploading
pr_auc  = [0.9319, 0.8984 ,0.8708]
roc_auc = [0.92, 0.88, 0.85]
bal_acc = [0.85, 0.75, 0.65]
f1      = [0.8696, 0.7826, 0.6667]

metrics = {
    "PR-AUC": pr_auc,
    "ROC-AUC": roc_auc,
    "Balanced accuracy": bal_acc,
    "F1": f1,
}

# ============================================================
# PLOT
# ============================================================

x = np.arange(len(environments))
width = 0.18

fig, ax = plt.subplots(figsize=(10, 6))

offsets = [-1.5, -0.5, 0.5, 1.5]

for offset, (metric_name, values) in zip(offsets, metrics.items()):
    ax.bar(
        x + offset * width,
        values,
        width=width,
        label=metric_name,
    )

# Labels
ax.set_xticks(x)
ax.set_xticklabels(environments)

ax.set_ylabel("Score")
ax.set_ylim(0.6, 1.0)

ax.set_title(
   # "ZZ quantum kernel: ideal, noisy, and real-hardware performance"
    "reuploading quantum kernel: ideal, noisy, and real-hardware performance"
)

ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False)

plt.tight_layout()

plt.savefig(
    "reuploading_aer_fake_qpu_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "reuploading_aer_fake_qpu_comparison.pdf",
    bbox_inches="tight"
)

plt.show()