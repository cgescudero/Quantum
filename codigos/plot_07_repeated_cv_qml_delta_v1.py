"""Figura 07: repeated CV para core_4 y metallicity_5.

Comando:
    # python3 plot_07_repeated_cv_qml_delta_v1.py --overwrite
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from plotting_external_xmp_ml_qml_utils_v1 import (
    COLORS,
    SCRIPT_DIR,
    parse_common_args,
    print_outputs,
    read_csv_checked,
    require_numeric_columns,
    save_figure,
    setup_matplotlib,
    validate_output_dir,
    write_table,
)


def main() -> None:
    """Genera boxplot con puntos por semilla de CV para delta PR-AUC."""
    args = parse_common_args("Figura 07: repeated CV delta QML")
    output_dir = validate_output_dir(args.output_dir)
    setup_matplotlib()
    delta = read_csv_checked(
        SCRIPT_DIR / "results_external_xmp_qml_confirm_core_features_v1/delta_by_cv_seed.csv",
        ["feature_subset", "cv_seed", "delta_qml_minus_classical_pr_auc"],
    )
    summary = read_csv_checked(
        SCRIPT_DIR / "results_external_xmp_qml_confirm_core_features_v1/delta_distribution_summary.csv",
        ["feature_subset", "delta_pr_auc_mean", "fraction_positive_delta"],
    )
    delta = require_numeric_columns(delta, ["delta_qml_minus_classical_pr_auc"], "delta_by_cv_seed")
    subsets = ["core_4", "metallicity_5"]
    data = [delta.loc[delta["feature_subset"] == subset, "delta_qml_minus_classical_pr_auc"].to_numpy() for subset in subsets]
    if any(len(values) == 0 for values in data):
        raise ValueError("Faltan datos de core_4 o metallicity_5 en delta_by_cv_seed.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    box = ax.boxplot(data, labels=subsets, patch_artist=True, showfliers=False)
    for patch in box["boxes"]:
        patch.set_facecolor(COLORS["quantum"])
        patch.set_alpha(0.35)
    rng = np.random.default_rng(42)
    for i, values in enumerate(data, start=1):
        x = rng.normal(i, 0.035, len(values))
        ax.scatter(x, values, s=35, color=COLORS["quantum_alt"], alpha=0.85, zorder=3)
    ax.axhline(0.0, color="black", lw=1.0)
    for i, subset in enumerate(subsets, start=1):
        row = summary[summary["feature_subset"] == subset].iloc[0]
        ax.text(i, max(data[i - 1]) + 0.006, f"media={row['delta_pr_auc_mean']:+.3f}\nfrac+={row['fraction_positive_delta']:.2f}", ha="center")
    ax.set_ylabel("Delta PR-AUC (mejor QML - mejor clásico)")
    ax.set_title("Repeated CV: comparabilidad y mejora ligera en subconjuntos compactos")
    paths = save_figure(fig, output_dir, "fig07_repeated_cv_qml_delta", args.overwrite)
    table_path = write_table(delta, output_dir, "fig07_repeated_cv_qml_delta_table.csv", args.overwrite)
    print_outputs([*paths, table_path])


if __name__ == "__main__":
    main()
