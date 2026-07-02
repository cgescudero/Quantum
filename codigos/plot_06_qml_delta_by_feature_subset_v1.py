"""Figura 06: delta QML - ML por subconjunto físico.

Comando:
    # python3 plot_06_qml_delta_by_feature_subset_v1.py --overwrite
"""

from __future__ import annotations

import matplotlib.pyplot as plt

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
    """Grafica delta PR-AUC = mejor QML - mejor clásico por subset."""
    args = parse_common_args("Figura 06: delta QML por subset físico")
    output_dir = validate_output_dir(args.output_dir)
    setup_matplotlib()
    path = SCRIPT_DIR / "results_external_xmp_qml_feature_scan_v1/matched_best_classical_vs_qml.csv"
    df = read_csv_checked(path, ["feature_subset", "delta_qml_minus_classical_pr_auc"])
    df = require_numeric_columns(df, ["delta_qml_minus_classical_pr_auc"], "feature scan")
    order = ["core_4", "metallicity_5", "oxygen_6", "all_10", "nebular_no_mass_z_8"]
    df["order"] = df["feature_subset"].map({name: i for i, name in enumerate(order)}).fillna(99)
    table = df.sort_values("order").drop(columns="order")
    colors = [COLORS["quantum"] if v > 0 else COLORS["negative"] for v in table["delta_qml_minus_classical_pr_auc"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(table["feature_subset"], table["delta_qml_minus_classical_pr_auc"], color=colors)
    ax.axhline(0.0, color="black", lw=1.0)
    for bar, value in zip(bars, table["delta_qml_minus_classical_pr_auc"]):
        va = "bottom" if value >= 0 else "top"
        offset = 0.003 if value >= 0 else -0.003
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.3f}", ha="center", va=va)
    ax.set_ylabel("Delta PR-AUC (mejor QML - mejor clásico)")
    ax.set_xlabel("Subconjunto físico")
    ax.set_title("QML favorable sólo en espacios compactos; no mejora globalmente")
    ax.tick_params(axis="x", rotation=20)
    paths = save_figure(fig, output_dir, "fig06_qml_delta_by_feature_subset", args.overwrite)
    table_path = write_table(table, output_dir, "fig06_qml_delta_by_feature_subset_table.csv", args.overwrite)
    print_outputs([*paths, table_path])


if __name__ == "__main__":
    main()
