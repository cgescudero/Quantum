"""Figura 03: controles pareados versus aleatorios en z y masa.

Comando:
    # python3 plot_03_matched_vs_random_controls_v1.py --overwrite
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting_external_xmp_ml_qml_utils_v1 import (
    COLORS,
    SCRIPT_DIR,
    find_column,
    parse_common_args,
    print_outputs,
    read_csv_checked,
    require_numeric_columns,
    save_figure,
    setup_matplotlib,
    validate_output_dir,
    write_table,
)


def split_samples(matched: pd.DataFrame, random: pd.DataFrame, z_col: str, mass_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye tabla larga con positivos, controles pareados y aleatorios."""
    rows = []
    specs = [
        (matched[matched["external_xmp_label"] == 1], "positivos externos"),
        (matched[matched["external_xmp_label"] == 0], "controles pareados"),
        (random[random["external_xmp_label"] == 0], "controles aleatorios"),
    ]
    for data, label in specs:
        clean = require_numeric_columns(data, [z_col, mass_col], label)
        for _, row in clean.iterrows():
            rows.append({"sample": label, "z": row[z_col], "log_stellar_mass": row[mass_col]})
    long = pd.DataFrame(rows)
    summary = long.groupby("sample", as_index=False).agg(
        n=("z", "size"), median_z=("z", "median"), median_log_stellar_mass=("log_stellar_mass", "median")
    )
    return long, summary


def main() -> None:
    """Genera histogramas normalizados para z y logM."""
    args = parse_common_args("Figura 03: matched vs random controls")
    output_dir = validate_output_dir(args.output_dir)
    setup_matplotlib()
    matched = read_csv_checked(SCRIPT_DIR / "results_external_xmp_supervised_dataset_v1/dataset_external_xmp_matched_controls_v1.csv", ["external_xmp_label"])
    random = read_csv_checked(SCRIPT_DIR / "results_external_xmp_supervised_dataset_v1/dataset_external_xmp_random_controls_v1.csv", ["external_xmp_label"])
    z_col = find_column(matched, ["z", "redshift"], "redshift")
    mass_col = find_column(matched, ["log_stellar_mass", "stellar_mass_log", "log_mass"], "masa estelar")
    long, summary = split_samples(matched, random, z_col, mass_col)
    colors = {"positivos externos": COLORS["external"], "controles pareados": COLORS["matched"], "controles aleatorios": COLORS["random"]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, col, xlabel in [(axes[0], "z", "redshift z"), (axes[1], "log_stellar_mass", "log stellar mass")]:
        bins = np.histogram_bin_edges(long[col], bins="auto")
        for sample, data in long.groupby("sample"):
            ax.hist(data[col], bins=bins, density=True, histtype="step", lw=2.0, label=sample, color=colors[sample])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Densidad normalizada")
    axes[0].set_title("A. Pareamiento en redshift")
    axes[1].set_title("B. Pareamiento en masa")
    axes[1].legend(frameon=False)
    fig.suptitle("Matched es el benchmark científicamente relevante; random introduce sesgo de selección", y=1.03)
    paths = save_figure(fig, output_dir, "fig03_matched_vs_random_controls", args.overwrite)
    table_path = write_table(summary, output_dir, "fig03_matched_vs_random_summary.csv", args.overwrite)
    print_outputs([*paths, table_path])


if __name__ == "__main__":
    main()
