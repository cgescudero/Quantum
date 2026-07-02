"""Figura 08: métricas top-k de la señal QML compacta.

Comando:
    # python3 plot_08_qml_topk_delta_v1.py --overwrite
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

METRICS = [
    ("delta_precision_at_20_mean", "delta precision@20", COLORS["quantum"]),
    ("delta_precision_at_50_mean", "delta precision@50", COLORS["quantum_alt"]),
    ("delta_recall_at_top5percent_mean", "delta recall@top5%", COLORS["classical_alt"]),
]


def main() -> None:
    """Grafica deltas top-k medios por subset compacto."""
    args = parse_common_args("Figura 08: deltas top-k QML")
    output_dir = validate_output_dir(args.output_dir)
    setup_matplotlib()
    required = ["feature_subset"] + [name for name, _, _ in METRICS]
    df = read_csv_checked(SCRIPT_DIR / "results_external_xmp_qml_confirm_core_features_v1/delta_distribution_summary.csv", required)
    df = require_numeric_columns(df, [name for name, _, _ in METRICS], "delta_distribution_summary")
    subsets = ["core_4", "metallicity_5"]
    table = df[df["feature_subset"].isin(subsets)].copy()
    long_rows = []
    for _, row in table.iterrows():
        for col, label, _ in METRICS:
            long_rows.append({"feature_subset": row["feature_subset"], "metric": label, "delta": row[col]})
    long = pd.DataFrame(long_rows)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(subsets))
    width = 0.24
    for j, (_, label, color) in enumerate(METRICS):
        values = [float(long[(long["feature_subset"] == subset) & (long["metric"] == label)]["delta"].iloc[0]) for subset in subsets]
        ax.bar(x + (j - 1) * width, values, width=width, label=label, color=color)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(subsets)
    ax.set_ylabel("Delta métrica top-k (QML - clásico)")
    ax.set_title("QML no pierde en ranking top-k y puede mejorar ligeramente precision@20")
    ax.legend(frameon=False)
    paths = save_figure(fig, output_dir, "fig08_qml_topk_delta", args.overwrite)
    table_path = write_table(long, output_dir, "fig08_qml_topk_delta_table.csv", args.overwrite)
    print_outputs([*paths, table_path])


if __name__ == "__main__":
    main()
