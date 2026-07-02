"""Figura 05: matched vs random como advertencia de sesgo.

Comando:
    # python3 plot_05_matched_vs_random_benchmark_bias_v1.py --overwrite
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting_external_xmp_ml_qml_utils_v1 import (
    COLORS,
    SCRIPT_DIR,
    first_existing,
    metric_column,
    model_base_family,
    model_display_name,
    parse_common_args,
    print_outputs,
    read_csv_checked,
    require_numeric_columns,
    save_figure,
    setup_matplotlib,
    validate_output_dir,
    write_table,
)


def load_summary(dataset: str) -> pd.DataFrame:
    """Carga resumen de métricas para matched o random y agrupa semillas QML por promedio."""
    path = first_existing([
        SCRIPT_DIR / f"results_external_xmp_qml_classical_{dataset}_v1/summary_metrics.csv",
        SCRIPT_DIR / f"results_external_xmp_qml_classical_{dataset}_v1/summary_by_model.csv",
    ])
    df = read_csv_checked(path, ["model"])
    pr_col = metric_column(df, "pr_auc_mean", "pr_auc")
    df = require_numeric_columns(df, [pr_col], f"{dataset} benchmark")
    df = df.assign(dataset=dataset, model_base=df["model"].map(model_base_family), aggregation="mean_by_model_base")
    grouped = df.groupby(["dataset", "model_base", "aggregation"], as_index=False).agg(pr_auc=(pr_col, "mean"), n_models=("model", "size"))
    return grouped


def main() -> None:
    """Genera bar plot agrupado de PR-AUC matched vs random."""
    args = parse_common_args("Figura 05: sesgo random vs matched")
    output_dir = validate_output_dir(args.output_dir)
    setup_matplotlib()
    table = pd.concat([load_summary("matched"), load_summary("random")], ignore_index=True)
    wanted = ["qksvc_reuploading", "svc_rbf", "svc_poly", "logistic_regression", "random_forest", "hist_gradient_boosting"]
    table = table[table["model_base"].isin(wanted)].copy()
    table["display_model"] = table["model_base"].map(model_display_name)
    table["order"] = table["model_base"].map({name: i for i, name in enumerate(wanted)})
    table = table.sort_values("order")

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(wanted))
    width = 0.38
    for offset, dataset, color in [(-width / 2, "matched", COLORS["matched"]), (width / 2, "random", COLORS["random"] )]:
        values = []
        for model in wanted:
            hit = table[(table["dataset"] == dataset) & (table["model_base"] == model)]
            values.append(float(hit["pr_auc"].iloc[0]) if not hit.empty else np.nan)
        ax.bar(x + offset, values, width=width, label=dataset, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([model_display_name(m) for m in wanted], rotation=25, ha="right")
    ax.set_ylabel("PR-AUC medio")
    ax.set_title("Random infla métricas; matched es el benchmark científicamente relevante")
    ax.legend(frameon=False)
    paths = save_figure(fig, output_dir, "fig05_matched_vs_random_bias", args.overwrite)
    table_path = write_table(table.drop(columns="order"), output_dir, "fig05_matched_vs_random_bias_table.csv", args.overwrite)
    print_outputs([*paths, table_path])


if __name__ == "__main__":
    main()
