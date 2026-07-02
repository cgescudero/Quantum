"""Figura 04: benchmark principal ML vs QML en dataset matched.

Comando:
    # python3 plot_04_main_matched_benchmark_v1.py --overwrite
"""

from __future__ import annotations

import matplotlib.pyplot as plt

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


def collapse_quantum_seeds_by_best_pr_auc(table):
    """Agrupa variantes QK-SVC por familia y conserva la de mayor PR-AUC.

    Los modelos clásicos se mantienen sin agrupar. Para modelos QML con varias
    semillas/configuraciones que representan la misma familia visual, se conserva
    sólo la fila con mayor PR-AUC, porque la figura 04 busca comparar familias de
    modelos y no repetir barras superpuestas de QK-SVC.
    """
    out = table.copy()
    out["model_base"] = out["model"].map(model_base_family)
    out["aggregation"] = "single_model"
    rows = []
    for _, group in out.groupby("model_base", sort=False):
        model_base = str(group["model_base"].iloc[0])
        if model_base.startswith("qksvc"):
            best = group.sort_values("pr_auc", ascending=False).iloc[0].copy()
            best["aggregation"] = "max_pr_auc_within_qml_family"
            best["n_models_collapsed"] = len(group)
            rows.append(best)
        else:
            for _, row in group.iterrows():
                row = row.copy()
                row["n_models_collapsed"] = 1
                rows.append(row)
    collapsed = type(table)(rows).reset_index(drop=True)
    collapsed["display_model"] = collapsed["model_base"].map(model_display_name)
    return collapsed


def main() -> None:
    """Grafica PR-AUC medio ordenado para el dataset pareado."""
    args = parse_common_args("Figura 04: benchmark matched ML/QML")
    output_dir = validate_output_dir(args.output_dir)
    setup_matplotlib()
    path = first_existing([
        SCRIPT_DIR / "results_external_xmp_qml_classical_matched_v1/summary_metrics.csv",
        SCRIPT_DIR / "results_external_xmp_qml_classical_matched_v1/summary_by_model.csv",
    ])
    df = read_csv_checked(path, ["model"])
    pr_col = metric_column(df, "pr_auc_mean", "pr_auc")
    roc_col = metric_column(df, "roc_auc_mean", "roc_auc")
    df = require_numeric_columns(df, [pr_col, roc_col], "matched benchmark")
    table = df[["model", pr_col, roc_col] + (["model_family"] if "model_family" in df.columns else [])].copy()
    table = table.rename(columns={pr_col: "pr_auc", roc_col: "roc_auc"})
    table = collapse_quantum_seeds_by_best_pr_auc(table)
    table = table.sort_values("pr_auc", ascending=True)
    colors = [COLORS["quantum"] if str(m).startswith("qksvc") else COLORS["classical"] for m in table["model"]]
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    bars = ax.barh(table["display_model"], table["pr_auc"], color=colors)
    for bar, value in zip(bars, table["pr_auc"]):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    ax.set_xlabel("PR-AUC medio")
    ax.set_title("Dataset matched/all_10: QML no supera al mejor ML clásico")
    ax.set_xlim(0, max(table["pr_auc"].max() * 1.18, 0.6))
    paths = save_figure(fig, output_dir, "fig04_main_matched_benchmark_pr_auc", args.overwrite)
    table_path = write_table(table.sort_values("pr_auc", ascending=False), output_dir, "fig04_main_matched_benchmark_table.csv", args.overwrite)
    print_outputs([*paths, table_path])


if __name__ == "__main__":
    main()
