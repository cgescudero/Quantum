"""Figura 01: flujo general y embudo SDSS.

Comando:
    # python3 plot_01_pipeline_and_sdss_funnel_v1.py --overwrite
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from plotting_external_xmp_ml_qml_utils_v1 import (
    COLORS,
    SCRIPT_DIR,
    parse_common_args,
    print_outputs,
    read_csv_checked,
    read_json_checked,
    save_figure,
    setup_matplotlib,
    validate_output_dir,
    write_table,
)

FALLBACK_COUNTS = {
    "Externos normalizados": 1292,
    "Externos usados para SDSS": 1214,
    "Matches SDSS por región": 960,
    "SpecObjID únicos": 501,
    "Filas con espectros/líneas": 386,
    "Features físicas válidas": 274,
    "Positivos externos únicos finales": 124,
}


def build_counts() -> pd.DataFrame:
    """Construye la tabla del embudo desde CSV/JSON con fallbacks documentados."""
    norm_csv = SCRIPT_DIR / "results_external_catalog_normalized_v1/external_rare_galaxy_catalogs_normalized_v1.csv"
    query_json = SCRIPT_DIR / "results_external_sdss_query_v1/external_sdss_query_metadata_v1.json"
    dataset_json = SCRIPT_DIR / "results_external_xmp_supervised_dataset_v1/dataset_external_xmp_metadata_v1.json"
    norm = read_csv_checked(norm_csv)
    query = read_json_checked(query_json)
    dataset = read_json_checked(dataset_json)
    rows = [
        ("Externos normalizados", len(norm), "csv_rows"),
        ("Externos usados para SDSS", query.get("n_external_queried", FALLBACK_COUNTS["Externos usados para SDSS"]), "json_or_fallback"),
        ("Matches SDSS por región", query.get("n_region_matches", FALLBACK_COUNTS["Matches SDSS por región"]), "json_or_fallback"),
        ("SpecObjID únicos", query.get("n_unique_specobjids_region", FALLBACK_COUNTS["SpecObjID únicos"]), "json_or_fallback"),
        ("Filas con espectros/líneas", query.get("n_spectroscopy_rows", FALLBACK_COUNTS["Filas con espectros/líneas"]), "json_or_fallback"),
        ("Features físicas válidas", query.get("n_feature_rows", FALLBACK_COUNTS["Features físicas válidas"]), "json_or_fallback"),
        ("Positivos externos únicos finales", dataset.get("n_positive_unique", FALLBACK_COUNTS["Positivos externos únicos finales"]), "json_or_fallback"),
    ]
    return pd.DataFrame(rows, columns=["stage", "count", "source"])


def main() -> None:
    """Genera figura compuesta de flujo conceptual y embudo cuantitativo."""
    args = parse_common_args("Figura 01: pipeline y embudo SDSS")
    output_dir = validate_output_dir(args.output_dir)
    setup_matplotlib()
    counts = build_counts()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.05, 1.25]})

    ax = axes[0]
    ax.axis("off")
    steps = [
        "Catálogos externos", "normalización", "crossmatch / consulta SDSS",
        "positivos XMP externos", "controles SDSS pareados", "benchmark ML vs QML",
        "subsets físicos",
    ]
    y_positions = list(reversed([i / (len(steps) - 1) for i in range(len(steps))]))
    for i, (step, y) in enumerate(zip(steps, y_positions)):
        color = COLORS["quantum"] if "QML" in step else COLORS["matched"]
        ax.text(0.5, y, step, ha="center", va="center", color="white",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": color, "edgecolor": "none"})
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y_positions[i + 1] + 0.04), xytext=(0.5, y - 0.04),
                        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "0.25"})
    ax.set_title("A. Flujo conceptual")

    ax = axes[1]
    plot_df = counts.iloc[::-1].reset_index(drop=True)
    bars = ax.barh(plot_df["stage"], plot_df["count"], color=COLORS["classical"])
    for bar, value in zip(bars, plot_df["count"]):
        ax.text(value + max(counts["count"]) * 0.015, bar.get_y() + bar.get_height() / 2, f"{int(value)}", va="center")
    ax.set_xlabel("Número de objetos/filas")
    ax.set_title("B. Embudo de recuperación en SDSS")
    ax.set_xlim(0, max(counts["count"]) * 1.18)
    fig.suptitle("Catálogos externos → dataset supervisado para benchmark ML/QML", y=1.02)
    paths = save_figure(fig, output_dir, "fig01_pipeline_and_sdss_funnel", args.overwrite)
    table_path = write_table(counts, output_dir, "fig01_funnel_counts.csv", args.overwrite)
    print_outputs([*paths, table_path])


if __name__ == "__main__":
    main()
