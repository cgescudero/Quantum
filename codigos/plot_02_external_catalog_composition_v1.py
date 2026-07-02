"""Figura 02: composición de catálogos externos normalizados.

Comando:
    # python3 plot_02_external_catalog_composition_v1.py --overwrite
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from plotting_external_xmp_ml_qml_utils_v1 import (
    COLORS,
    SCRIPT_DIR,
    find_column,
    parse_common_args,
    print_outputs,
    read_csv_checked,
    save_figure,
    setup_matplotlib,
    validate_output_dir,
    write_table,
)


def main() -> None:
    """Cuenta clases externas y genera bar plot."""
    args = parse_common_args("Figura 02: composición de catálogos externos")
    output_dir = validate_output_dir(args.output_dir)
    setup_matplotlib()
    csv_path = SCRIPT_DIR / "results_external_catalog_normalized_v1/external_rare_galaxy_catalogs_normalized_v1.csv"
    df = read_csv_checked(csv_path)
    class_col = find_column(df, ["external_class", "tentative_class", "class"], "clase externa")
    counts = df[class_col].fillna("missing").value_counts().rename_axis("external_class").reset_index(name="count")
    order = ["green_pea", "xmp_or_metal_poor", "lyc_or_lae_candidate"]
    counts["order"] = counts["external_class"].map({name: i for i, name in enumerate(order)}).fillna(99)
    counts = counts.sort_values(["order", "external_class"]).drop(columns="order")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(counts["external_class"], counts["count"], color=[COLORS["external"], COLORS["classical"], COLORS["quantum"]][: len(counts)])
    for bar, value in zip(bars, counts["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(counts["count"]) * 0.02, str(int(value)), ha="center")
    ax.set_ylabel("Número de objetos normalizados")
    ax.set_xlabel("Clase tentativa externa")
    ax.set_title("Composición de catálogos externos")
    ax.tick_params(axis="x", rotation=20)
    paths = save_figure(fig, output_dir, "fig02_external_catalog_composition", args.overwrite)
    table_path = write_table(counts, output_dir, "fig02_external_catalog_composition_counts.csv", args.overwrite)
    print_outputs([*paths, table_path])


if __name__ == "__main__":
    main()
