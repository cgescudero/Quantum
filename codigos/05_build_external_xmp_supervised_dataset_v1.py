"""Construcción de dataset supervisado XMP/metal-poor para ML vs QML.

Este script toma los positivos externos recuperados directamente desde SDSS en
``04_query_sdss_for_external_rare_objects_v1.py`` y construye datasets
supervisados pequeños para clasificación/ranking de galaxias raras.

El producto principal es la versión B: controles SDSS pareados en redshift y masa
estelar. Esto reduce el riesgo de que un clasificador distinga positivos por
sesgos triviales de selección en ``z`` o ``log_stellar_mass``. Como control
metodológico secundario también se guarda una versión A con controles aleatorios.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_POSITIVE_CSV = SCRIPT_DIR / "results_external_sdss_query_v1" / "external_sdss_features_physical_v1.csv"
DEFAULT_SDSS_CSV = PROJECT_DIR / "sdss_galaxies_features_physical.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_external_xmp_supervised_dataset_v1"
SCRIPT_VERSION = "v1"

FEATURE_COLUMNS = [
    "log_nii_ha",
    "log_oiii_hb",
    "log_oii_hb",
    "log_sii_ha",
    "log_r23",
    "log_o32",
    "log_stellar_mass",
    "z",
]
ID_COLUMNS = ["specobjid", "ra", "dec"]
TARGET_COLUMN = "metallicity"
MATCH_COLUMNS = ["z", "log_stellar_mass"]


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Construye dataset supervisado XMP con controles pareados en z y masa."
    )
    parser.add_argument("--positive-csv", type=Path, default=DEFAULT_POSITIVE_CSV)
    parser.add_argument("--sdss-csv", type=Path, default=DEFAULT_SDSS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--controls-per-positive", type=int, default=5)
    parser.add_argument("--random-controls-per-positive", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-redshift", type=float, default=0.02)
    parser.add_argument("--max-redshift", type=float, default=0.10)
    parser.add_argument(
        "--max-match-distance",
        type=float,
        default=0.0,
        help="Distancia máxima estandarizada para controles pareados. 0 desactiva caliper.",
    )
    return parser.parse_args()


def validate_output_dir(output_dir: Path) -> Path:
    """Valida y crea directorio de salida dentro de Parte_3."""
    resolved = output_dir.expanduser().resolve()
    allowed_root = SCRIPT_DIR.resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        raise ValueError(f"Por seguridad, --output-dir debe estar dentro de Parte_3: {allowed_root}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def normalize_specobjid_series(series: pd.Series) -> pd.Series:
    """Normaliza specobjid como cadena entera estable."""
    def one(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        text = str(value).strip()
        if not text:
            return ""
        if text.isdigit():
            return text
        try:
            return str(int(float(text)))
        except ValueError:
            return text

    return series.map(one)


def load_sdss(sdss_csv: Path, min_z: float, max_z: float) -> pd.DataFrame:
    """Carga la muestra SDSS base y aplica validación básica."""
    if not sdss_csv.exists():
        raise FileNotFoundError(f"No existe SDSS CSV: {sdss_csv}")
    data = pd.read_csv(sdss_csv)
    required = set(ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas SDSS obligatorias: {missing}")
    data = data.loc[:, ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    data["specobjid"] = normalize_specobjid_series(data["specobjid"])
    for column in FEATURE_COLUMNS + [TARGET_COLUMN, "ra", "dec"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    finite = np.isfinite(data[FEATURE_COLUMNS + [TARGET_COLUMN, "ra", "dec"]]).all(axis=1)
    physical = data["z"].between(min_z, max_z) & data[TARGET_COLUMN].between(7.0, 10.0)
    return data.loc[finite & physical].drop_duplicates("specobjid").reset_index(drop=True)


def load_positives(positive_csv: Path, min_z: float, max_z: float) -> pd.DataFrame:
    """Carga positivos externos recuperados desde SDSS y deduplica por specobjid."""
    if not positive_csv.exists():
        raise FileNotFoundError(f"No existe CSV de positivos externos: {positive_csv}")
    data = pd.read_csv(positive_csv)
    required = set(ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN, "external_class", "external_catalog_id"])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas en positivos externos: {missing}")
    keep_extra = [
        "external_object_id",
        "external_class",
        "external_catalog_id",
        "external_object_name",
        "external_redshift",
        "external_oxygen_abundance_12logOH",
        "bptclass",
        "delta_redshift",
    ]
    columns = ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN] + [col for col in keep_extra if col in data.columns]
    data = data.loc[:, columns].copy()
    data["specobjid"] = normalize_specobjid_series(data["specobjid"])
    for column in FEATURE_COLUMNS + [TARGET_COLUMN, "ra", "dec"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    finite = np.isfinite(data[FEATURE_COLUMNS + [TARGET_COLUMN, "ra", "dec"]]).all(axis=1)
    physical = data["z"].between(min_z, max_z) & data[TARGET_COLUMN].between(7.0, 10.0)
    data = data.loc[finite & physical].copy()
    if "external_oxygen_abundance_12logOH" in data.columns:
        data["external_oxygen_abundance_12logOH"] = pd.to_numeric(
            data["external_oxygen_abundance_12logOH"], errors="coerce"
        )
        data = data.sort_values(["specobjid", "external_oxygen_abundance_12logOH"], na_position="last")
    return data.drop_duplicates("specobjid", keep="first").reset_index(drop=True)


def add_label_columns(data: pd.DataFrame, label: int, source: str) -> pd.DataFrame:
    """Añade etiqueta supervisada y diagnósticos derivados."""
    result = data.copy()
    result["external_xmp_label"] = int(label)
    result["sample_source"] = source
    result["o3n2"] = result["log_oiii_hb"] - result["log_nii_ha"]
    result["Z_N2_PP04"] = 8.90 + 0.57 * result["log_nii_ha"]
    result["Z_O3N2_PP04"] = 8.73 - 0.32 * result["o3n2"]
    result["delta_N2_O3N2"] = np.abs(result["Z_N2_PP04"] - result["Z_O3N2_PP04"])
    return result


def standardize_match_space(positives: pd.DataFrame, controls: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Estandariza z y masa usando positivos+pool de controles."""
    combined = pd.concat([positives[MATCH_COLUMNS], controls[MATCH_COLUMNS]], ignore_index=True)
    mean = combined.mean(axis=0).to_numpy(dtype=float)
    std = combined.std(axis=0).replace(0, 1.0).to_numpy(dtype=float)
    return (
        (positives[MATCH_COLUMNS].to_numpy(dtype=float) - mean) / std,
        (controls[MATCH_COLUMNS].to_numpy(dtype=float) - mean) / std,
    )


def select_matched_controls(
    positives: pd.DataFrame,
    control_pool: pd.DataFrame,
    controls_per_positive: int,
    max_distance: float,
) -> pd.DataFrame:
    """Selecciona controles únicos cercanos en z y masa estelar."""
    if controls_per_positive <= 0:
        raise ValueError("controls_per_positive debe ser positivo.")
    if positives.empty:
        raise ValueError("No hay positivos para parear controles.")
    if control_pool.empty:
        raise ValueError("El pool de controles está vacío.")

    pos_x, ctrl_x = standardize_match_space(positives, control_pool)
    available = np.ones(len(control_pool), dtype=bool)
    selected_rows = []
    for pos_idx, pos_vec in enumerate(pos_x):
        distances = np.sqrt(np.sum((ctrl_x - pos_vec) ** 2, axis=1))
        distances[~available] = np.inf
        if max_distance > 0:
            distances[distances > max_distance] = np.inf
        chosen = np.argsort(distances)[:controls_per_positive]
        chosen = [idx for idx in chosen if np.isfinite(distances[idx])]
        for rank, ctrl_idx in enumerate(chosen, start=1):
            row = control_pool.iloc[ctrl_idx].copy()
            row["matched_positive_specobjid"] = positives.iloc[pos_idx]["specobjid"]
            row["match_distance_z_mass"] = float(distances[ctrl_idx])
            row["match_rank_for_positive"] = rank
            selected_rows.append(row)
            available[ctrl_idx] = False
    if not selected_rows:
        raise ValueError("No se seleccionó ningún control pareado; revisar max-match-distance.")
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def select_random_controls(
    control_pool: pd.DataFrame,
    n_controls: int,
    seed: int,
) -> pd.DataFrame:
    """Selecciona controles aleatorios sin reemplazo."""
    if n_controls <= 0:
        raise ValueError("n_controls debe ser positivo.")
    n = min(n_controls, len(control_pool))
    return control_pool.sample(n=n, random_state=seed).reset_index(drop=True)


def build_dataset(positives: pd.DataFrame, controls: pd.DataFrame, source: str) -> pd.DataFrame:
    """Concatena positivos y controles con columnas homogéneas."""
    pos = add_label_columns(positives, 1, "external_xmp_positive")
    ctrl = add_label_columns(controls, 0, source)
    for col in pos.columns:
        if col not in ctrl.columns:
            ctrl[col] = np.nan
    for col in ctrl.columns:
        if col not in pos.columns:
            pos[col] = np.nan
    data = pd.concat([pos[ctrl.columns], ctrl], ignore_index=True)
    return data.sample(frac=1.0, random_state=123).reset_index(drop=True)


def summarize_dataset(name: str, data: pd.DataFrame) -> dict[str, Any]:
    """Resume conteos principales de un dataset."""
    return {
        "dataset": name,
        "n_rows": int(len(data)),
        "n_positive": int(data["external_xmp_label"].sum()),
        "n_negative": int((data["external_xmp_label"] == 0).sum()),
        "positive_fraction": float(data["external_xmp_label"].mean()) if len(data) else np.nan,
        "z_positive_median": float(data.loc[data["external_xmp_label"] == 1, "z"].median()),
        "z_negative_median": float(data.loc[data["external_xmp_label"] == 0, "z"].median()),
        "mass_positive_median": float(data.loc[data["external_xmp_label"] == 1, "log_stellar_mass"].median()),
        "mass_negative_median": float(data.loc[data["external_xmp_label"] == 0, "log_stellar_mass"].median()),
        "metallicity_positive_median": float(data.loc[data["external_xmp_label"] == 1, TARGET_COLUMN].median()),
        "metallicity_negative_median": float(data.loc[data["external_xmp_label"] == 0, TARGET_COLUMN].median()),
    }


def write_outputs(
    output_dir: Path,
    matched: pd.DataFrame,
    random_dataset: pd.DataFrame,
    positives: pd.DataFrame,
    control_pool: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """Guarda datasets y metadatos."""
    matched.to_csv(output_dir / "dataset_external_xmp_matched_controls_v1.csv", index=False)
    random_dataset.to_csv(output_dir / "dataset_external_xmp_random_controls_v1.csv", index=False)
    positives.to_csv(output_dir / "external_xmp_positive_unique_v1.csv", index=False)
    control_pool.to_csv(output_dir / "sdss_control_pool_excluding_external_xmp_v1.csv", index=False)
    summary.to_csv(output_dir / "dataset_external_xmp_summary_v1.csv", index=False)
    with (output_dir / "dataset_external_xmp_metadata_v1.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Construye datasets supervisados XMP externo vs controles SDSS."""
    args = parse_args()
    output_dir = validate_output_dir(args.output_dir)
    sdss = load_sdss(args.sdss_csv, args.min_redshift, args.max_redshift)
    positives = load_positives(args.positive_csv, args.min_redshift, args.max_redshift)
    positive_ids = set(positives["specobjid"])
    control_pool = sdss.loc[~sdss["specobjid"].isin(positive_ids)].reset_index(drop=True)

    matched_controls = select_matched_controls(
        positives=positives,
        control_pool=control_pool,
        controls_per_positive=args.controls_per_positive,
        max_distance=args.max_match_distance,
    )
    random_controls = select_random_controls(
        control_pool=control_pool,
        n_controls=args.random_controls_per_positive * len(positives),
        seed=args.seed,
    )
    matched_dataset = build_dataset(positives, matched_controls, "sdss_z_mass_matched_control")
    random_dataset = build_dataset(positives, random_controls, "sdss_random_control")
    summary = pd.DataFrame(
        [
            summarize_dataset("matched_controls", matched_dataset),
            summarize_dataset("random_controls", random_dataset),
        ]
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "positive_csv": str(args.positive_csv.resolve()),
        "sdss_csv": str(args.sdss_csv.resolve()),
        "output_dir": str(output_dir),
        "controls_per_positive": args.controls_per_positive,
        "random_controls_per_positive": args.random_controls_per_positive,
        "seed": args.seed,
        "min_redshift": args.min_redshift,
        "max_redshift": args.max_redshift,
        "max_match_distance": args.max_match_distance,
        "n_sdss_after_filters": int(len(sdss)),
        "n_positive_unique": int(len(positives)),
        "n_control_pool": int(len(control_pool)),
        "n_matched_controls": int(len(matched_controls)),
        "n_random_controls": int(len(random_controls)),
        "notes": "Matched controls are paired only in z and stellar mass, not metallicity or line ratios.",
    }
    write_outputs(output_dir, matched_dataset, random_dataset, positives, control_pool, summary, metadata)

    print(f"Salida escrita en: {output_dir}")
    print(f"Positivos externos únicos: {len(positives)}")
    print(f"Pool de controles SDSS: {len(control_pool)}")
    print(f"Dataset B pareado: {len(matched_dataset)} filas")
    print(f"Dataset A aleatorio: {len(random_dataset)} filas")


if __name__ == "__main__":
    main()
