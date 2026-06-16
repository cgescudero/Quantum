"""Preprocesamiento físico de features SDSS para ML/QML.

Lee el CSV espectroscópico descargado desde SDSS, valida columnas y rangos
físicos básicos, construye ratios nebulares en escala logarítmica y guarda una
tabla de features lista para pipelines de regresión de metalicidad.

Este script no escala variables, no aplica PCA y no divide train/test. Esas
operaciones deben hacerse dentro de cada validación cruzada para evitar leakage.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "sdss_galaxies_spectroscopy_v2.csv"
OUTPUT_CSV = SCRIPT_DIR / "sdss_galaxies_features_physical.csv"
OUTPUT_METADATA = SCRIPT_DIR / "sdss_galaxies_features_physical_metadata.json"

SOURCE_DATA_RELEASE = "SDSS DR18"

RAW_REQUIRED_COLUMNS = [
    "specobjid",
    "ra",
    "dec",
    "z",
    "h_alpha_flux",
    "h_beta_flux",
    "oiii_5007_flux",
    "nii_6584_flux",
    "oii_3726_flux",
    "oii_3729_flux",
    "sii_6717_flux",
    "sii_6731_flux",
    "log_stellar_mass",
    "metallicity",
]

POSITIVE_FLUX_COLUMNS = [
    "h_alpha_flux",
    "h_beta_flux",
    "oiii_5007_flux",
    "nii_6584_flux",
    "oii_3726_flux",
    "oii_3729_flux",
    "sii_6717_flux",
    "sii_6731_flux",
]

NUMERIC_COLUMNS = [
    "ra",
    "dec",
    "z",
    "log_stellar_mass",
    "metallicity",
] + POSITIVE_FLUX_COLUMNS

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

OUTPUT_COLUMNS = [
    "specobjid",
    "ra",
    "dec",
] + FEATURE_COLUMNS + ["metallicity"]


def load_raw_catalog(input_csv: Path = INPUT_CSV) -> pd.DataFrame:
    """Carga el catálogo espectroscópico SDSS.

    Parameters
    ----------
    input_csv:
        Ruta al CSV generado por ``datos_sdss_v2.py``.

    Returns
    -------
    pandas.DataFrame
        Catálogo crudo leído desde disco.

    Raises
    ------
    FileNotFoundError
        Si el archivo de entrada no existe.
    pandas.errors.EmptyDataError
        Si el archivo existe pero está vacío o no es un CSV válido.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_csv}")

    return pd.read_csv(input_csv)


def validate_raw_catalog(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Valida columnas, finitud, positividad y rangos físicos básicos.

    Parameters
    ----------
    df:
        Catálogo SDSS crudo.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, int]]
        DataFrame limpio y conteos de filas descartadas por criterio.

    Raises
    ------
    ValueError
        Si faltan columnas obligatorias o la muestra final queda vacía.
    """
    missing = sorted(set(RAW_REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    clean = df.loc[:, RAW_REQUIRED_COLUMNS].copy()
    initial_rows = len(clean)

    for column in NUMERIC_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    finite_mask = np.isfinite(clean[NUMERIC_COLUMNS]).all(axis=1)
    positive_flux_mask = (clean[POSITIVE_FLUX_COLUMNS] > 0).all(axis=1)
    range_mask = (
        clean["z"].between(0.02, 0.10)
        & clean["metallicity"].between(7.0, 10.0)
        & clean["log_stellar_mass"].between(6.0, 13.0)
    )

    valid_mask = finite_mask & positive_flux_mask & range_mask

    rejection_counts = {
        "initial_rows": initial_rows,
        "non_finite_rows": int((~finite_mask).sum()),
        "non_positive_flux_rows": int((finite_mask & ~positive_flux_mask).sum()),
        "out_of_physical_range_rows": int((finite_mask & positive_flux_mask & ~range_mask).sum()),
        "final_valid_rows": int(valid_mask.sum()),
        "total_discarded_rows": int(initial_rows - valid_mask.sum()),
    }

    clean = clean.loc[valid_mask].reset_index(drop=True)
    if clean.empty:
        raise ValueError("La validación dejó una muestra vacía.")

    return clean, rejection_counts


def build_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye ratios nebulares logarítmicos para regresión de metalicidad.

    Parameters
    ----------
    df:
        Catálogo validado con flujos positivos.

    Returns
    -------
    pandas.DataFrame
        Tabla con identificadores, coordenadas, features físicas y target.

    Notes
    -----
    Las features se definen como:

    - ``log_nii_ha = log10([NII]6584 / Halpha)``
    - ``log_oiii_hb = log10([OIII]5007 / Hbeta)``
    - ``log_oii_hb = log10(([OII]3726 + [OII]3729) / Hbeta)``
    - ``log_sii_ha = log10(([SII]6717 + [SII]6731) / Halpha)``
    - ``log_r23 = log10(([OII]3726 + [OII]3729 + [OIII]5007) / Hbeta)``
    - ``log_o32 = log10([OIII]5007 / ([OII]3726 + [OII]3729))``
    """
    features = df.loc[:, ["specobjid", "ra", "dec", "log_stellar_mass", "z", "metallicity"]].copy()

    oii_total = df["oii_3726_flux"] + df["oii_3729_flux"]
    sii_total = df["sii_6717_flux"] + df["sii_6731_flux"]

    features["log_nii_ha"] = np.log10(df["nii_6584_flux"] / df["h_alpha_flux"])
    features["log_oiii_hb"] = np.log10(df["oiii_5007_flux"] / df["h_beta_flux"])
    features["log_oii_hb"] = np.log10(oii_total / df["h_beta_flux"])
    features["log_sii_ha"] = np.log10(sii_total / df["h_alpha_flux"])
    features["log_r23"] = np.log10((oii_total + df["oiii_5007_flux"]) / df["h_beta_flux"])
    features["log_o32"] = np.log10(df["oiii_5007_flux"] / oii_total)

    features = features.loc[:, OUTPUT_COLUMNS]
    finite_features_mask = np.isfinite(features[FEATURE_COLUMNS + ["metallicity"]]).all(axis=1)
    discarded = int((~finite_features_mask).sum())
    if discarded > 0:
        print(f"Advertencia: {discarded} filas descartadas tras construir ratios logarítmicos.")

    features = features.loc[finite_features_mask].reset_index(drop=True)
    if features.empty:
        raise ValueError("La construcción de features dejó una muestra vacía.")

    return features


def write_metadata(
    metadata_path: Path,
    input_csv: Path,
    output_csv: Path,
    rejection_counts: dict[str, int],
    final_rows: int,
) -> None:
    """Guarda metadatos de reproducibilidad en formato JSON.

    Parameters
    ----------
    metadata_path:
        Ruta del archivo JSON de salida.
    input_csv:
        Catálogo de entrada.
    output_csv:
        Tabla de features generada.
    rejection_counts:
        Conteos de validación y descarte.
    final_rows:
        Número final de objetos en la tabla de features.
    """
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_data_release": SOURCE_DATA_RELEASE,
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "final_rows": final_rows,
        "target_column": "metallicity",
        "feature_columns": FEATURE_COLUMNS,
        "output_columns": OUTPUT_COLUMNS,
        "validation": rejection_counts,
        "physical_assumptions": {
            "metallicity_scale": "12 + log(O/H), column oh_p50 from galSpecExtra",
            "line_ratios": "Base-10 logarithms of positive emission-line flux ratios",
            "no_scaling_or_pca": True,
            "intended_scaling_stage": "inside each ML/QML cross-validation pipeline",
        },
    }

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Ejecuta el preprocesamiento físico completo."""
    print(f"Leyendo catálogo SDSS: {INPUT_CSV}")
    raw = load_raw_catalog(INPUT_CSV)
    validated, rejection_counts = validate_raw_catalog(raw)
    features = build_physical_features(validated)

    features.to_csv(OUTPUT_CSV, index=False)
    write_metadata(
        metadata_path=OUTPUT_METADATA,
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        rejection_counts=rejection_counts,
        final_rows=len(features),
    )

    print("Preprocesamiento completado.")
    print(f"Filas iniciales: {rejection_counts['initial_rows']}")
    print(f"Filas finales: {len(features)}")
    print(f"Filas descartadas: {rejection_counts['initial_rows'] - len(features)}")
    print(f"CSV guardado en: {OUTPUT_CSV}")
    print(f"Metadatos guardados en: {OUTPUT_METADATA}")
    print("Features generadas:")
    print(", ".join(FEATURE_COLUMNS))


if __name__ == "__main__":
    main()
