"""Crossmatch de catálogos externos raros con la muestra SDSS procesada.

Este script toma la tabla homogénea generada por
``02_normalize_external_rare_galaxy_catalogs_v1.py`` y la cruza por coordenadas
con ``sdss_galaxies_features_physical.csv``. El objetivo es identificar cuáles
galaxias raras externas aparecen dentro de la muestra SDSS que ya usamos para
ML/QML, y construir etiquetas externas para futuros experimentos supervisados o
de ranking.

No entrena modelos. La salida principal es una tabla de matches auditables y una
versión de la muestra SDSS con columnas nuevas de etiqueta externa.
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
DEFAULT_EXTERNAL_CSV = (
    SCRIPT_DIR
    / "results_external_catalog_normalized_v1"
    / "external_rare_galaxy_catalogs_normalized_v1.csv"
)
DEFAULT_SDSS_CSV = PROJECT_DIR / "sdss_galaxies_features_physical.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_external_sdss_crossmatch_v1"
SCRIPT_VERSION = "v1"

SDSS_REQUIRED_COLUMNS = [
    "specobjid",
    "ra",
    "dec",
    "log_nii_ha",
    "log_oiii_hb",
    "log_oii_hb",
    "log_sii_ha",
    "log_r23",
    "log_o32",
    "log_stellar_mass",
    "z",
    "metallicity",
]

EXTERNAL_REQUIRED_COLUMNS = [
    "external_object_id",
    "external_class",
    "catalog_id",
    "catalog_description",
    "keyword",
    "object_name",
    "ra_deg",
    "dec_deg",
    "redshift",
    "oxygen_abundance_12logOH",
    "usable_for_crossmatch",
]


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Cruza catálogos externos raros con la muestra SDSS procesada."
    )
    parser.add_argument(
        "--external-csv",
        type=Path,
        default=DEFAULT_EXTERNAL_CSV,
        help="CSV normalizado generado por el Script 02.",
    )
    parser.add_argument(
        "--sdss-csv",
        type=Path,
        default=DEFAULT_SDSS_CSV,
        help="CSV SDSS procesado con features físicas.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio de salida dentro de Parte_3.",
    )
    parser.add_argument(
        "--match-radius-arcsec",
        type=float,
        default=2.0,
        help="Radio máximo de crossmatch por coordenadas, en arcsec.",
    )
    parser.add_argument(
        "--redshift-tolerance",
        type=float,
        default=0.005,
        help="Tolerancia para marcar consistencia de redshift si ambos z existen.",
    )
    parser.add_argument(
        "--require-redshift-consistency",
        action="store_true",
        help="Sólo etiqueta como positivos los matches con redshift consistente o sin z externo.",
    )
    parser.add_argument(
        "--exclude-classes",
        type=str,
        default="lyc_or_lae_candidate",
        help="Clases externas a excluir antes del crossmatch, separadas por coma. Por defecto excluye alto-z/lensing.",
    )
    return parser.parse_args()


def parse_class_list(text: str) -> set[str]:
    """Parsea una lista CSV de clases externas."""
    return {item.strip() for item in text.split(",") if item.strip()}


def validate_part3_output_dir(output_dir: Path) -> Path:
    """Valida que la salida quede dentro de Parte_3."""
    resolved = output_dir.expanduser().resolve()
    allowed_root = SCRIPT_DIR.resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        raise ValueError(f"Por seguridad, --output-dir debe estar dentro de Parte_3: {allowed_root}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_sdss(sdss_csv: Path) -> pd.DataFrame:
    """Carga y valida la muestra SDSS procesada."""
    if not sdss_csv.exists():
        raise FileNotFoundError(f"No existe archivo SDSS: {sdss_csv}")
    data = pd.read_csv(sdss_csv)
    missing = sorted(set(SDSS_REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas SDSS obligatorias: {missing}")
    for column in ["ra", "dec", "z"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    finite = np.isfinite(data[["ra", "dec"]]).all(axis=1)
    if not finite.all():
        discarded = int((~finite).sum())
        print(f"Advertencia: se descartan {discarded} filas SDSS sin coordenadas finitas.")
    return data.loc[finite].reset_index(drop=True)


def load_external(external_csv: Path, excluded_classes: set[str]) -> pd.DataFrame:
    """Carga y filtra la tabla externa normalizada."""
    if not external_csv.exists():
        raise FileNotFoundError(f"No existe archivo externo normalizado: {external_csv}")
    data = pd.read_csv(external_csv)
    missing = sorted(set(EXTERNAL_REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas externas obligatorias: {missing}")
    for column in ["ra_deg", "dec_deg", "redshift", "oxygen_abundance_12logOH"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    usable = data["usable_for_crossmatch"].astype(bool)
    finite = np.isfinite(data[["ra_deg", "dec_deg"]]).all(axis=1)
    class_mask = ~data["external_class"].astype(str).isin(excluded_classes)
    return data.loc[usable & finite & class_mask].reset_index(drop=True)


def build_coordinates(ra_deg: pd.Series, dec_deg: pd.Series) -> Any:
    """Construye SkyCoord en grados."""
    try:
        from astropy.coordinates import SkyCoord  # type: ignore
        import astropy.units as u  # type: ignore
    except Exception as exc:  # pragma: no cover - depende del entorno local
        raise ImportError("Este script requiere astropy para el crossmatch por coordenadas.") from exc
    return SkyCoord(ra=ra_deg.to_numpy(dtype=float) * u.deg, dec=dec_deg.to_numpy(dtype=float) * u.deg)


def crossmatch_external_to_sdss(
    external: pd.DataFrame,
    sdss: pd.DataFrame,
    radius_arcsec: float,
    redshift_tolerance: float,
) -> pd.DataFrame:
    """Cruza externos contra SDSS y devuelve todos los matches dentro del radio."""
    if radius_arcsec <= 0:
        raise ValueError("match_radius_arcsec debe ser positivo.")
    if redshift_tolerance < 0:
        raise ValueError("redshift_tolerance no puede ser negativa.")

    import astropy.units as u  # type: ignore

    external_coord = build_coordinates(external["ra_deg"], external["dec_deg"])
    sdss_coord = build_coordinates(sdss["ra"], sdss["dec"])
    idx_ext, idx_sdss, sep2d, _ = sdss_coord.search_around_sky(external_coord, radius_arcsec * u.arcsec)

    rows: list[dict[str, Any]] = []
    for ext_i, sdss_i, sep in zip(idx_ext, idx_sdss, sep2d.arcsec):
        ext = external.iloc[int(ext_i)]
        obj = sdss.iloc[int(sdss_i)]
        external_z = ext.get("redshift", np.nan)
        sdss_z = obj.get("z", np.nan)
        has_both_z = bool(np.isfinite(external_z) and np.isfinite(sdss_z))
        dz = abs(float(sdss_z) - float(external_z)) if has_both_z else np.nan
        redshift_consistent = bool(dz <= redshift_tolerance) if has_both_z else False
        rows.append(
            {
                "external_object_id": ext["external_object_id"],
                "external_class": ext["external_class"],
                "external_catalog_id": ext["catalog_id"],
                "external_catalog_description": ext.get("catalog_description", ""),
                "external_keyword": ext.get("keyword", ""),
                "external_object_name": ext.get("object_name", ""),
                "external_ra_deg": ext["ra_deg"],
                "external_dec_deg": ext["dec_deg"],
                "external_redshift": external_z,
                "external_oxygen_abundance_12logOH": ext.get("oxygen_abundance_12logOH", np.nan),
                "sdss_row_index": int(sdss_i),
                "specobjid": obj["specobjid"],
                "sdss_ra": obj["ra"],
                "sdss_dec": obj["dec"],
                "sdss_z": sdss_z,
                "sdss_metallicity": obj.get("metallicity", np.nan),
                "match_sep_arcsec": float(sep),
                "has_both_redshifts": has_both_z,
                "delta_redshift": dz,
                "redshift_consistent": redshift_consistent,
            }
        )
    matches = pd.DataFrame(rows)
    if matches.empty:
        return matches
    return matches.sort_values(["match_sep_arcsec", "external_class", "external_catalog_id"]).reset_index(drop=True)


def choose_best_matches(matches: pd.DataFrame, require_redshift_consistency: bool) -> pd.DataFrame:
    """Elige un único mejor match externo por objeto SDSS."""
    if matches.empty:
        return matches.copy()
    candidates = matches.copy()
    if require_redshift_consistency:
        no_external_z = ~np.isfinite(candidates["external_redshift"].to_numpy(dtype=float))
        candidates = candidates.loc[candidates["redshift_consistent"] | no_external_z].copy()
    if candidates.empty:
        return candidates
    candidates["redshift_priority"] = np.where(candidates["redshift_consistent"], 0, 1)
    candidates = candidates.sort_values(["sdss_row_index", "redshift_priority", "match_sep_arcsec"])
    return candidates.drop_duplicates("sdss_row_index", keep="first").drop(columns=["redshift_priority"]).reset_index(drop=True)


def build_labeled_sdss(sdss: pd.DataFrame, best_matches: pd.DataFrame) -> pd.DataFrame:
    """Añade etiquetas externas a la muestra SDSS."""
    labeled = sdss.copy()
    labeled["external_rare_label"] = 0
    labeled["external_class"] = ""
    labeled["external_catalog_id"] = ""
    labeled["external_object_id"] = ""
    labeled["external_object_name"] = ""
    labeled["external_match_sep_arcsec"] = np.nan
    labeled["external_redshift"] = np.nan
    labeled["external_redshift_consistent"] = False
    labeled["external_oxygen_abundance_12logOH"] = np.nan

    for _, match in best_matches.iterrows():
        idx = int(match["sdss_row_index"])
        labeled.loc[idx, "external_rare_label"] = 1
        labeled.loc[idx, "external_class"] = match["external_class"]
        labeled.loc[idx, "external_catalog_id"] = match["external_catalog_id"]
        labeled.loc[idx, "external_object_id"] = match["external_object_id"]
        labeled.loc[idx, "external_object_name"] = match["external_object_name"]
        labeled.loc[idx, "external_match_sep_arcsec"] = match["match_sep_arcsec"]
        labeled.loc[idx, "external_redshift"] = match["external_redshift"]
        labeled.loc[idx, "external_redshift_consistent"] = bool(match["redshift_consistent"])
        labeled.loc[idx, "external_oxygen_abundance_12logOH"] = match["external_oxygen_abundance_12logOH"]
    return labeled


def summarize_matches(
    external: pd.DataFrame,
    sdss: pd.DataFrame,
    matches: pd.DataFrame,
    best_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Construye resumen compacto por clase externa."""
    rows = []
    external_counts = external.groupby("external_class").size().to_dict() if not external.empty else {}
    for external_class, n_external in sorted(external_counts.items()):
        class_matches = matches.loc[matches["external_class"] == external_class] if not matches.empty else matches
        class_best = best_matches.loc[best_matches["external_class"] == external_class] if not best_matches.empty else best_matches
        rows.append(
            {
                "external_class": external_class,
                "n_external_input": int(n_external),
                "n_raw_coordinate_matches": int(len(class_matches)),
                "n_unique_sdss_labeled": int(class_best["specobjid"].nunique()) if not class_best.empty else 0,
                "n_redshift_consistent_matches": int(class_matches["redshift_consistent"].sum()) if not class_matches.empty else 0,
                "median_sep_arcsec": float(class_matches["match_sep_arcsec"].median()) if not class_matches.empty else np.nan,
            }
        )
    rows.append(
        {
            "external_class": "TOTAL",
            "n_external_input": int(len(external)),
            "n_raw_coordinate_matches": int(len(matches)),
            "n_unique_sdss_labeled": int(best_matches["specobjid"].nunique()) if not best_matches.empty else 0,
            "n_redshift_consistent_matches": int(matches["redshift_consistent"].sum()) if not matches.empty else 0,
            "median_sep_arcsec": float(matches["match_sep_arcsec"].median()) if not matches.empty else np.nan,
        }
    )
    return pd.DataFrame(rows)


def write_outputs(
    output_dir: Path,
    matches: pd.DataFrame,
    best_matches: pd.DataFrame,
    labeled_sdss: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """Escribe productos de crossmatch."""
    matches.to_csv(output_dir / "external_sdss_crossmatch_all_matches_v1.csv", index=False)
    best_matches.to_csv(output_dir / "external_sdss_crossmatch_best_matches_v1.csv", index=False)
    labeled_sdss.to_csv(output_dir / "sdss_with_external_rare_labels_v1.csv", index=False)
    summary.to_csv(output_dir / "external_sdss_crossmatch_summary_v1.csv", index=False)
    with (output_dir / "crossmatch_metadata_v1.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Ejecuta el crossmatch externo-SDSS."""
    args = parse_args()
    output_dir = validate_part3_output_dir(args.output_dir)
    excluded_classes = parse_class_list(args.exclude_classes)
    sdss = load_sdss(args.sdss_csv)
    external = load_external(args.external_csv, excluded_classes)
    matches = crossmatch_external_to_sdss(
        external=external,
        sdss=sdss,
        radius_arcsec=args.match_radius_arcsec,
        redshift_tolerance=args.redshift_tolerance,
    )
    best_matches = choose_best_matches(matches, args.require_redshift_consistency)
    labeled_sdss = build_labeled_sdss(sdss, best_matches)
    summary = summarize_matches(external, sdss, matches, best_matches)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "external_csv": str(args.external_csv.resolve()),
        "sdss_csv": str(args.sdss_csv.resolve()),
        "output_dir": str(output_dir),
        "match_radius_arcsec": args.match_radius_arcsec,
        "redshift_tolerance": args.redshift_tolerance,
        "require_redshift_consistency": args.require_redshift_consistency,
        "excluded_classes": sorted(excluded_classes),
        "n_sdss_input": int(len(sdss)),
        "n_external_input_after_filters": int(len(external)),
        "n_all_matches": int(len(matches)),
        "n_unique_sdss_labeled": int(best_matches["specobjid"].nunique()) if not best_matches.empty else 0,
        "n_positive_labels_in_sdss": int(labeled_sdss["external_rare_label"].sum()),
        "notes": "Coordinate crossmatch only. Redshift consistency is recorded and can optionally restrict labels.",
    }
    write_outputs(output_dir, matches, best_matches, labeled_sdss, summary, metadata)

    print(f"Salida escrita en: {output_dir}")
    print(f"SDSS input: {metadata['n_sdss_input']}")
    print(f"Externos usados: {metadata['n_external_input_after_filters']}")
    print(f"Matches coordenados: {metadata['n_all_matches']}")
    print(f"SDSS etiquetadas como raras externas: {metadata['n_positive_labels_in_sdss']}")


if __name__ == "__main__":
    main()
