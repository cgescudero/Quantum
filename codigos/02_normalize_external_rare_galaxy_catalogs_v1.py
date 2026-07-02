"""Normalización de catálogos externos de galaxias raras para Parte 3.

Este script toma la salida cruda de ``01_discover_external_rare_galaxy_catalogs_v1.py``
y construye una tabla homogénea preliminar para validación externa. La tarea
central es convertir tablas VizieR heterogéneas a una representación común con
coordenadas en grados, clase externa tentativa y columnas físicas cuando estén
disponibles.

No realiza todavía crossmatch con SDSS. Ese paso debe hacerse después, usando
``ra_deg`` y ``dec_deg`` normalizados aquí. La normalización conserva trazabilidad
al catálogo, tabla y fila originales para poder auditar cada objeto.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DISCOVERY_DIR = SCRIPT_DIR / "results_external_catalog_discovery_v1"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_external_catalog_normalized_v1"
SCRIPT_VERSION = "v1"

NORMALIZED_COLUMNS = [
    "external_object_id",
    "external_class",
    "catalog_id",
    "catalog_description",
    "keyword",
    "table_name",
    "table_index",
    "raw_file",
    "raw_row_index",
    "object_name",
    "ra_deg",
    "dec_deg",
    "coord_source_columns",
    "coord_parse_status",
    "redshift",
    "redshift_source_column",
    "oxygen_abundance_12logOH",
    "oxygen_abundance_source_column",
    "log_stellar_mass",
    "log_stellar_mass_source_column",
    "o32_ratio",
    "o32_source_column",
    "ew_halpha",
    "ew_halpha_source_column",
    "has_coordinates",
    "has_redshift",
    "has_metallicity",
    "has_o32",
    "usable_for_crossmatch",
]


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Normaliza catálogos externos raros descargados desde VizieR."
    )
    parser.add_argument(
        "--discovery-dir",
        type=Path,
        default=DEFAULT_DISCOVERY_DIR,
        help="Directorio generado por el Script 01.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio de salida dentro de Parte_3.",
    )
    parser.add_argument(
        "--min-redshift",
        type=float,
        default=None,
        help="Redshift mínimo opcional para filtrar objetos normalizados.",
    )
    parser.add_argument(
        "--max-redshift",
        type=float,
        default=None,
        help="Redshift máximo opcional para filtrar objetos normalizados.",
    )
    parser.add_argument(
        "--require-redshift",
        action="store_true",
        help="Conserva sólo objetos con redshift normalizado finito.",
    )
    return parser.parse_args()


def validate_inside_part3(path: Path, label: str) -> Path:
    """Valida que una ruta esté dentro de Parte_3."""
    resolved = path.expanduser().resolve()
    allowed_root = SCRIPT_DIR.resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        raise ValueError(f"Por seguridad, {label} debe estar dentro de Parte_3: {allowed_root}")
    return resolved


def prepare_output_dir(output_dir: Path) -> Path:
    """Crea el directorio de salida validado."""
    resolved = validate_inside_part3(output_dir, "--output-dir")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_discovery_tables(discovery_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga inventarios del Script 01."""
    resolved = validate_inside_part3(discovery_dir, "--discovery-dir")
    catalog_path = resolved / "catalog_inventory.csv"
    column_path = resolved / "column_inventory.csv"
    if not catalog_path.exists():
        raise FileNotFoundError(f"No existe inventario de catálogos: {catalog_path}")
    if not column_path.exists():
        raise FileNotFoundError(f"No existe inventario de columnas: {column_path}")
    return pd.read_csv(catalog_path), pd.read_csv(column_path)


def sanitize_filename(text: str) -> str:
    """Replica la sanitización usada por el Script 01."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return sanitized[:160] if sanitized else "unnamed"


def raw_file_for_table(discovery_dir: Path, keyword: str, catalog_id: str, table_name: str, table_index: int) -> Path | None:
    """Localiza el CSV raw asociado a una tabla del inventario."""
    keyword_dir = discovery_dir / "raw_tables" / f"keyword_{sanitize_filename(keyword)}"
    expected = (
        f"{sanitize_filename(catalog_id)}__{sanitize_filename(table_name)}__table{int(table_index)}.csv"
    )
    candidate = keyword_dir / expected
    if candidate.exists():
        return candidate
    matches = sorted((discovery_dir / "raw_tables").glob(f"**/*__table{int(table_index)}.csv"))
    catalog_token = sanitize_filename(catalog_id)
    table_token = sanitize_filename(table_name)
    for path in matches:
        if catalog_token in path.name and table_token in path.name:
            return path
    return None


def infer_external_class(keyword: str, description: str, catalog_id: str) -> str:
    """Asigna una clase externa tentativa a partir de keyword/descripción."""
    text = f"{keyword} {description} {catalog_id}".lower()
    if "green pea" in text:
        return "green_pea"
    if "lyman" in text or "lyc" in text or "continuum leaker" in text:
        return "lyc_or_lae_candidate"
    if "o32" in text or "o3o2" in text:
        return "high_o32_candidate"
    if "xmp" in text or "extremely metal" in text or "metal-poor" in text or "metal poor" in text:
        return "xmp_or_metal_poor"
    return "rare_galaxy_candidate"


def lower_map(columns: list[str]) -> dict[str, str]:
    """Mapa de nombres normalizados a nombres originales."""
    return {re.sub(r"[^a-z0-9]+", "", col.lower()): col for col in columns}


def choose_column(columns: list[str], preferred: list[str], contains: list[str] | None = None) -> str | None:
    """Elige una columna por coincidencia exacta normalizada o por contenido."""
    mapping = lower_map(columns)
    for name in preferred:
        key = re.sub(r"[^a-z0-9]+", "", name.lower())
        if key in mapping:
            return mapping[key]
    if contains:
        for col in columns:
            key = re.sub(r"[^a-z0-9]+", "", col.lower())
            if any(token in key for token in contains):
                return col
    return None


def choose_coordinate_columns(columns: list[str]) -> tuple[str | None, str | None]:
    """Selecciona columnas candidatas RA/Dec."""
    ra_col = choose_column(columns, ["_RA", "RAJ2000", "RA_ICRS", "RA", "RAdeg", "ra_deg"])
    dec_col = choose_column(columns, ["_DE", "DEJ2000", "DECJ2000", "DE_ICRS", "DEC", "DE", "DEdeg", "dec_deg"])
    return ra_col, dec_col


def to_float(value: Any) -> float:
    """Convierte un valor escalar a float finito o NaN."""
    if value is None:
        return math.nan
    try:
        if pd.isna(value):
            return math.nan
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text in {"--", "...", "nan", "NaN"}:
        return math.nan
    text = text.replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if not match:
        return math.nan
    try:
        return float(match.group(0))
    except ValueError:
        return math.nan


def parse_coordinates(ra_value: Any, dec_value: Any, ra_col: str, dec_col: str) -> tuple[float, float, str]:
    """Convierte coordenadas a grados usando Astropy si hace falta."""
    ra_num = to_float(ra_value)
    dec_num = to_float(dec_value)
    if np.isfinite(ra_num) and np.isfinite(dec_num) and -90.0 <= dec_num <= 90.0:
        if 0.0 <= ra_num <= 360.0:
            return ra_num, dec_num, "numeric_degrees"

    try:
        from astropy.coordinates import SkyCoord  # type: ignore
        import astropy.units as u  # type: ignore

        coord = SkyCoord(str(ra_value), str(dec_value), unit=(u.hourangle, u.deg), frame="icrs")
        return float(coord.ra.deg), float(coord.dec.deg), "astropy_hourangle_deg"
    except Exception:
        pass

    try:
        from astropy.coordinates import SkyCoord  # type: ignore
        import astropy.units as u  # type: ignore

        coord = SkyCoord(str(ra_value), str(dec_value), unit=(u.deg, u.deg), frame="icrs")
        return float(coord.ra.deg), float(coord.dec.deg), "astropy_deg_deg"
    except Exception:
        return math.nan, math.nan, f"failed:{ra_col},{dec_col}"


def choose_object_name(row: pd.Series, columns: list[str]) -> tuple[str, str | None]:
    """Elige un identificador legible del objeto."""
    col = choose_column(
        columns,
        ["SDSS", "ID", "Name", "ObjName", "SimbadName", "OName", "SpObjID", "recno", "Seq"],
        contains=["sdss", "name", "obj", "simbad"],
    )
    if col is None:
        return "", None
    value = row.get(col, "")
    if pd.isna(value):
        return "", col
    return str(value), col


def choose_physical_columns(columns: list[str]) -> dict[str, str | None]:
    """Selecciona columnas físicas normalizables."""
    return {
        "redshift": choose_column(columns, ["zsp", "z_spec", "z", "redshift", "zhel", "cz"]),
        "metallicity": choose_column(
            columns,
            ["12+log(O/H)", "logO/H-DM", "logO/H", "(O/H)", "OH", "O_H", "metallicity"],
            contains=["12logoh", "logoh", "metallicity"],
        ),
        "log_mass": choose_column(columns, ["logM", "logMass", "log_stellar_mass", "lgm", "Mstar"]),
        "o32": choose_column(columns, ["[OIII]/[OII]", "O32", "O3O2", "OIII/OII"], contains=["oiiioii", "o32", "o3o2"]),
        "ew_halpha": choose_column(columns, ["EW(Ha)", "EW_Ha", "EWHa", "EW(Halpha)", "EWHalpha"], contains=["ewha", "ewhalpha"]),
    }


def table_groups(column_inventory: pd.DataFrame, catalog_inventory: pd.DataFrame) -> pd.DataFrame:
    """Construye una fila por tabla con coordenadas candidatas."""
    if column_inventory.empty:
        return pd.DataFrame()
    keys = ["keyword", "catalog_id", "table_index", "table_name"]
    grouped = (
        column_inventory.groupby(keys, dropna=False)
        .agg(
            n_rows=("n_rows", "first"),
            has_crossmatch_coordinates=("has_crossmatch_coordinates", "max"),
            has_redshift_candidate=("has_redshift_candidate", "max"),
            has_metallicity_candidate=("has_metallicity_candidate", "max"),
            has_o32_or_r23_candidate=("has_o32_or_r23_candidate", "max"),
        )
        .reset_index()
    )
    descriptions = catalog_inventory.drop_duplicates("catalog_id").loc[:, ["catalog_id", "description"]]
    return grouped.merge(descriptions, on="catalog_id", how="left")


def normalize_one_table(group: pd.Series, discovery_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normaliza una tabla raw individual."""
    raw_path = raw_file_for_table(
        discovery_dir=discovery_dir,
        keyword=str(group["keyword"]),
        catalog_id=str(group["catalog_id"]),
        table_name=str(group["table_name"]),
        table_index=int(group["table_index"]),
    )
    summary = {
        "keyword": group["keyword"],
        "catalog_id": group["catalog_id"],
        "table_name": group["table_name"],
        "table_index": int(group["table_index"]),
        "raw_file": "" if raw_path is None else str(raw_path),
        "input_rows": 0,
        "normalized_rows": 0,
        "usable_for_crossmatch_rows": 0,
        "status": "not_started",
        "message": "",
    }
    if raw_path is None:
        summary["status"] = "missing_raw_file"
        return pd.DataFrame(columns=NORMALIZED_COLUMNS), summary

    data = pd.read_csv(raw_path)
    summary["input_rows"] = len(data)
    columns = list(data.columns)
    ra_col, dec_col = choose_coordinate_columns(columns)
    if ra_col is None or dec_col is None:
        summary["status"] = "no_coordinate_columns"
        summary["message"] = "No RA/Dec columns selected."
        return pd.DataFrame(columns=NORMALIZED_COLUMNS), summary

    physical_cols = choose_physical_columns(columns)
    external_class = infer_external_class(
        str(group["keyword"]), str(group.get("description", "")), str(group["catalog_id"])
    )
    rows = []
    for raw_idx, row in data.iterrows():
        ra_deg, dec_deg, coord_status = parse_coordinates(row.get(ra_col), row.get(dec_col), ra_col, dec_col)
        object_name, _ = choose_object_name(row, columns)
        redshift = to_float(row.get(physical_cols["redshift"])) if physical_cols["redshift"] else math.nan
        metallicity = to_float(row.get(physical_cols["metallicity"])) if physical_cols["metallicity"] else math.nan
        log_mass = to_float(row.get(physical_cols["log_mass"])) if physical_cols["log_mass"] else math.nan
        o32 = to_float(row.get(physical_cols["o32"])) if physical_cols["o32"] else math.nan
        ew_halpha = to_float(row.get(physical_cols["ew_halpha"])) if physical_cols["ew_halpha"] else math.nan
        has_coords = np.isfinite(ra_deg) and np.isfinite(dec_deg)
        external_object_id = f"{group['catalog_id']}|{group['table_index']}|{raw_idx}"
        rows.append(
            {
                "external_object_id": external_object_id,
                "external_class": external_class,
                "catalog_id": group["catalog_id"],
                "catalog_description": group.get("description", ""),
                "keyword": group["keyword"],
                "table_name": group["table_name"],
                "table_index": int(group["table_index"]),
                "raw_file": str(raw_path),
                "raw_row_index": int(raw_idx),
                "object_name": object_name,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "coord_source_columns": f"{ra_col},{dec_col}",
                "coord_parse_status": coord_status,
                "redshift": redshift,
                "redshift_source_column": physical_cols["redshift"] or "",
                "oxygen_abundance_12logOH": metallicity,
                "oxygen_abundance_source_column": physical_cols["metallicity"] or "",
                "log_stellar_mass": log_mass,
                "log_stellar_mass_source_column": physical_cols["log_mass"] or "",
                "o32_ratio": o32,
                "o32_source_column": physical_cols["o32"] or "",
                "ew_halpha": ew_halpha,
                "ew_halpha_source_column": physical_cols["ew_halpha"] or "",
                "has_coordinates": bool(has_coords),
                "has_redshift": bool(np.isfinite(redshift)),
                "has_metallicity": bool(np.isfinite(metallicity)),
                "has_o32": bool(np.isfinite(o32)),
                "usable_for_crossmatch": bool(has_coords),
            }
        )

    result = pd.DataFrame(rows).reindex(columns=NORMALIZED_COLUMNS)
    summary["normalized_rows"] = len(result)
    summary["usable_for_crossmatch_rows"] = int(result["usable_for_crossmatch"].sum())
    summary["status"] = "ok"
    return result, summary


def apply_optional_filters(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Aplica filtros opcionales de redshift."""
    result = data.copy()
    if args.require_redshift:
        result = result.loc[np.isfinite(result["redshift"].to_numpy(dtype=float))]
    if args.min_redshift is not None:
        result = result.loc[result["redshift"] >= args.min_redshift]
    if args.max_redshift is not None:
        result = result.loc[result["redshift"] <= args.max_redshift]
    return result.reset_index(drop=True)


def write_outputs(
    normalized: pd.DataFrame,
    table_summary: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    """Guarda tablas normalizadas y metadatos."""
    normalized.to_csv(output_dir / "external_rare_galaxy_catalogs_normalized_v1.csv", index=False)
    table_summary.to_csv(output_dir / "normalization_table_summary_v1.csv", index=False)
    with (output_dir / "normalization_metadata_v1.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Ejecuta la normalización completa."""
    args = parse_args()
    discovery_dir = validate_inside_part3(args.discovery_dir, "--discovery-dir")
    output_dir = prepare_output_dir(args.output_dir)
    catalog_inventory, column_inventory = load_discovery_tables(discovery_dir)
    groups = table_groups(column_inventory, catalog_inventory)

    frames = []
    summaries = []
    for _, group in groups.iterrows():
        if not bool(group.get("has_crossmatch_coordinates", False)):
            summaries.append(
                {
                    "keyword": group["keyword"],
                    "catalog_id": group["catalog_id"],
                    "table_name": group["table_name"],
                    "table_index": int(group["table_index"]),
                    "raw_file": "",
                    "input_rows": int(group.get("n_rows", 0)),
                    "normalized_rows": 0,
                    "usable_for_crossmatch_rows": 0,
                    "status": "skipped_no_crossmatch_coordinates",
                    "message": "Table inventory lacks both RA and Dec candidates.",
                }
            )
            continue
        frame, summary = normalize_one_table(group, discovery_dir)
        frames.append(frame)
        summaries.append(summary)

    normalized = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=NORMALIZED_COLUMNS)
    before_filter_rows = len(normalized)
    normalized = apply_optional_filters(normalized, args)
    table_summary = pd.DataFrame(summaries)

    if not normalized.empty:
        normalized["coord_round_key"] = (
            normalized["ra_deg"].round(6).astype(str) + "," + normalized["dec_deg"].round(6).astype(str)
        )
        normalized["duplicate_coord_count"] = normalized.groupby("coord_round_key")["coord_round_key"].transform("size")
        normalized = normalized.drop(columns=["coord_round_key"])

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "discovery_dir": str(discovery_dir),
        "output_dir": str(output_dir),
        "n_input_catalog_records": int(len(catalog_inventory)),
        "n_input_column_records": int(len(column_inventory)),
        "n_candidate_tables": int(len(groups)),
        "n_rows_before_optional_filters": int(before_filter_rows),
        "n_rows_after_optional_filters": int(len(normalized)),
        "n_usable_for_crossmatch": int(normalized["usable_for_crossmatch"].sum()) if not normalized.empty else 0,
        "n_with_redshift": int(normalized["has_redshift"].sum()) if not normalized.empty else 0,
        "n_with_metallicity": int(normalized["has_metallicity"].sum()) if not normalized.empty else 0,
        "n_with_o32": int(normalized["has_o32"].sum()) if not normalized.empty else 0,
        "require_redshift": bool(args.require_redshift),
        "min_redshift": args.min_redshift,
        "max_redshift": args.max_redshift,
        "notes": "Normalization-only step. Crossmatch with SDSS is intentionally deferred to Script 03.",
    }
    write_outputs(normalized, table_summary, metadata, output_dir)

    print(f"Salida escrita en: {output_dir}")
    print(f"Objetos normalizados: {metadata['n_rows_after_optional_filters']}")
    print(f"Usables para crossmatch: {metadata['n_usable_for_crossmatch']}")
    print(f"Con redshift: {metadata['n_with_redshift']}")
    print(f"Con metalicidad: {metadata['n_with_metallicity']}")


if __name__ == "__main__":
    main()
