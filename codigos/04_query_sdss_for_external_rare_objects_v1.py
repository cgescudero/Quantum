"""Consulta SDSS alrededor de objetos externos raros.

Este script completa la ruta iniciada por los scripts 01-03: en lugar de buscar
los catálogos externos sólo dentro de la muestra SDSS previa de 10.000 galaxias,
consulta SDSS directamente alrededor de cada objeto externo normalizado. El
objetivo es recuperar espectros, líneas nebulares, masa estelar y metalicidad
MPA-JHU cuando existan, para construir una muestra positiva externa más directa.

La salida incluye una tabla de matches de posición, una tabla espectroscópica con
flujos/líneas y una tabla de features físicas en el mismo espíritu que
``sdss_galaxies_features_physical.csv``. No selecciona aún negativos ni entrena
modelos; eso queda para scripts posteriores.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXTERNAL_CSV = (
    SCRIPT_DIR
    / "results_external_catalog_normalized_v1"
    / "external_rare_galaxy_catalogs_normalized_v1.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_external_sdss_query_v1"
SCRIPT_VERSION = "v1"

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

SPECTROSCOPY_COLUMNS = [
    "specobjid",
    "plate",
    "mjd",
    "fiberid",
    "ra",
    "dec",
    "z",
    "h_alpha_flux",
    "h_alpha_flux_err",
    "h_beta_flux",
    "h_beta_flux_err",
    "oiii_5007_flux",
    "oiii_5007_flux_err",
    "nii_6584_flux",
    "nii_6584_flux_err",
    "oii_3726_flux",
    "oii_3726_flux_err",
    "oii_3729_flux",
    "oii_3729_flux_err",
    "sii_6717_flux",
    "sii_6717_flux_err",
    "sii_6731_flux",
    "sii_6731_flux_err",
    "bptclass",
    "log_stellar_mass",
    "metallicity",
]

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


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Consulta SDSS alrededor de catálogos externos raros normalizados."
    )
    parser.add_argument(
        "--external-csv",
        type=Path,
        default=DEFAULT_EXTERNAL_CSV,
        help="CSV normalizado generado por el Script 02.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio de salida dentro de Parte_3.",
    )
    parser.add_argument(
        "--search-radius-arcsec",
        type=float,
        default=2.0,
        help="Radio de búsqueda en SDSS alrededor de cada objeto externo.",
    )
    parser.add_argument(
        "--data-release",
        type=int,
        default=18,
        help="Data release de SDSS usado por astroquery.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout para consultas SDSS, en segundos.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=0,
        help="Número máximo de objetos externos a consultar. 0 significa todos.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Índice inicial de la tabla externa filtrada. Útil para correr por bloques.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=0,
        help="Índice final exclusivo de la tabla externa filtrada. 0 significa hasta el final.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.05,
        help="Pausa entre consultas de región para no saturar SDSS.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Tamaño de lote para recuperar líneas por specobjid.",
    )
    parser.add_argument(
        "--exclude-classes",
        type=str,
        default="lyc_or_lae_candidate",
        help="Clases externas a excluir antes de consultar SDSS, separadas por coma.",
    )
    parser.add_argument(
        "--require-external-redshift",
        action="store_true",
        help="Consulta sólo objetos externos con redshift disponible.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Guarda matches/fallos parciales cada N objetos consultados. 0 desactiva checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reusa checkpoints parciales y omite objetos externos ya consultados.",
    )
    parser.add_argument(
        "--reuse-region-matches",
        action="store_true",
        help="No consulta regiones de nuevo; reutiliza external_sdss_region_matches_v1.csv del output-dir.",
    )
    return parser.parse_args()


def parse_class_list(text: str) -> set[str]:
    """Parsea clases separadas por coma."""
    return {item.strip() for item in text.split(",") if item.strip()}


def validate_output_dir(output_dir: Path) -> Path:
    """Valida que la salida quede dentro de Parte_3."""
    resolved = output_dir.expanduser().resolve()
    allowed_root = SCRIPT_DIR.resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        raise ValueError(f"Por seguridad, --output-dir debe estar dentro de Parte_3: {allowed_root}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_external_catalog(external_csv: Path, excluded_classes: set[str], require_redshift: bool) -> pd.DataFrame:
    """Carga y filtra objetos externos normalizados."""
    if not external_csv.exists():
        raise FileNotFoundError(f"No existe archivo externo normalizado: {external_csv}")
    data = pd.read_csv(external_csv)
    missing = sorted(set(EXTERNAL_REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas externas obligatorias: {missing}")
    for column in ["ra_deg", "dec_deg", "redshift", "oxygen_abundance_12logOH"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    mask = data["usable_for_crossmatch"].astype(bool)
    mask &= np.isfinite(data["ra_deg"]) & np.isfinite(data["dec_deg"])
    mask &= ~data["external_class"].astype(str).isin(excluded_classes)
    if require_redshift:
        mask &= np.isfinite(data["redshift"])
    return data.loc[mask].reset_index(drop=True)


def import_sdss_tools(timeout: int) -> tuple[Any, Any, Any]:
    """Importa astroquery/astropy necesarios para consultas por región."""
    try:
        from astroquery.sdss import SDSS  # type: ignore
        from astropy.coordinates import SkyCoord  # type: ignore
        import astropy.units as u  # type: ignore
    except Exception as exc:  # pragma: no cover - depende del entorno local
        raise ImportError("Este script requiere astroquery y astropy.") from exc
    SDSS.TIMEOUT = timeout
    return SDSS, SkyCoord, u


def read_optional_csv(path: Path) -> pd.DataFrame:
    """Lee un CSV si existe; si no, devuelve una tabla vacía."""
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def write_region_checkpoints(
    output_dir: Path,
    matches: pd.DataFrame,
    failures: pd.DataFrame,
    queried: pd.DataFrame,
) -> None:
    """Guarda checkpoints parciales de consultas por región."""
    matches.to_csv(output_dir / "external_sdss_region_matches_partial_v1.csv", index=False)
    failures.to_csv(output_dir / "external_sdss_region_failures_partial_v1.csv", index=False)
    queried.to_csv(output_dir / "external_sdss_region_queried_partial_v1.csv", index=False)


def query_region_matches(
    external: pd.DataFrame,
    radius_arcsec: float,
    data_release: int,
    timeout: int,
    sleep_sec: float,
    output_dir: Path,
    checkpoint_every: int,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Consulta SDSS por región para cada objeto externo."""
    if radius_arcsec <= 0:
        raise ValueError("search_radius_arcsec debe ser positivo.")
    SDSS, SkyCoord, u = import_sdss_tools(timeout)
    existing_matches = read_optional_csv(output_dir / "external_sdss_region_matches_partial_v1.csv") if resume else pd.DataFrame()
    existing_failures = read_optional_csv(output_dir / "external_sdss_region_failures_partial_v1.csv") if resume else pd.DataFrame()
    existing_queried = read_optional_csv(output_dir / "external_sdss_region_queried_partial_v1.csv") if resume else pd.DataFrame()
    match_rows: list[dict[str, Any]] = existing_matches.to_dict("records") if not existing_matches.empty else []
    failure_rows: list[dict[str, Any]] = existing_failures.to_dict("records") if not existing_failures.empty else []
    queried_rows: list[dict[str, Any]] = existing_queried.to_dict("records") if not existing_queried.empty else []
    done_ids = set()
    if resume:
        if not existing_queried.empty and "external_object_id" in existing_queried.columns:
            done_ids |= set(existing_queried["external_object_id"].astype(str))

    if done_ids:
        print(f"Resume activo: {len(done_ids)} objetos externos ya tienen checkpoint y serán omitidos.", flush=True)

    total = len(external)
    for loop_index, (idx, row) in enumerate(external.iterrows(), start=1):
        external_id = str(row["external_object_id"])
        if external_id in done_ids:
            continue
        coord = SkyCoord(ra=float(row["ra_deg"]) * u.deg, dec=float(row["dec_deg"]) * u.deg)
        before_matches = len(match_rows)
        try:
            result = SDSS.query_region(
                coord,
                radius=radius_arcsec * u.arcsec,
                spectro=True,
                data_release=data_release,
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "external_object_id": external_id,
                    "external_class": row["external_class"],
                    "ra_deg": row["ra_deg"],
                    "dec_deg": row["dec_deg"],
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
            )
            result = None

        if result is None or len(result) == 0:
            frame = pd.DataFrame()
        else:
            frame = result.to_pandas()
        if not frame.empty:
            for _, sdss_row in frame.iterrows():
                specobjid = sdss_row.get("specobjid", sdss_row.get("specObjID", np.nan))
                if pd.isna(specobjid):
                    continue
                match_rows.append(
                    {
                        "external_object_id": row["external_object_id"],
                        "external_class": row["external_class"],
                        "external_catalog_id": row["catalog_id"],
                        "external_catalog_description": row.get("catalog_description", ""),
                        "external_keyword": row.get("keyword", ""),
                        "external_object_name": row.get("object_name", ""),
                        "external_ra_deg": row["ra_deg"],
                        "external_dec_deg": row["dec_deg"],
                        "external_redshift": row.get("redshift", np.nan),
                        "external_oxygen_abundance_12logOH": row.get("oxygen_abundance_12logOH", np.nan),
                        "sdss_specobjid_region": str(specobjid),
                        "sdss_region_ra": sdss_row.get("ra", np.nan),
                        "sdss_region_dec": sdss_row.get("dec", np.nan),
                        "sdss_region_run2d": sdss_row.get("run2d", ""),
                        "sdss_region_plate": sdss_row.get("plate", np.nan),
                        "sdss_region_mjd": sdss_row.get("mjd", np.nan),
                        "sdss_region_fiberid": sdss_row.get("fiberID", sdss_row.get("fiberid", np.nan)),
                        "external_row_index": int(idx),
                    }
                )
        new_matches = len(match_rows) - before_matches
        queried_rows.append(
            {
                "external_object_id": external_id,
                "external_class": row["external_class"],
                "external_row_index": int(idx),
                "n_matches": int(new_matches),
                "status": "queried",
            }
        )
        if loop_index == 1 or loop_index % max(checkpoint_every, 1) == 0 or loop_index == total:
            print(
                f"Consultados {loop_index}/{total} objetos del bloque; "
                f"matches acumulados={len(match_rows)}; fallos={len(failure_rows)}; "
                f"último objeto matches={new_matches}",
                flush=True,
            )
        if checkpoint_every > 0 and (loop_index % checkpoint_every == 0 or loop_index == total):
            write_region_checkpoints(
                output_dir,
                pd.DataFrame(match_rows),
                pd.DataFrame(failure_rows),
                pd.DataFrame(queried_rows),
            )
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    matches = pd.DataFrame(match_rows)
    failures = pd.DataFrame(failure_rows)
    if not matches.empty:
        matches = matches.drop_duplicates(["external_object_id", "sdss_specobjid_region"]).reset_index(drop=True)
    return matches, failures


def normalize_specobjid(value: Any) -> str:
    """Normaliza specobjid a cadena entera sin notación científica."""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = str(value).strip()
    if not text:
        return ""
    if re_fullmatch_digits(text):
        return text
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def normalize_int_value(value: Any) -> str:
    """Normaliza identificadores enteros pequeños como plate/mjd/fiberid."""
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
        return ""


def re_fullmatch_digits(text: str) -> bool:
    """Devuelve True si el texto contiene sólo dígitos decimales."""
    return text.isdigit()


def build_plate_mjd_fiber_batches(matches: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    """Construye lotes únicos de plate/mjd/fiberid para consulta SQL."""
    if batch_size <= 0:
        raise ValueError("batch_size debe ser positivo.")
    required = ["sdss_region_plate", "sdss_region_mjd", "sdss_region_fiberid"]
    missing = sorted(set(required) - set(matches.columns))
    if missing:
        raise ValueError(f"Faltan columnas plate/mjd/fiberid en matches: {missing}")
    ids = matches.loc[:, required].copy()
    ids["plate"] = ids["sdss_region_plate"].map(normalize_int_value)
    ids["mjd"] = ids["sdss_region_mjd"].map(normalize_int_value)
    ids["fiberid"] = ids["sdss_region_fiberid"].map(normalize_int_value)
    ids = ids.loc[(ids["plate"] != "") & (ids["mjd"] != "") & (ids["fiberid"] != "")]
    ids = ids.loc[:, ["plate", "mjd", "fiberid"]].drop_duplicates().reset_index(drop=True)
    return [ids.iloc[start : start + batch_size].copy() for start in range(0, len(ids), batch_size)]


def build_spectroscopy_query(batch: pd.DataFrame) -> str:
    """Construye consulta SQL para recuperar líneas usando plate/mjd/fiberid."""
    clauses = []
    for _, row in batch.iterrows():
        clauses.append(
            f"(s.plate = {row['plate']} AND s.mjd = {row['mjd']} AND s.fiberid = {row['fiberid']})"
        )
    where_clause = " OR\n    ".join(clauses)
    return f"""
SELECT
    s.specobjid,
    s.plate,
    s.mjd,
    s.fiberid,
    s.ra,
    s.dec,
    s.z,
    l.h_alpha_flux,
    l.h_alpha_flux_err,
    l.h_beta_flux,
    l.h_beta_flux_err,
    l.oiii_5007_flux,
    l.oiii_5007_flux_err,
    l.nii_6584_flux,
    l.nii_6584_flux_err,
    l.oii_3726_flux,
    l.oii_3726_flux_err,
    l.oii_3729_flux,
    l.oii_3729_flux_err,
    l.sii_6717_flux,
    l.sii_6717_flux_err,
    l.sii_6731_flux,
    l.sii_6731_flux_err,
    e.bptclass,
    e.lgm_tot_p50 AS log_stellar_mass,
    e.oh_p50 AS metallicity
FROM SpecObj AS s
JOIN galSpecLine AS l ON s.specobjid = l.specobjid
JOIN galSpecExtra AS e ON s.specobjid = e.specobjid
WHERE
    {where_clause}
"""


def fetch_sdss_sql(query: str, data_release: int, timeout: int) -> pd.DataFrame:
    """Ejecuta una consulta SQL SDSS y devuelve DataFrame."""
    from astroquery.sdss import SDSS  # type: ignore

    SDSS.TIMEOUT = timeout
    response = SDSS.query_sql_async(query, data_release=data_release)
    if response.status_code != requests.codes.ok:
        raise RuntimeError(f"SDSS devolvió HTTP {response.status_code}: {response.text[:500]}")
    text = response.text.strip()
    if not text:
        return pd.DataFrame()
    if text.startswith("<"):
        raise RuntimeError(f"SDSS devolvió HTML en lugar de CSV: {text[:1000]}")
    return pd.read_csv(io.StringIO(text), comment="#")


def fetch_spectroscopy_for_matches(
    matches: pd.DataFrame,
    batch_size: int,
    data_release: int,
    timeout: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recupera líneas y propiedades para specobjids encontrados."""
    if matches.empty:
        return pd.DataFrame(columns=SPECTROSCOPY_COLUMNS), pd.DataFrame()
    batches = build_plate_mjd_fiber_batches(matches, batch_size)
    frames = []
    failures = []
    print(f"Recuperando espectros/líneas en {len(batches)} lotes SQL de hasta {batch_size} objetos.", flush=True)
    for batch_index, batch in enumerate(batches):
        try:
            frame = fetch_sdss_sql(build_spectroscopy_query(batch), data_release, timeout)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            failures.append(
                {
                    "batch_index": batch_index,
                    "n_specobjids": len(batch),
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
            )
        print(
            f"Lote SQL {batch_index + 1}/{len(batches)} procesado; "
            f"filas acumuladas={sum(len(frame) for frame in frames)}; fallos={len(failures)}",
            flush=True,
        )
    spectroscopy = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SPECTROSCOPY_COLUMNS)
    if not spectroscopy.empty:
        spectroscopy["specobjid"] = spectroscopy["specobjid"].map(normalize_specobjid)
        for column in ["plate", "mjd", "fiberid"]:
            spectroscopy[column] = spectroscopy[column].map(normalize_int_value)
        spectroscopy["pmf_key"] = spectroscopy["plate"] + "-" + spectroscopy["mjd"] + "-" + spectroscopy["fiberid"]
        spectroscopy = spectroscopy.drop_duplicates("pmf_key").reset_index(drop=True)
    return spectroscopy, pd.DataFrame(failures)


def attach_external_context(matches: pd.DataFrame, spectroscopy: pd.DataFrame) -> pd.DataFrame:
    """Une contexto externo a la tabla espectroscópica SDSS."""
    if matches.empty or spectroscopy.empty:
        return pd.DataFrame()
    context = matches.copy()
    context["specobjid"] = context["sdss_specobjid_region"].map(normalize_specobjid)
    context["plate"] = context["sdss_region_plate"].map(normalize_int_value)
    context["mjd"] = context["sdss_region_mjd"].map(normalize_int_value)
    context["fiberid"] = context["sdss_region_fiberid"].map(normalize_int_value)
    context["pmf_key"] = context["plate"] + "-" + context["mjd"] + "-" + context["fiberid"]
    merged = context.merge(spectroscopy, on="pmf_key", how="inner", suffixes=("_region", ""))
    for column in ["external_redshift", "z"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["has_both_redshifts"] = np.isfinite(merged["external_redshift"]) & np.isfinite(merged["z"])
    merged["delta_redshift"] = np.where(
        merged["has_both_redshifts"],
        np.abs(merged["external_redshift"] - merged["z"]),
        np.nan,
    )
    return merged.sort_values(["external_object_id", "specobjid"]).reset_index(drop=True)


def build_physical_features(spectroscopy_with_context: pd.DataFrame) -> pd.DataFrame:
    """Construye features físicas comparables a la muestra SDSS previa."""
    if spectroscopy_with_context.empty:
        return pd.DataFrame()
    data = spectroscopy_with_context.copy()
    numeric = SPECTROSCOPY_COLUMNS[3:]
    for column in numeric:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    positive_flux_columns = [
        "h_alpha_flux",
        "h_beta_flux",
        "oiii_5007_flux",
        "nii_6584_flux",
        "oii_3726_flux",
        "oii_3729_flux",
        "sii_6717_flux",
        "sii_6731_flux",
    ]
    finite = np.isfinite(data[["ra", "dec", "z", "log_stellar_mass", "metallicity"] + positive_flux_columns]).all(axis=1)
    positive = (data[positive_flux_columns] > 0).all(axis=1)
    physical_range = (
        data["z"].between(0.0, 1.0)
        & data["log_stellar_mass"].between(0.0, 13.5)
        & data["metallicity"].between(7.0, 10.0)
    )
    valid = finite & positive & physical_range
    data = data.loc[valid].copy()
    if data.empty:
        return pd.DataFrame()

    oii_total = data["oii_3726_flux"] + data["oii_3729_flux"]
    sii_total = data["sii_6717_flux"] + data["sii_6731_flux"]
    features = data.loc[
        :,
        [
            "external_object_id",
            "external_class",
            "external_catalog_id",
            "external_object_name",
            "external_redshift",
            "external_oxygen_abundance_12logOH",
            "specobjid",
            "ra",
            "dec",
            "z",
            "log_stellar_mass",
            "metallicity",
            "bptclass",
            "delta_redshift",
        ],
    ].copy()
    features["log_nii_ha"] = np.log10(data["nii_6584_flux"] / data["h_alpha_flux"])
    features["log_oiii_hb"] = np.log10(data["oiii_5007_flux"] / data["h_beta_flux"])
    features["log_oii_hb"] = np.log10(oii_total / data["h_beta_flux"])
    features["log_sii_ha"] = np.log10(sii_total / data["h_alpha_flux"])
    features["log_r23"] = np.log10((oii_total + data["oiii_5007_flux"]) / data["h_beta_flux"])
    features["log_o32"] = np.log10(data["oiii_5007_flux"] / oii_total)
    features["external_sdss_positive_label"] = 1
    return features.reset_index(drop=True)


def write_outputs(
    output_dir: Path,
    region_matches: pd.DataFrame,
    region_failures: pd.DataFrame,
    spectroscopy: pd.DataFrame,
    sql_failures: pd.DataFrame,
    spectroscopy_with_context: pd.DataFrame,
    features: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """Guarda productos de la consulta SDSS directa."""
    region_matches.to_csv(output_dir / "external_sdss_region_matches_v1.csv", index=False)
    region_failures.to_csv(output_dir / "external_sdss_region_failures_v1.csv", index=False)
    spectroscopy.to_csv(output_dir / "external_sdss_spectroscopy_v1.csv", index=False)
    sql_failures.to_csv(output_dir / "external_sdss_spectroscopy_failures_v1.csv", index=False)
    spectroscopy_with_context.to_csv(output_dir / "external_sdss_spectroscopy_with_context_v1.csv", index=False)
    features.to_csv(output_dir / "external_sdss_features_physical_v1.csv", index=False)
    with (output_dir / "external_sdss_query_metadata_v1.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Ejecuta la búsqueda directa en SDSS para objetos externos raros."""
    args = parse_args()
    output_dir = validate_output_dir(args.output_dir)
    excluded_classes = parse_class_list(args.exclude_classes)
    external = load_external_catalog(args.external_csv, excluded_classes, args.require_external_redshift)
    if args.start_index < 0:
        raise ValueError("start-index no puede ser negativo.")
    if args.end_index < 0:
        raise ValueError("end-index no puede ser negativo.")
    if args.end_index > 0 and args.end_index < args.start_index:
        raise ValueError("end-index debe ser mayor o igual que start-index.")
    total_external_after_filters = len(external)
    end_index = args.end_index if args.end_index > 0 else len(external)
    external = external.iloc[args.start_index : end_index].reset_index(drop=True)
    if args.max_objects > 0:
        external = external.iloc[: args.max_objects].reset_index(drop=True)
    print(
        f"Objetos externos tras filtros: {total_external_after_filters}; "
        f"bloque a consultar: {len(external)} "
        f"(start={args.start_index}, end={end_index}, max_objects={args.max_objects})",
        flush=True,
    )

    started = time.perf_counter()
    if args.reuse_region_matches:
        region_path = output_dir / "external_sdss_region_matches_v1.csv"
        if not region_path.exists():
            raise FileNotFoundError(
                "--reuse-region-matches requiere que exista: "
                f"{region_path}"
            )
        region_matches = pd.read_csv(region_path)
        region_failures = read_optional_csv(output_dir / "external_sdss_region_failures_v1.csv")
        print(f"Reutilizando matches de región existentes: {len(region_matches)} filas.", flush=True)
    else:
        region_matches, region_failures = query_region_matches(
            external=external,
            radius_arcsec=args.search_radius_arcsec,
            data_release=args.data_release,
            timeout=args.timeout,
            sleep_sec=args.sleep_sec,
            output_dir=output_dir,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
        )
    spectroscopy, sql_failures = fetch_spectroscopy_for_matches(
        matches=region_matches,
        batch_size=args.batch_size,
        data_release=args.data_release,
        timeout=args.timeout,
    )
    spectroscopy_with_context = attach_external_context(region_matches, spectroscopy)
    features = build_physical_features(spectroscopy_with_context)
    elapsed = time.perf_counter() - started

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "external_csv": str(args.external_csv.resolve()),
        "output_dir": str(output_dir),
        "data_release": args.data_release,
        "search_radius_arcsec": args.search_radius_arcsec,
        "excluded_classes": sorted(excluded_classes),
        "require_external_redshift": args.require_external_redshift,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "max_objects": args.max_objects,
        "checkpoint_every": args.checkpoint_every,
        "resume": args.resume,
        "reuse_region_matches": args.reuse_region_matches,
        "n_external_after_filters_before_range": int(total_external_after_filters),
        "n_external_queried": int(len(external)),
        "n_region_matches": int(len(region_matches)),
        "n_unique_specobjids_region": int(region_matches["sdss_specobjid_region"].nunique()) if not region_matches.empty else 0,
        "n_region_failures": int(len(region_failures)),
        "n_spectroscopy_rows": int(len(spectroscopy)),
        "n_spectroscopy_with_context_rows": int(len(spectroscopy_with_context)),
        "n_feature_rows": int(len(features)),
        "n_sql_failures": int(len(sql_failures)),
        "elapsed_sec": elapsed,
        "notes": "Direct SDSS query around external rare objects. No negative/control sample is built here.",
    }
    write_outputs(
        output_dir=output_dir,
        region_matches=region_matches,
        region_failures=region_failures,
        spectroscopy=spectroscopy,
        sql_failures=sql_failures,
        spectroscopy_with_context=spectroscopy_with_context,
        features=features,
        metadata=metadata,
    )

    print(f"Salida escrita en: {output_dir}")
    print(f"Externos consultados: {metadata['n_external_queried']}")
    print(f"Matches SDSS por región: {metadata['n_region_matches']}")
    print(f"SpecObjID únicos: {metadata['n_unique_specobjids_region']}")
    print(f"Filas con espectros/líneas: {metadata['n_spectroscopy_rows']}")
    print(f"Filas de features físicas: {metadata['n_feature_rows']}")


if __name__ == "__main__":
    main()
