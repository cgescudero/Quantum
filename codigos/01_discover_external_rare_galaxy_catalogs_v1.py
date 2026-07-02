"""Descubrimiento de catálogos externos de galaxias raras para Parte 3.

Este script inicia la ruta supervisada/few-shot del proyecto Quantum/SDSS. Su
objetivo no es construir etiquetas finales ni cruzar catálogos todavía, sino
identificar fuentes externas potencialmente útiles para clases raras: Green
Peas, galaxias extremadamente pobres en metales (XMP), objetos high-O32 y
análogos/leakers de continuo de Lyman.

La salida principal es un inventario reproducible de catálogos y columnas. Las
únicas columnas realmente obligatorias para una validación externa posterior son
coordenadas RA/Dec; redshift, abundancias, ratios y líneas nebulares son campos
deseables que se registran para decidir qué catálogos merecen normalización en
un script posterior.

El script es deliberadamente conservador: escribe sólo dentro de Parte_3 por
defecto, no modifica datos SDSS previos y degrada con gracia si astroquery o
VizieR no están disponibles.
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_external_catalog_discovery_v1"
SCRIPT_VERSION = "v1"

DEFAULT_KEYWORDS = [
    "green pea galaxies",
    "Green Pea",
    "XMP galaxies",
    "extremely metal poor galaxies",
    "metal poor star forming galaxies",
    "high O32 galaxies",
    "O32 compact star forming galaxies",
    "LyC leakers",
    "Lyman continuum leakers",
    "compact star-forming galaxies Izotov",
]

CATALOG_COLUMNS = [
    "keyword",
    "catalog_id",
    "description",
    "rank_within_keyword",
    "download_attempted",
    "download_success",
    "n_tables",
    "error_message",
]

COLUMN_COLUMNS = [
    "keyword",
    "catalog_id",
    "table_index",
    "table_name",
    "n_rows",
    "column_name",
    "dtype",
    "unit",
    "description",
    "semantic_flag",
    "has_ra_candidate",
    "has_dec_candidate",
    "has_crossmatch_coordinates",
    "has_redshift_candidate",
    "has_metallicity_candidate",
    "has_line_flux_candidate",
    "has_o32_or_r23_candidate",
    "has_equivalent_width_candidate",
]


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Descubre catálogos externos VizieR para galaxias raras."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio de salida dentro de Parte_3.",
    )
    parser.add_argument(
        "--max-catalogs-per-keyword",
        type=int,
        default=5,
        help="Número máximo de catálogos VizieR a inspeccionar por keyword.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout aproximado para consultas VizieR, en segundos.",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default=",".join(DEFAULT_KEYWORDS),
        help="Lista de keywords separadas por coma.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Sólo busca catálogos; no descarga tablas.",
    )
    return parser.parse_args()


def parse_keywords(text: str) -> list[str]:
    """Convierte una cadena CSV de keywords en una lista limpia."""
    keywords = [item.strip() for item in text.split(",") if item.strip()]
    if not keywords:
        raise ValueError("Debe proporcionarse al menos una keyword.")
    return keywords


def validate_output_dir(output_dir: Path) -> Path:
    """Valida que la salida quede dentro de Parte_3 y crea subdirectorios."""
    resolved = output_dir.expanduser().resolve()
    allowed_root = SCRIPT_DIR.resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        raise ValueError(
            "Por seguridad, --output-dir debe estar dentro de Parte_3: "
            f"{allowed_root}"
        )
    (resolved / "raw_tables").mkdir(parents=True, exist_ok=True)
    (resolved / "logs").mkdir(parents=True, exist_ok=True)
    return resolved


def build_manual_suggestions() -> dict[str, Any]:
    """Devuelve búsquedas manuales/literatura útil si VizieR no basta."""
    return {
        "purpose": "Fallback manual para localizar catálogos externos de galaxias raras.",
        "minimum_future_crossmatch_columns": ["RA", "Dec"],
        "useful_future_columns": [
            "redshift",
            "12+log(O/H)",
            "[OII] flux",
            "[OIII] flux",
            "Halpha flux",
            "Hbeta flux",
            "[NII] flux",
            "O32",
            "R23",
            "equivalent widths",
            "object name or SDSS identifier",
        ],
        "manual_search_terms": [
            "Green Pea galaxies Cardamone 2009 catalog RA Dec redshift",
            "Yang 2017 Green Pea compact starburst catalog",
            "Izotov Green Pea Lyman continuum leakers table",
            "extremely metal poor galaxies XMP catalog RA Dec oxygen abundance",
            "Sanchez Almeida extremely metal poor galaxies catalog",
            "Morales-Luis XMP galaxies catalog",
            "Berg extremely metal poor galaxies oxygen abundance table",
            "Hirschauer extremely metal poor galaxies catalog",
            "high O32 galaxies SDSS catalog",
            "Lyman continuum leakers low redshift catalog Flury",
            "Izotov compact star-forming galaxies LyC leakers",
            "EMPRESS extremely metal poor galaxies catalog",
        ],
        "candidate_classes": {
            "green_peas": "Compactas, alta excitacion, usualmente con lineas intensas y redshift bajo/intermedio.",
            "xmp": "Galaxias con baja abundancia de oxigeno; ideal si hay 12+log(O/H).",
            "high_o32": "Objetos con [OIII]/[OII] alto; utiles para validar recuperacion de alta ionizacion.",
            "lyc_leakers": "Muestras pequenas pero fisicamente extremas; validar como enriquecimiento, no como clase completa.",
        },
    }


def import_vizier() -> tuple[bool, Any | None, str | None]:
    """Importa astroquery.vizier.Vizier si está disponible."""
    try:
        from astroquery.vizier import Vizier  # type: ignore

        return True, Vizier, None
    except Exception as exc:  # pragma: no cover - depende del entorno local
        return False, None, f"{type(exc).__name__}: {exc}"


def safe_description(resource: Any) -> str:
    """Extrae una descripción razonable de un recurso VizieR."""
    for attr in ("description", "title", "name"):
        value = getattr(resource, attr, None)
        if value:
            return str(value)
    return ""


def search_vizier_catalogs(
    vizier_cls: Any,
    keywords: list[str],
    max_catalogs_per_keyword: int,
    timeout: int,
    log_messages: list[str],
) -> list[dict[str, Any]]:
    """Busca catálogos candidatos con Vizier.find_catalogs."""
    if max_catalogs_per_keyword <= 0:
        raise ValueError("max_catalogs_per_keyword debe ser positivo.")
    try:
        vizier_cls.TIMEOUT = timeout
    except Exception as exc:
        log_messages.append(f"No se pudo fijar Vizier.TIMEOUT={timeout}: {exc}")

    records: list[dict[str, Any]] = []
    vizier = vizier_cls()
    for keyword in keywords:
        try:
            found = vizier.find_catalogs(keyword)
        except Exception as exc:
            log_messages.append(f"Fallo buscando keyword '{keyword}': {type(exc).__name__}: {exc}")
            continue

        for rank, (catalog_id, resource) in enumerate(list(found.items())[:max_catalogs_per_keyword], start=1):
            records.append(
                {
                    "keyword": keyword,
                    "catalog_id": str(catalog_id),
                    "description": safe_description(resource),
                    "rank_within_keyword": rank,
                    "download_attempted": False,
                    "download_success": False,
                    "n_tables": 0,
                    "error_message": "",
                }
            )
    return records


def classify_column_semantics(column_name: str, unit: str = "", description: str = "") -> str:
    """Clasifica columnas por posible utilidad física/observacional."""
    text = f"{column_name} {unit} {description}".lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)
    name = column_name.lower()
    name_compact = re.sub(r"[^a-z0-9]+", "", name)

    if re.fullmatch(r"_?ra(j2000)?|ra(icrs)?|radeg|alpha", name.replace("-", "_")):
        return "ra_candidate"
    if re.fullmatch(r"_?de(c|j2000)?|_?dec(j2000)?|dedeg|delta", name.replace("-", "_")):
        return "dec_candidate"
    if re.search(r"\b(redshift|zsp|z_spec|zhel|cz)\b", text) or name in {"z", "zspec", "z_sp"}:
        return "redshift_candidate"
    if any(token in compact for token in ["12logoh", "logoh", "oh", "oxygenabundance", "metallicity", "abund"]):
        return "metallicity_candidate"
    if any(token in compact for token in ["o32", "o3o2", "oiiioii"]):
        return "o32_candidate"
    if "r23" in compact:
        return "r23_candidate"
    if any(token in compact for token in ["equivalentwidth", "ewha", "ewhb", "ewoiii", "ewoii"]):
        return "ew_candidate"
    if any(
        token in compact
        for token in [
            "oii3726",
            "oii3727",
            "oii3729",
            "oiii4959",
            "oiii5007",
            "hbeta",
            "hb4861",
            "halpha",
            "ha6563",
            "nii6584",
            "sii6717",
            "sii6731",
            "flux",
            "fha",
            "fhb",
        ]
    ):
        return "line_flux_candidate"
    if (
        name_compact in {"id", "objid", "objectid", "name", "objname", "sdss", "plate", "mjd", "fiber", "pid", "recno"}
        or any(token in name_compact for token in ["name", "object", "sdss"])
    ):
        return "object_name_candidate"
    return "unknown"


def sanitize_filename(text: str) -> str:
    """Construye un nombre seguro de archivo a partir de texto libre."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return sanitized[:160] if sanitized else "unnamed"


def table_to_dataframe_safe(table: Any) -> pd.DataFrame:
    """Convierte una tabla Astropy a DataFrame preservando valores nulos."""
    try:
        return table.to_pandas()
    except Exception:
        return pd.DataFrame({name: table[name].filled(None).tolist() for name in table.colnames})


def column_description(table: Any, column_name: str) -> str:
    """Extrae descripción de columna Astropy si existe."""
    column = table[column_name]
    description = getattr(column, "description", None)
    return "" if description is None else str(description)


def column_unit(table: Any, column_name: str) -> str:
    """Extrae unidad de columna Astropy si existe."""
    unit = getattr(table[column_name], "unit", None)
    return "" if unit is None else str(unit)


def build_table_column_records(
    keyword: str,
    catalog_id: str,
    table_index: int,
    table: Any,
) -> list[dict[str, Any]]:
    """Construye inventario semántico de columnas para una tabla VizieR."""
    table_name = str(getattr(table, "meta", {}).get("name", f"table_{table_index}"))
    flags = []
    base_records = []
    for column_name in table.colnames:
        unit = column_unit(table, column_name)
        description = column_description(table, column_name)
        semantic = classify_column_semantics(column_name, unit, description)
        flags.append(semantic)
        base_records.append(
            {
                "keyword": keyword,
                "catalog_id": catalog_id,
                "table_index": table_index,
                "table_name": table_name,
                "n_rows": len(table),
                "column_name": column_name,
                "dtype": str(getattr(table[column_name], "dtype", "")),
                "unit": unit,
                "description": description,
                "semantic_flag": semantic,
            }
        )

    has_ra = "ra_candidate" in flags
    has_dec = "dec_candidate" in flags
    table_flags = {
        "has_ra_candidate": has_ra,
        "has_dec_candidate": has_dec,
        "has_crossmatch_coordinates": has_ra and has_dec,
        "has_redshift_candidate": "redshift_candidate" in flags,
        "has_metallicity_candidate": "metallicity_candidate" in flags,
        "has_line_flux_candidate": "line_flux_candidate" in flags,
        "has_o32_or_r23_candidate": bool({"o32_candidate", "r23_candidate"} & set(flags)),
        "has_equivalent_width_candidate": "ew_candidate" in flags,
    }
    return [{**record, **table_flags} for record in base_records]


def download_candidate_catalogs(
    vizier_cls: Any,
    catalog_records: list[dict[str, Any]],
    raw_tables_dir: Path,
    timeout: int,
    log_messages: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Descarga tablas candidatas y genera inventario de columnas."""
    try:
        vizier_cls.TIMEOUT = timeout
    except Exception:
        pass
    vizier = vizier_cls(columns=["**"])
    vizier.ROW_LIMIT = -1

    updated_records = []
    column_records: list[dict[str, Any]] = []
    seen_catalog_ids: set[str] = set()

    for record in catalog_records:
        current = dict(record)
        current["download_attempted"] = True
        catalog_id = current["catalog_id"]
        keyword = current["keyword"]

        if catalog_id in seen_catalog_ids:
            current["error_message"] = "Catalog already downloaded for an earlier keyword."
            updated_records.append(current)
            continue
        seen_catalog_ids.add(catalog_id)

        try:
            tables = vizier.get_catalogs(catalog_id)
            current["download_success"] = True
            current["n_tables"] = len(tables)
            keyword_dir = raw_tables_dir / f"keyword_{sanitize_filename(keyword)}"
            keyword_dir.mkdir(parents=True, exist_ok=True)

            for table_index, table in enumerate(tables):
                table_name = str(getattr(table, "meta", {}).get("name", f"table_{table_index}"))
                filename = f"{sanitize_filename(catalog_id)}__{sanitize_filename(table_name)}__table{table_index}.csv"
                table_to_dataframe_safe(table).to_csv(keyword_dir / filename, index=False)
                column_records.extend(build_table_column_records(keyword, catalog_id, table_index, table))
        except Exception as exc:
            current["download_success"] = False
            current["error_message"] = f"{type(exc).__name__}: {exc}"
            log_messages.append(f"Fallo descargando {catalog_id}: {traceback.format_exc()}")
        updated_records.append(current)
    return updated_records, column_records


def empty_catalog_inventory() -> pd.DataFrame:
    """Tabla vacía con esquema estable para inventario de catálogos."""
    return pd.DataFrame(columns=CATALOG_COLUMNS)


def empty_column_inventory() -> pd.DataFrame:
    """Tabla vacía con esquema estable para inventario de columnas."""
    return pd.DataFrame(columns=COLUMN_COLUMNS)


def write_outputs(
    catalog_inventory: list[dict[str, Any]],
    column_inventory: list[dict[str, Any]],
    metadata: dict[str, Any],
    manual_suggestions: dict[str, Any],
    output_dir: Path,
    log_messages: list[str],
) -> None:
    """Escribe CSV/JSON de salida de forma reproducible."""
    catalog_df = pd.DataFrame(catalog_inventory)
    if catalog_df.empty:
        catalog_df = empty_catalog_inventory()
    else:
        catalog_df = catalog_df.reindex(columns=CATALOG_COLUMNS)
    catalog_df.to_csv(output_dir / "catalog_inventory.csv", index=False)

    column_df = pd.DataFrame(column_inventory)
    if column_df.empty:
        column_df = empty_column_inventory()
    else:
        column_df = column_df.reindex(columns=COLUMN_COLUMNS)
    column_df.to_csv(output_dir / "column_inventory.csv", index=False)

    with (output_dir / "manual_targets_suggestions.json").open("w", encoding="utf-8") as handle:
        json.dump(manual_suggestions, handle, indent=2, ensure_ascii=False)
    with (output_dir / "discovery_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    with (output_dir / "logs" / "discovery_log.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(log_messages) + ("\n" if log_messages else ""))


def main() -> None:
    """Ejecuta el descubrimiento de catálogos externos candidatos."""
    args = parse_args()
    output_dir = validate_output_dir(args.output_dir)
    keywords = parse_keywords(args.keywords)
    log_messages: list[str] = []
    manual_suggestions = build_manual_suggestions()

    astroquery_available, vizier_cls, astroquery_error = import_vizier()
    catalog_inventory: list[dict[str, Any]] = []
    column_inventory: list[dict[str, Any]] = []
    global_errors: list[str] = []

    if not astroquery_available or vizier_cls is None:
        message = f"astroquery no disponible: {astroquery_error}"
        global_errors.append(message)
        log_messages.append(message)
    else:
        try:
            catalog_inventory = search_vizier_catalogs(
                vizier_cls=vizier_cls,
                keywords=keywords,
                max_catalogs_per_keyword=args.max_catalogs_per_keyword,
                timeout=args.timeout,
                log_messages=log_messages,
            )
            if not args.skip_download and catalog_inventory:
                catalog_inventory, column_inventory = download_candidate_catalogs(
                    vizier_cls=vizier_cls,
                    catalog_records=catalog_inventory,
                    raw_tables_dir=output_dir / "raw_tables",
                    timeout=args.timeout,
                    log_messages=log_messages,
                )
        except Exception as exc:
            message = f"Fallo global de VizieR: {type(exc).__name__}: {exc}"
            global_errors.append(message)
            log_messages.append(traceback.format_exc())

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "output_dir": str(output_dir),
        "keywords": keywords,
        "max_catalogs_per_keyword": args.max_catalogs_per_keyword,
        "timeout": args.timeout,
        "skip_download": args.skip_download,
        "astroquery_available": astroquery_available,
        "astroquery_error": astroquery_error,
        "global_errors": global_errors,
        "n_catalog_records": len(catalog_inventory),
        "n_unique_catalog_ids": len({record["catalog_id"] for record in catalog_inventory}),
        "n_downloaded_catalog_records": sum(bool(record.get("download_success")) for record in catalog_inventory),
        "n_column_records": len(column_inventory),
        "notes": (
            "Discovery-only step. RA/Dec candidates are required for later crossmatch; "
            "physical columns are only inventoried here."
        ),
    }

    write_outputs(
        catalog_inventory=catalog_inventory,
        column_inventory=column_inventory,
        metadata=metadata,
        manual_suggestions=manual_suggestions,
        output_dir=output_dir,
        log_messages=log_messages,
    )

    print(f"Salida escrita en: {output_dir}")
    print(f"Catálogos inventariados: {metadata['n_catalog_records']}")
    print(f"Registros de columnas: {metadata['n_column_records']}")
    if global_errors:
        print("Advertencia: hubo errores globales; revisar discovery_metadata.json y logs/discovery_log.txt")


if __name__ == "__main__":
    main()
