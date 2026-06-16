"""Descarga robusta de galaxias SDSS para regresión de metalicidad.

El script consulta SDSS DR17/MPA-JHU, aplica cortes básicos de calidad
espectroscópica y guarda un CSV reproducible para los scripts ML/QML.

No calcula ratios físicos derivados: conserva los flujos y errores medidos.
La construcción de features físicos debe hacerse en el pipeline de ML para
mantener trazabilidad entre datos crudos, limpieza y transformación.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astroquery.sdss import SDSS


DATA_RELEASE = 18
N_GALAXIES = 10_000
BATCH_SIZE = 1_000
MIN_SNR = 3.0
SDSS.TIMEOUT = 180

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR / "sdss_galaxies_spectroscopy_v2.csv"

REQUIRED_COLUMNS = [
    "specobjid",
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

PHYSICAL_COLUMNS = [
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
    "log_stellar_mass",
    "metallicity",
]


def build_query(
    n_galaxies: int = BATCH_SIZE,
    min_snr: float = MIN_SNR,
    min_specobjid: int = 0,
) -> str:
    """Construye la consulta SQL para SDSS.

    Parameters
    ----------
    n_galaxies:
        Número máximo de objetos a solicitar con ``TOP``.
    min_snr:
        Relación señal/ruido mínima para las líneas usadas como features.
    min_specobjid:
        Límite inferior exclusivo de ``specobjid`` para descargar por lotes
        reproducibles sin forzar un ``ORDER BY`` global de 10 000 objetos.

    Returns
    -------
    str
        Consulta SQL para SkyServer/SDSS.

    Notes
    -----
    La muestra se restringe a galaxias clasificadas como star-forming por
    ``bptclass = 1`` y a metalicidades nebulares físicamente razonables en
    escala ``12 + log(O/H)``.
    """
    if n_galaxies <= 0:
        raise ValueError("n_galaxies debe ser positivo.")
    if min_snr <= 0:
        raise ValueError("min_snr debe ser positivo.")
    if min_specobjid < 0:
        raise ValueError("min_specobjid no puede ser negativo.")

    return f"""
SELECT TOP {n_galaxies}
    s.specobjid,
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
    s.[class] = 'GALAXY'
    AND s.specobjid > CAST({min_specobjid} AS BIGINT)
    AND s.z BETWEEN 0.02 AND 0.10
    AND e.bptclass = 1
    AND e.oh_p50 BETWEEN 7.0 AND 10.0
    AND e.lgm_tot_p50 BETWEEN 6.0 AND 13.0
    AND l.h_alpha_flux > 0
    AND l.h_beta_flux > 0
    AND l.oiii_5007_flux > 0
    AND l.nii_6584_flux > 0
    AND l.oii_3726_flux > 0
    AND l.oii_3729_flux > 0
    AND l.sii_6717_flux > 0
    AND l.sii_6731_flux > 0
    AND l.h_alpha_flux_err > 0
    AND l.h_beta_flux_err > 0
    AND l.oiii_5007_flux_err > 0
    AND l.nii_6584_flux_err > 0
    AND l.oii_3726_flux_err > 0
    AND l.oii_3729_flux_err > 0
    AND l.sii_6717_flux_err > 0
    AND l.sii_6731_flux_err > 0
    AND l.h_alpha_flux / l.h_alpha_flux_err >= {min_snr}
    AND l.h_beta_flux / l.h_beta_flux_err >= {min_snr}
    AND l.oiii_5007_flux / l.oiii_5007_flux_err >= {min_snr}
    AND l.nii_6584_flux / l.nii_6584_flux_err >= {min_snr}
    AND l.oii_3726_flux / l.oii_3726_flux_err >= {min_snr}
    AND l.oii_3729_flux / l.oii_3729_flux_err >= {min_snr}
    AND l.sii_6717_flux / l.sii_6717_flux_err >= {min_snr}
    AND l.sii_6731_flux / l.sii_6731_flux_err >= {min_snr}
ORDER BY s.specobjid
"""


def fetch_sdss_csv(query: str, data_release: int = DATA_RELEASE) -> pd.DataFrame:
    """Ejecuta una consulta SQL en SDSS y devuelve un DataFrame.

    Parameters
    ----------
    query:
        Consulta SQL válida para SDSS SkyServer.
    data_release:
        Data release de SDSS usado por ``astroquery.sdss.SDSS``.

    Returns
    -------
    pandas.DataFrame
        Tabla devuelta por SDSS.

    Raises
    ------
    RuntimeError
        Si SDSS devuelve HTML, un código HTTP no exitoso o una respuesta vacía.
    pandas.errors.EmptyDataError
        Si la respuesta no puede interpretarse como CSV.
    """
    response = SDSS.query_sql_async(query, data_release=data_release)

    if response.status_code != requests.codes.ok:
        raise RuntimeError(
            f"SDSS devolvió HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    text = response.text.strip()
    if not text:
        raise RuntimeError("SDSS devolvió una respuesta vacía.")
    if text.startswith("<"):
        raise RuntimeError(
            "SDSS devolvió HTML en lugar de CSV. Primeros caracteres:\n"
            f"{text[:1200]}"
        )

    return pd.read_csv(io.StringIO(text), comment="#")


def fetch_sdss_batches(
    n_total: int = N_GALAXIES,
    batch_size: int = BATCH_SIZE,
    min_snr: float = MIN_SNR,
) -> pd.DataFrame:
    """Descarga la muestra SDSS en lotes ordenados por ``specobjid``.

    Parameters
    ----------
    n_total:
        Número máximo de galaxias a descargar.
    batch_size:
        Tamaño de cada consulta ``TOP``. Un valor de 1000 evita errores de
        runtime de SkyServer que aparecen al ordenar 10 000 objetos de golpe.
    min_snr:
        Relación señal/ruido mínima aplicada en SQL.

    Returns
    -------
    pandas.DataFrame
        Concatenación de los lotes descargados, truncada a ``n_total`` filas.

    Raises
    ------
    RuntimeError
        Si SDSS no devuelve ningún objeto válido.
    """
    if n_total <= 0:
        raise ValueError("n_total debe ser positivo.")
    if batch_size <= 0:
        raise ValueError("batch_size debe ser positivo.")

    frames: list[pd.DataFrame] = []
    last_specobjid = 0

    while sum(len(frame) for frame in frames) < n_total:
        remaining = n_total - sum(len(frame) for frame in frames)
        current_batch_size = min(batch_size, remaining)
        query = build_query(
            n_galaxies=current_batch_size,
            min_snr=min_snr,
            min_specobjid=last_specobjid,
        )

        print(
            "  -> Descargando lote "
            f"TOP {current_batch_size} con specobjid > {last_specobjid}..."
        )
        batch = fetch_sdss_csv(query)
        if batch.empty:
            break

        batch["specobjid"] = pd.to_numeric(batch["specobjid"], errors="raise")
        batch = batch.sort_values("specobjid").reset_index(drop=True)
        frames.append(batch)
        last_specobjid = int(batch["specobjid"].max())

        print(f"     Lote recibido: {len(batch)} objetos. Total parcial: {sum(len(frame) for frame in frames)}")

        if len(batch) < current_batch_size:
            break

    if not frames:
        raise RuntimeError("SDSS no devolvió objetos para los cortes solicitados.")

    return pd.concat(frames, ignore_index=True).head(n_total)


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Valida columnas, finitud y rangos físicos básicos.

    Parameters
    ----------
    df:
        DataFrame descargado desde SDSS.

    Returns
    -------
    pandas.DataFrame
        Copia limpia del DataFrame. Informa cuántas filas se descartan si
        aparecen valores no finitos o fuera de rango.

    Raises
    ------
    ValueError
        Si faltan columnas obligatorias o la limpieza deja una muestra vacía.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en SDSS: {missing}")

    clean = df.loc[:, REQUIRED_COLUMNS].copy()
    before = len(clean)

    for column in PHYSICAL_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    finite_mask = np.isfinite(clean[PHYSICAL_COLUMNS]).all(axis=1)
    range_mask = (
        clean["z"].between(0.02, 0.10)
        & clean["metallicity"].between(7.0, 10.0)
        & clean["log_stellar_mass"].between(6.0, 13.0)
    )
    positive_flux_mask = (clean[[col for col in PHYSICAL_COLUMNS if "flux" in col]] > 0).all(axis=1)

    valid = finite_mask & range_mask & positive_flux_mask
    clean = clean.loc[valid].reset_index(drop=True)
    discarded = before - len(clean)

    if discarded > 0:
        print(f"Advertencia: se descartaron {discarded} filas tras validación local.")
    if clean.empty:
        raise ValueError("La validación dejó una muestra vacía.")

    return clean


def main() -> None:
    """Descarga y guarda la muestra SDSS curada para ML/QML."""
    print(
        "Enviando consulta al SDSS "
        f"DR{DATA_RELEASE} para {N_GALAXIES} galaxias "
        f"en lotes de {BATCH_SIZE}, S/N >= {MIN_SNR:.1f}..."
    )

    try:
        df_raw = fetch_sdss_batches()
        df = validate_dataframe(df_raw)
        df = df.sort_values("specobjid").reset_index(drop=True)
        df.to_csv(OUTPUT_CSV, index=False)
    except (RuntimeError, requests.exceptions.RequestException) as exc:
        raise RuntimeError(f"Falló la descarga desde SDSS: {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise RuntimeError(f"La respuesta de SDSS no pudo leerse como CSV: {exc}") from exc
    except OSError as exc:
        raise OSError(f"No se pudo guardar el archivo '{OUTPUT_CSV}': {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"Datos SDSS inválidos: {exc}") from exc

    print(f"Descarga completada: {len(df)} galaxias válidas.")
    print(f"Archivo guardado en: {OUTPUT_CSV}")
    print("Columnas disponibles:")
    print(", ".join(df.columns))


if __name__ == "__main__":
    main()
