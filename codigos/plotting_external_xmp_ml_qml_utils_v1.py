"""Utilidades comunes para figuras de la Parte 3 ML/QML.

Todas las funciones asumen rutas dentro del directorio ``Parte_3`` y validan
entradas antes de graficar. La métrica primaria del proyecto es PR-AUC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FIGURE_DIR = SCRIPT_DIR / "figures_external_xmp_ml_qml_v1"

COLORS = {
    "classical": "#4C78A8",
    "classical_alt": "#7F7F7F",
    "quantum": "#F58518",
    "quantum_alt": "#7B3F98",
    "vqc": "#D62728",
    "positive": "#F58518",
    "negative": "#9E9E9E",
    "matched": "#4C78A8",
    "random": "#E45756",
    "external": "#54A24B",
}


def parse_common_args(description: str) -> argparse.Namespace:
    """Parsea argumentos comunes para scripts de figuras.

    Parameters
    ----------
    description:
        Descripción corta del script.

    Returns
    -------
    argparse.Namespace
        Argumentos con ``output_dir`` y ``overwrite``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_output_dir(output_dir: Path) -> Path:
    """Crea y valida que el directorio de salida esté dentro de Parte_3."""
    resolved = output_dir.expanduser().resolve()
    root = SCRIPT_DIR.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"--output-dir debe quedar dentro de Parte_3: {root}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def require_file(path: Path) -> Path:
    """Devuelve una ruta existente o falla con mensaje explícito."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")
    if not path.is_file():
        raise ValueError(f"La ruta no es un archivo regular: {path}")
    return path


def read_csv_checked(path: Path, required_columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Lee un CSV y valida tamaño y columnas requeridas.

    Parameters
    ----------
    path:
        Archivo CSV.
    required_columns:
        Columnas obligatorias.

    Returns
    -------
    pandas.DataFrame
        Tabla leída.
    """
    require_file(path)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV vacío: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSV mal formado: {path}") from exc
    if df.empty:
        raise ValueError(f"CSV sin filas: {path}")
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en {path}: {missing}. Columnas disponibles: {list(df.columns)}")
    return df


def read_json_checked(path: Path) -> dict:
    """Lee JSON y devuelve un diccionario validado."""
    require_file(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON mal formado: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"JSON debe contener un objeto raíz: {path}")
    return data


def first_existing(paths: Iterable[Path]) -> Path:
    """Devuelve el primer archivo existente entre varias opciones."""
    candidates = list(paths)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No existe ninguno de estos archivos: " + ", ".join(str(p) for p in candidates))


def find_column(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    """Detecta la primera columna existente entre candidatos."""
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No se encontró columna para {label}. Candidatas: {list(candidates)}; disponibles: {list(df.columns)}")


def numeric_series(df: pd.DataFrame, column: str, context: str) -> pd.Series:
    """Convierte una columna a numérica y valida valores finitos disponibles."""
    values = pd.to_numeric(df[column], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float))
    if finite.sum() == 0:
        raise ValueError(f"Columna {column} sin valores finitos en {context}")
    dropped = len(values) - int(finite.sum())
    if dropped > 0:
        print(f"[{context}] descartados {dropped} valores no finitos en {column}")
    return values[finite]


def require_numeric_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> pd.DataFrame:
    """Valida columnas numéricas y descarta filas con NaN/inf, informando pérdida."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            raise ValueError(f"Falta columna numérica {col} en {context}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    mask = np.ones(len(out), dtype=bool)
    for col in columns:
        mask &= np.isfinite(out[col].to_numpy(dtype=float))
    dropped = len(out) - int(mask.sum())
    if dropped > 0:
        print(f"[{context}] descartadas {dropped}/{len(out)} filas por NaN/inf en {list(columns)}")
    out = out.loc[mask].copy()
    if out.empty:
        raise ValueError(f"No quedan filas válidas en {context} tras limpieza numérica")
    return out


def metric_column(df: pd.DataFrame, mean_name: str, bare_name: str) -> str:
    """Detecta columna de métrica, prefiriendo la versión ``*_mean``."""
    if mean_name in df.columns:
        return mean_name
    if bare_name in df.columns:
        return bare_name
    raise ValueError(f"No se encontró métrica {mean_name} ni {bare_name}. Columnas: {list(df.columns)}")


def model_display_name(model: str) -> str:
    """Normaliza nombres de modelos para etiquetas cortas de figura."""
    replacements = {
        "hist_gradient_boosting": "HistGradientBoosting",
        "logistic_regression": "Logistic regression",
        "random_forest": "Random forest",
        "svc_poly": "SVC poly",
        "svc_rbf": "SVC RBF",
        "qksvc_zz": "QK-SVC ZZ",
    }
    if model in replacements:
        return replacements[model]
    if model.startswith("qksvc_reuploading"):
        return "QK-SVC reuploading"
    return model.replace("_", " ")


def model_base_family(model: str) -> str:
    """Agrupa semillas/configuraciones QML por familia base."""
    if model.startswith("qksvc_reuploading"):
        return "qksvc_reuploading"
    if model.startswith("qksvc_zz"):
        return "qksvc_zz"
    return model


def setup_matplotlib() -> None:
    """Configura estilo sobrio y reproducible para figuras científicas."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, overwrite: bool) -> list[Path]:
    """Guarda figura en PNG y PDF, respetando ``--overwrite``."""
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Ya existe {path}. Usa --overwrite para reemplazar.")
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def write_table(df: pd.DataFrame, output_dir: Path, filename: str, overwrite: bool) -> Path:
    """Escribe CSV tabular asociado a una figura."""
    path = output_dir / filename
    if path.exists() and not overwrite:
        raise FileExistsError(f"Ya existe {path}. Usa --overwrite para reemplazar.")
    df.to_csv(path, index=False)
    return path


def print_outputs(paths: Sequence[Path]) -> None:
    """Imprime rutas generadas de forma consistente."""
    for path in paths:
        print(f"Generado: {path}")
