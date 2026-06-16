"""Baseline clásico reproducible para regresión de metalicidad SDSS.

Primera etapa de la comparación ML/QML: establece benchmarks clásicos sobre
features físicas nebulares. El VQR/QNN se añadirá en una versión posterior, una
vez cuantificado el nivel de referencia clásico y el efecto de comprimir con PCA.

El script evita leakage: scaler y PCA se ajustan dentro de cada fold usando solo
el subconjunto de entrenamiento.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "sdss_galaxies_features_physical.csv"
RESULTS_DIR = SCRIPT_DIR / "results_ml_qml_v1"

RANDOM_STATE = 42
DEFAULT_N_GALAXIES = 5_000
DEFAULT_N_FOLDS = 5
PCA_COMPONENTS = (4, 6, 8)

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
TARGET_COLUMN = "metallicity"
ID_COLUMNS = ["specobjid", "ra", "dec"]


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos.

    Returns
    -------
    argparse.Namespace
        Configuración de ejecución.
    """
    parser = argparse.ArgumentParser(
        description="Benchmarks clásicos para regresión de metalicidad SDSS."
    )
    parser.add_argument("--input", type=Path, default=INPUT_CSV, help="CSV de features físicas.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Directorio de resultados.")
    parser.add_argument("--n-galaxies", type=int, default=DEFAULT_N_GALAXIES, help="Número máximo de galaxias.")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS, help="Número de folds KFold.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Semilla aleatoria.")
    parser.add_argument(
        "--skip-svr",
        action="store_true",
        help="Omite SVR si se quiere una corrida más rápida.",
    )
    parser.add_argument(
        "--skip-mlp",
        action="store_true",
        help="Omite MLP si se quiere una corrida más rápida.",
    )
    return parser.parse_args()


def load_dataset(input_csv: Path, n_galaxies: int, seed: int) -> pd.DataFrame:
    """Carga y valida la tabla de features físicas.

    Parameters
    ----------
    input_csv:
        Ruta a ``sdss_galaxies_features_physical.csv``.
    n_galaxies:
        Número máximo de objetos a usar. Si es menor que el total, se toma una
        muestra aleatoria reproducible.
    seed:
        Semilla para muestreo reproducible.

    Returns
    -------
    pandas.DataFrame
        Tabla validada y, si corresponde, submuestreada.

    Raises
    ------
    FileNotFoundError
        Si no existe el archivo de entrada.
    ValueError
        Si faltan columnas o quedan valores no finitos.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_csv}")
    if n_galaxies <= 0:
        raise ValueError("n_galaxies debe ser positivo.")

    data = pd.read_csv(input_csv)
    required = set(ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    data = data.loc[:, ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    numeric_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    finite_mask = np.isfinite(data[numeric_columns]).all(axis=1)
    discarded = int((~finite_mask).sum())
    if discarded > 0:
        print(f"Advertencia: se descartan {discarded} filas no finitas.")
    data = data.loc[finite_mask].reset_index(drop=True)

    if data.empty:
        raise ValueError("La tabla validada quedó vacía.")
    if len(data) > n_galaxies:
        data = data.sample(n=n_galaxies, random_state=seed).sort_index().reset_index(drop=True)

    return data


def add_optional_xgboost(models: dict[str, RegressorMixin], seed: int) -> None:
    """Añade XGBoost si está instalado; si no, usa HistGradientBoosting.

    Parameters
    ----------
    models:
        Diccionario de modelos a modificar in-place.
    seed:
        Semilla reproducible.
    """
    try:
        from xgboost import XGBRegressor  # type: ignore

        models["xgboost_full"] = XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
    except ImportError:
        models["hist_gradient_boosting_full"] = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.03,
            max_leaf_nodes=31,
            random_state=seed,
        )


def build_full_feature_models(seed: int, skip_svr: bool, skip_mlp: bool) -> dict[str, Any]:
    """Construye modelos clásicos sobre el espacio completo de features.

    Parameters
    ----------
    seed:
        Semilla reproducible.
    skip_svr:
        Si ``True``, omite SVR.
    skip_mlp:
        Si ``True``, omite MLP.

    Returns
    -------
    dict[str, Any]
        Modelos sklearn listos para ``fit`` y ``predict``.
    """
    models: dict[str, Any] = {
        "dummy_mean_full": DummyRegressor(strategy="mean"),
        "ridge_full": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "elasticnet_full": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=20_000, random_state=seed)),
            ]
        ),
        "random_forest_full": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    if not skip_svr:
        models["svr_rbf_full"] = TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="rbf", C=10.0, epsilon=0.02, gamma="scale")),
                ]
            ),
            transformer=StandardScaler(),
        )

    if not skip_mlp:
        models["mlp_full"] = TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            solver="adam",
                            alpha=1e-4,
                            learning_rate_init=1e-3,
                            max_iter=800,
                            early_stopping=True,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            transformer=StandardScaler(),
        )

    add_optional_xgboost(models, seed)
    return models


def build_pca_control_models(seed: int) -> dict[str, Any]:
    """Construye modelos clásicos con PCA para comparación controlada con QML.

    Parameters
    ----------
    seed:
        Semilla reproducible.

    Returns
    -------
    dict[str, Any]
        Modelos clásicos en espacios PCA de dimensión 4, 6 y 8.
    """
    models: dict[str, Any] = {}
    for n_components in PCA_COMPONENTS:
        suffix = f"pca{n_components}"
        models[f"ridge_{suffix}"] = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=seed)),
                ("model", Ridge(alpha=1.0)),
            ]
        )
        models[f"random_forest_{suffix}"] = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=seed)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return models


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula métricas de regresión en unidades físicas de metalicidad.

    Parameters
    ----------
    y_true:
        Metalicidades reales.
    y_pred:
        Metalicidades predichas.

    Returns
    -------
    dict[str, float]
        R2, MAE, MSE y RMSE.
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
    }


def run_cross_validation(
    data: pd.DataFrame,
    models: dict[str, Any],
    n_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta validación cruzada y guarda métricas/predicciones por fold.

    Parameters
    ----------
    data:
        Tabla de features físicas.
    models:
        Diccionario de modelos sklearn.
    n_folds:
        Número de folds.
    seed:
        Semilla del KFold.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Métricas por fold y predicciones por objeto/modelo.
    """
    if n_folds < 2:
        raise ValueError("n_folds debe ser >= 2.")
    if len(data) < n_folds:
        raise ValueError("La muestra debe tener al menos n_folds objetos.")

    x_all = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_all = data[TARGET_COLUMN].to_numpy(dtype=float)
    ids = data[ID_COLUMNS].reset_index(drop=True)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(x_all), start=1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        x_train, x_test = x_all[train_idx], x_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        for model_name, model in models.items():
            start = time.perf_counter()
            model.fit(x_train, y_train)
            y_pred = np.asarray(model.predict(x_test), dtype=float)
            elapsed = time.perf_counter() - start

            fold_metrics = compute_metrics(y_test, y_pred)
            metrics_rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "fit_predict_time_sec": elapsed,
                    **fold_metrics,
                }
            )

            fold_predictions = ids.iloc[test_idx].copy()
            fold_predictions["fold"] = fold
            fold_predictions["model"] = model_name
            fold_predictions["y_true"] = y_test
            fold_predictions["y_pred"] = y_pred
            fold_predictions["residual"] = y_pred - y_test
            prediction_rows.append(fold_predictions)

            print(
                f"{model_name:28s} "
                f"R2={fold_metrics['r2']:+.4f} "
                f"MAE={fold_metrics['mae']:.4f} "
                f"RMSE={fold_metrics['rmse']:.4f} "
                f"t={elapsed:.2f}s"
            )

    return pd.DataFrame(metrics_rows), pd.concat(prediction_rows, ignore_index=True)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Resume métricas por modelo.

    Parameters
    ----------
    metrics:
        Métricas por fold.

    Returns
    -------
    pandas.DataFrame
        Media y desviación estándar de cada métrica por modelo.
    """
    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            time_mean_sec=("fit_predict_time_sec", "mean"),
            time_std_sec=("fit_predict_time_sec", "std"),
        )
        .sort_values("rmse_mean")
        .reset_index(drop=True)
    )
    return summary


def save_outputs(
    output_dir: Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Guarda resultados y configuración de corrida.

    Parameters
    ----------
    output_dir:
        Directorio de salida.
    metrics:
        Métricas por fold.
    predictions:
        Predicciones por objeto/fold/modelo.
    summary:
        Resumen por modelo.
    config:
        Configuración reproducible de la corrida.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    predictions.to_csv(output_dir / "predictions_by_fold.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Ejecuta benchmarks clásicos completos y con PCA controlado."""
    args = parse_args()
    start_total = time.perf_counter()

    data = load_dataset(args.input, args.n_galaxies, args.seed)
    print(f"Muestra cargada: {len(data)} galaxias")
    print(f"Features: {', '.join(FEATURE_COLUMNS)}")
    print(f"Target: {TARGET_COLUMN}")

    models = build_full_feature_models(args.seed, args.skip_svr, args.skip_mlp)
    models.update(build_pca_control_models(args.seed))

    print(f"Modelos a evaluar: {', '.join(models.keys())}")
    metrics, predictions = run_cross_validation(data, models, args.n_folds, args.seed)
    summary = summarize_metrics(metrics)

    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "input_csv": str(args.input.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "n_galaxies": len(data),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "pca_components": list(PCA_COMPONENTS),
        "models": list(models.keys()),
        "skip_svr": args.skip_svr,
        "skip_mlp": args.skip_mlp,
        "notes": "Classical baseline only; QML models will be added in a later version.",
    }
    save_outputs(args.output_dir, metrics, predictions, summary, config)

    print("\n" + "=" * 80)
    print("RESUMEN ORDENADO POR RMSE")
    print("=" * 80)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("=" * 80)
    print(f"Tiempo total: {(time.perf_counter() - start_total) / 60:.2f} min")
    print(f"Resultados guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()
