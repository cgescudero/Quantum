"""Comparación controlada VQR vs ML clásico PCA para metalicidad SDSS.

Este script implementa la primera comparación QML trazable del proyecto:

- Lee las features físicas generadas por ``preprocess_sdss_features.py``.
- En cada fold ajusta ``StandardScaler`` y ``PCA`` usando solo train.
- Compara un Variational Quantum Regressor (VQR) contra Ridge y Random Forest
  entrenados sobre la misma representación PCA.
- Escala el target del VQR a [-1, 1] dentro de cada fold, porque el observable
  ``PauliZ`` produce valores en ese rango.
- Guarda métricas, predicciones, historia de entrenamiento y configuración.

No pretende todavía ser un barrido exhaustivo. Es una línea base QML prudente
para evaluar estabilidad, coste y cercanía a controles clásicos compactos.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as onp
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "sdss_galaxies_features_physical.csv"
RESULTS_DIR = SCRIPT_DIR / "results_qml_vqr_v1"

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


@dataclass(frozen=True)
class VQRConfig:
    """Configuración de un circuito VQR.

    Attributes
    ----------
    n_qubits:
        Número de qubits y componentes PCA usadas.
    n_layers:
        Número de capas variacionales ``StronglyEntanglingLayers``.
    """

    n_qubits: int
    n_layers: int


class VQRModel:
    """Variational Quantum Regressor con AngleEmbedding y capas entrelazantes.

    Parameters
    ----------
    n_qubits:
        Número de qubits del circuito.
    n_layers:
        Número de capas ``StronglyEntanglingLayers``.
    learning_rate:
        Paso del optimizador Adam de PennyLane.
    batch_size:
        Tamaño de minibatch para estimar el coste.
    epochs:
        Número de épocas de optimización.
    seed:
        Semilla para inicialización y minibatches.
    log_interval:
        Frecuencia con la que se registra la pérdida de entrenamiento.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        seed: int,
        log_interval: int,
    ) -> None:
        if n_qubits <= 0:
            raise ValueError("n_qubits debe ser positivo.")
        if n_layers <= 0:
            raise ValueError("n_layers debe ser positivo.")
        if learning_rate <= 0:
            raise ValueError("learning_rate debe ser positivo.")
        if batch_size <= 0:
            raise ValueError("batch_size debe ser positivo.")
        if epochs <= 0:
            raise ValueError("epochs debe ser positivo.")
        if log_interval <= 0:
            raise ValueError("log_interval debe ser positivo.")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.log_interval = log_interval
        self.rng = onp.random.default_rng(seed)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(params: pnp.ndarray, x: pnp.ndarray) -> pnp.ndarray:
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        initial = self.rng.uniform(0.0, 2.0 * onp.pi, size=(n_layers, n_qubits, 3))
        self.params = pnp.array(initial, requires_grad=True)
        self.history_: list[dict[str, float | int]] = []

    def fit(self, x_train: onp.ndarray, y_train: onp.ndarray) -> None:
        """Entrena el VQR con minibatches.

        Parameters
        ----------
        x_train:
            Features angulares en rango [0, pi], forma ``(n_samples, n_qubits)``.
        y_train:
            Target escalado a [-1, 1], forma ``(n_samples,)``.
        """
        if x_train.ndim != 2 or x_train.shape[1] != self.n_qubits:
            raise ValueError(
                f"x_train debe tener forma (n_samples, {self.n_qubits}); "
                f"recibido {x_train.shape}."
            )
        if len(x_train) != len(y_train):
            raise ValueError("x_train e y_train deben tener la misma longitud.")

        x_pnp = pnp.array(x_train, requires_grad=False)
        y_pnp = pnp.array(y_train, requires_grad=False)
        effective_batch = min(self.batch_size, len(x_train))
        optimizer = qml.AdamOptimizer(stepsize=self.learning_rate)

        for epoch in range(1, self.epochs + 1):
            batch_idx = self.rng.choice(len(x_train), size=effective_batch, replace=False)
            x_batch = x_pnp[batch_idx]
            y_batch = y_pnp[batch_idx]

            def cost(params: pnp.ndarray) -> pnp.ndarray:
                preds = [self.circuit(params, x) for x in x_batch]
                return pnp.mean((pnp.array(preds) - y_batch) ** 2)

            self.params = optimizer.step(cost, self.params)

            if epoch == 1 or epoch % self.log_interval == 0 or epoch == self.epochs:
                batch_loss = float(cost(self.params))
                self.history_.append({"epoch": epoch, "batch_loss": batch_loss})

    def predict_scaled(self, x: onp.ndarray) -> onp.ndarray:
        """Predice en la escala interna [-1, 1].

        Parameters
        ----------
        x:
            Features angulares en rango [0, pi].

        Returns
        -------
        numpy.ndarray
            Predicciones VQR en escala [-1, 1].
        """
        if x.ndim != 2 or x.shape[1] != self.n_qubits:
            raise ValueError(f"x debe tener forma (n_samples, {self.n_qubits}).")

        return onp.array([float(self.circuit(self.params, pnp.array(row))) for row in x])


def parse_configs(config_text: str) -> list[VQRConfig]:
    """Convierte texto tipo ``'4:2,6:2,6:4'`` en configuraciones VQR.

    Parameters
    ----------
    config_text:
        Cadena con pares ``n_qubits:n_layers`` separados por coma.

    Returns
    -------
    list[VQRConfig]
        Configuraciones parseadas.
    """
    configs: list[VQRConfig] = []
    for item in config_text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            n_qubits_text, n_layers_text = stripped.split(":")
            config = VQRConfig(n_qubits=int(n_qubits_text), n_layers=int(n_layers_text))
        except ValueError as exc:
            raise ValueError(
                "Formato inválido en --vqr-configs. Use, por ejemplo, '4:2,6:2,6:4'."
            ) from exc
        if config.n_qubits <= 0 or config.n_layers <= 0:
            raise ValueError("n_qubits y n_layers deben ser positivos.")
        configs.append(config)

    if not configs:
        raise ValueError("Debe especificarse al menos una configuración VQR.")
    return configs


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos.

    Returns
    -------
    argparse.Namespace
        Configuración de ejecución.
    """
    parser = argparse.ArgumentParser(description="VQR vs controles clásicos PCA para SDSS.")
    parser.add_argument("--input", type=Path, default=INPUT_CSV, help="CSV de features físicas.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Directorio de salida.")
    parser.add_argument("--n-galaxies", type=int, default=500, help="Número máximo de galaxias.")
    parser.add_argument("--n-folds", type=int, default=3, help="Número de folds KFold.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument(
        "--vqr-configs",
        type=str,
        default="4:2,6:2",
        help="Configuraciones n_qubits:n_layers separadas por coma. Ej: '4:2,6:2,6:4'.",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Épocas de entrenamiento VQR.")
    parser.add_argument("--batch-size", type=int, default=16, help="Minibatch VQR.")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Learning rate Adam VQR.")
    parser.add_argument("--log-interval", type=int, default=10, help="Intervalo de logging de pérdida.")
    return parser.parse_args()


def load_dataset(input_csv: Path, n_galaxies: int, seed: int) -> pd.DataFrame:
    """Carga y valida el dataset de features físicas.

    Parameters
    ----------
    input_csv:
        Ruta al CSV de features físicas.
    n_galaxies:
        Número máximo de objetos a usar.
    seed:
        Semilla de submuestreo.

    Returns
    -------
    pandas.DataFrame
        Tabla validada.
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

    finite_mask = onp.isfinite(data[numeric_columns]).all(axis=1)
    discarded = int((~finite_mask).sum())
    if discarded > 0:
        print(f"Advertencia: se descartan {discarded} filas no finitas.")
    data = data.loc[finite_mask].reset_index(drop=True)

    if data.empty:
        raise ValueError("El dataset validado quedó vacío.")
    if len(data) > n_galaxies:
        data = data.sample(n=n_galaxies, random_state=seed).sort_index().reset_index(drop=True)

    return data


def compute_metrics(y_true: onp.ndarray, y_pred: onp.ndarray) -> dict[str, float]:
    """Calcula métricas de regresión en escala física.

    Parameters
    ----------
    y_true:
        Metalicidad real.
    y_pred:
        Metalicidad predicha.

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
        "rmse": float(onp.sqrt(mse)),
    }


def prepare_fold_features(
    x_train: onp.ndarray,
    x_test: onp.ndarray,
    n_components: int,
    seed: int,
) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray, onp.ndarray]:
    """Ajusta scaler/PCA en train y devuelve features PCA y angulares.

    Parameters
    ----------
    x_train:
        Features físicas de entrenamiento.
    x_test:
        Features físicas de test.
    n_components:
        Número de componentes PCA, igual a ``n_qubits``.
    seed:
        Semilla PCA.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``x_train_pca``, ``x_test_pca``, ``x_train_angles``, ``x_test_angles``.
    """
    if n_components > x_train.shape[1]:
        raise ValueError(
            f"n_components={n_components} excede el número de features={x_train.shape[1]}."
        )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    pca = PCA(n_components=n_components, random_state=seed)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    angle_scaler = MinMaxScaler(feature_range=(0.0, onp.pi))
    x_train_angles = angle_scaler.fit_transform(x_train_pca)
    x_test_angles = angle_scaler.transform(x_test_pca)
    x_test_angles = onp.clip(x_test_angles, 0.0, onp.pi)

    return x_train_pca, x_test_pca, x_train_angles, x_test_angles


def evaluate_classical_controls(
    x_train_pca: onp.ndarray,
    x_test_pca: onp.ndarray,
    y_train: onp.ndarray,
    y_test: onp.ndarray,
    fold: int,
    config: VQRConfig,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, onp.ndarray]]:
    """Entrena Ridge y Random Forest sobre la misma PCA que el VQR.

    Parameters
    ----------
    x_train_pca, x_test_pca:
        Representación PCA ajustada solo con train.
    y_train, y_test:
        Target físico.
    fold:
        Índice de fold.
    config:
        Configuración VQR que define la dimensionalidad PCA.
    seed:
        Semilla de modelos clásicos.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, numpy.ndarray]]
        Métricas y predicciones por modelo.
    """
    models: dict[str, Any] = {
        f"ridge_pca{config.n_qubits}": Ridge(alpha=1.0),
        f"random_forest_pca{config.n_qubits}": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    }
    metric_rows: list[dict[str, Any]] = []
    predictions: dict[str, onp.ndarray] = {}

    for model_name, model in models.items():
        start = time.perf_counter()
        model.fit(x_train_pca, y_train)
        y_pred = onp.asarray(model.predict(x_test_pca), dtype=float)
        elapsed = time.perf_counter() - start
        metrics = compute_metrics(y_test, y_pred)
        metric_rows.append(
            {
                "fold": fold,
                "model": model_name,
                "model_family": "classical_control",
                "n_qubits": config.n_qubits,
                "n_layers": 0,
                "fit_predict_time_sec": elapsed,
                **metrics,
            }
        )
        predictions[model_name] = y_pred

    return metric_rows, predictions


def evaluate_vqr(
    x_train_angles: onp.ndarray,
    x_test_angles: onp.ndarray,
    y_train: onp.ndarray,
    y_test: onp.ndarray,
    fold: int,
    config: VQRConfig,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], onp.ndarray, pd.DataFrame]:
    """Entrena y evalúa un VQR para un fold/configuración.

    Parameters
    ----------
    x_train_angles, x_test_angles:
        Features PCA escaladas a [0, pi].
    y_train, y_test:
        Target físico.
    fold:
        Índice de fold.
    config:
        Configuración VQR.
    args:
        Argumentos globales.

    Returns
    -------
    tuple[dict[str, Any], numpy.ndarray, pandas.DataFrame]
        Métricas, predicciones físicas e historia de entrenamiento.
    """
    y_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    model_name = f"vqr_q{config.n_qubits}_l{config.n_layers}"
    model_seed = args.seed + 1000 * fold + 100 * config.n_qubits + config.n_layers
    model = VQRModel(
        n_qubits=config.n_qubits,
        n_layers=config.n_layers,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=model_seed,
        log_interval=args.log_interval,
    )

    start = time.perf_counter()
    model.fit(x_train_angles, y_train_scaled)
    y_pred_scaled = model.predict_scaled(x_test_angles)
    y_pred_scaled = onp.clip(y_pred_scaled, -1.0, 1.0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    elapsed = time.perf_counter() - start

    metrics = compute_metrics(y_test, y_pred)
    metric_row = {
        "fold": fold,
        "model": model_name,
        "model_family": "vqr",
        "n_qubits": config.n_qubits,
        "n_layers": config.n_layers,
        "fit_predict_time_sec": elapsed,
        **metrics,
    }

    history = pd.DataFrame(model.history_)
    history["fold"] = fold
    history["model"] = model_name
    history["n_qubits"] = config.n_qubits
    history["n_layers"] = config.n_layers

    return metric_row, y_pred, history


def run_experiment(
    data: pd.DataFrame,
    configs: list[VQRConfig],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ejecuta la comparación VQR vs controles PCA.

    Parameters
    ----------
    data:
        Dataset validado.
    configs:
        Configuraciones VQR.
    args:
        Argumentos globales.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Métricas, predicciones e historia de entrenamiento.
    """
    if args.n_folds < 2:
        raise ValueError("n_folds debe ser >= 2.")
    if len(data) < args.n_folds:
        raise ValueError("La muestra debe tener al menos n_folds objetos.")

    x_all = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_all = data[TARGET_COLUMN].to_numpy(dtype=float)
    ids = data[ID_COLUMNS].reset_index(drop=True)

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []

    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(x_all), start=1):
        print(f"\n--- Fold {fold}/{args.n_folds} ---")
        x_train, x_test = x_all[train_idx], x_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        for config in configs:
            print(f"Configuración: n_qubits={config.n_qubits}, n_layers={config.n_layers}")
            x_train_pca, x_test_pca, x_train_angles, x_test_angles = prepare_fold_features(
                x_train=x_train,
                x_test=x_test,
                n_components=config.n_qubits,
                seed=args.seed,
            )

            classical_metrics, classical_predictions = evaluate_classical_controls(
                x_train_pca=x_train_pca,
                x_test_pca=x_test_pca,
                y_train=y_train,
                y_test=y_test,
                fold=fold,
                config=config,
                seed=args.seed,
            )
            metrics_rows.extend(classical_metrics)

            for model_name, y_pred in classical_predictions.items():
                pred_frame = ids.iloc[test_idx].copy()
                pred_frame["fold"] = fold
                pred_frame["model"] = model_name
                pred_frame["model_family"] = "classical_control"
                pred_frame["n_qubits"] = config.n_qubits
                pred_frame["n_layers"] = 0
                pred_frame["y_true"] = y_test
                pred_frame["y_pred"] = y_pred
                pred_frame["residual"] = y_pred - y_test
                prediction_frames.append(pred_frame)

            vqr_metric, vqr_pred, vqr_history = evaluate_vqr(
                x_train_angles=x_train_angles,
                x_test_angles=x_test_angles,
                y_train=y_train,
                y_test=y_test,
                fold=fold,
                config=config,
                args=args,
            )
            metrics_rows.append(vqr_metric)
            history_frames.append(vqr_history)

            pred_frame = ids.iloc[test_idx].copy()
            pred_frame["fold"] = fold
            pred_frame["model"] = vqr_metric["model"]
            pred_frame["model_family"] = "vqr"
            pred_frame["n_qubits"] = config.n_qubits
            pred_frame["n_layers"] = config.n_layers
            pred_frame["y_true"] = y_test
            pred_frame["y_pred"] = vqr_pred
            pred_frame["residual"] = vqr_pred - y_test
            prediction_frames.append(pred_frame)

            print(
                f"  {vqr_metric['model']:18s} "
                f"R2={vqr_metric['r2']:+.4f} "
                f"MAE={vqr_metric['mae']:.4f} "
                f"RMSE={vqr_metric['rmse']:.4f} "
                f"t={vqr_metric['fit_predict_time_sec']:.1f}s"
            )

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    return metrics, predictions, history


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Resume métricas por modelo.

    Parameters
    ----------
    metrics:
        Métricas por fold.

    Returns
    -------
    pandas.DataFrame
        Tabla resumen ordenada por RMSE medio.
    """
    return (
        metrics.groupby(["model", "model_family", "n_qubits", "n_layers"], as_index=False)
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


def save_outputs(
    output_dir: Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Guarda resultados de la corrida.

    Parameters
    ----------
    output_dir:
        Carpeta de salida.
    metrics:
        Métricas por fold.
    predictions:
        Predicciones por objeto/fold/modelo.
    history:
        Historia de entrenamiento VQR.
    summary:
        Resumen agregado por modelo.
    config:
        Configuración reproducible.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    predictions.to_csv(output_dir / "predictions_by_fold.csv", index=False)
    history.to_csv(output_dir / "training_history.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Ejecuta la comparación VQR vs controles clásicos PCA."""
    args = parse_args()
    configs = parse_configs(args.vqr_configs)
    start_total = time.perf_counter()

    data = load_dataset(args.input, args.n_galaxies, args.seed)
    print(f"Muestra cargada: {len(data)} galaxias")
    print(f"Features: {', '.join(FEATURE_COLUMNS)}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Configuraciones VQR: {configs}")

    metrics, predictions, history = run_experiment(data, configs, args)
    summary = summarize_metrics(metrics)

    run_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "input_csv": str(args.input.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "n_galaxies": len(data),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "vqr_configs": [config.__dict__ for config in configs],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "log_interval": args.log_interval,
        "classical_controls": ["Ridge", "RandomForestRegressor"],
        "qml_embedding": "AngleEmbedding(rotation='Y')",
        "qml_ansatz": "StronglyEntanglingLayers",
        "qml_observable": "PauliZ(0)",
        "notes": "VQR baseline with fold-local StandardScaler, PCA, angle scaling and target scaling.",
    }

    save_outputs(args.output_dir, metrics, predictions, history, summary, run_config)

    print("\n" + "=" * 90)
    print("RESUMEN VQR VS CONTROLES PCA")
    print("=" * 90)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("=" * 90)
    print(f"Tiempo total: {(time.perf_counter() - start_total) / 60:.2f} min")
    print(f"Resultados guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()
