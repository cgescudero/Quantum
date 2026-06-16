"""Optimización controlada de VQR para regresión de metalicidad SDSS.

Este script explora configuraciones de Variational Quantum Regressors (VQR)
manteniendo el mismo protocolo físico/metodológico usado en
``compare_qml_vqr_metallicity_v1.py``:

- features físicas nebulares desde ``sdss_galaxies_features_physical.csv``;
- escalado, PCA y escalado angular ajustados solo con train en cada fold;
- target escalado a [-1, 1] dentro de cada fold;
- métricas finales en unidades físicas de 12 + log(O/H).

La diferencia respecto al script anterior es que aquí se optimiza sobre una
grilla de hiperparámetros VQR: número de qubits, capas, learning rates,
rotaciones de embedding y semillas de inicialización. También se guarda la
mejor configuración según RMSE medio.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as onp
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "sdss_galaxies_features_physical.csv"
RESULTS_DIR = SCRIPT_DIR / "results_vqr_optimization_v1"

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
class VQRHyperParams:
    """Hiperparámetros de una corrida VQR.

    Attributes
    ----------
    n_qubits:
        Número de qubits y componentes PCA.
    n_layers:
        Número de capas variacionales.
    learning_rate:
        Learning rate del optimizador Adam.
    rotation:
        Eje de rotación para ``AngleEmbedding``: ``X``, ``Y`` o ``Z``.
    init_seed:
        Semilla de inicialización y minibatches del VQR.
    """

    n_qubits: int
    n_layers: int
    learning_rate: float
    rotation: str
    init_seed: int


class OptimizedVQR:
    """VQR con AngleEmbedding configurable y StronglyEntanglingLayers.

    Parameters
    ----------
    params:
        Hiperparámetros VQR.
    batch_size:
        Tamaño de minibatch.
    epochs:
        Número máximo de épocas.
    log_interval:
        Frecuencia para registrar pérdidas.
    patience:
        Paciencia de early stopping basada en pérdida full-train aproximada.
        Si es 0, se desactiva.
    min_delta:
        Mejora mínima requerida para reiniciar paciencia.
    full_loss_interval:
        Cada cuántas épocas se evalúa pérdida full-train.
    """

    def __init__(
        self,
        params: VQRHyperParams,
        batch_size: int,
        epochs: int,
        log_interval: int,
        patience: int,
        min_delta: float,
        full_loss_interval: int,
    ) -> None:
        if params.n_qubits <= 0:
            raise ValueError("n_qubits debe ser positivo.")
        if params.n_layers <= 0:
            raise ValueError("n_layers debe ser positivo.")
        if params.learning_rate <= 0:
            raise ValueError("learning_rate debe ser positivo.")
        if params.rotation not in {"X", "Y", "Z"}:
            raise ValueError("rotation debe ser 'X', 'Y' o 'Z'.")
        if batch_size <= 0:
            raise ValueError("batch_size debe ser positivo.")
        if epochs <= 0:
            raise ValueError("epochs debe ser positivo.")
        if log_interval <= 0:
            raise ValueError("log_interval debe ser positivo.")
        if full_loss_interval <= 0:
            raise ValueError("full_loss_interval debe ser positivo.")

        self.params_config = params
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_interval = log_interval
        self.patience = patience
        self.min_delta = min_delta
        self.full_loss_interval = full_loss_interval
        self.rng = onp.random.default_rng(params.init_seed)
        self.history_: list[dict[str, float | int | str | bool | None]] = []
        self.stop_epoch_: int | None = None

        dev = qml.device("default.qubit", wires=params.n_qubits)

        @qml.qnode(dev)
        def circuit(weights: pnp.ndarray, x: pnp.ndarray) -> pnp.ndarray:
            qml.AngleEmbedding(x, wires=range(params.n_qubits), rotation=params.rotation)
            qml.StronglyEntanglingLayers(weights, wires=range(params.n_qubits))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        initial = self.rng.uniform(
            low=0.0,
            high=2.0 * onp.pi,
            size=(params.n_layers, params.n_qubits, 3),
        )
        self.weights = pnp.array(initial, requires_grad=True)

    def _predict_scaled_pnp(self, weights: pnp.ndarray, x_data: pnp.ndarray) -> pnp.ndarray:
        """Predice en escala [-1, 1] usando arrays PennyLane."""
        return pnp.array([self.circuit(weights, row) for row in x_data])

    def fit(self, x_train: onp.ndarray, y_train_scaled: onp.ndarray) -> None:
        """Entrena el VQR.

        Parameters
        ----------
        x_train:
            Features angulares en [0, pi], forma ``(n_samples, n_qubits)``.
        y_train_scaled:
            Target escalado a [-1, 1].
        """
        if x_train.ndim != 2 or x_train.shape[1] != self.params_config.n_qubits:
            raise ValueError(
                f"x_train debe tener forma (n_samples, {self.params_config.n_qubits})."
            )
        if len(x_train) != len(y_train_scaled):
            raise ValueError("x_train e y_train_scaled deben tener la misma longitud.")

        x_pnp = pnp.array(x_train, requires_grad=False)
        y_pnp = pnp.array(y_train_scaled, requires_grad=False)
        effective_batch = min(self.batch_size, len(x_train))
        optimizer = qml.AdamOptimizer(stepsize=self.params_config.learning_rate)

        best_full_loss = onp.inf
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            batch_idx = self.rng.choice(len(x_train), size=effective_batch, replace=False)
            x_batch = x_pnp[batch_idx]
            y_batch = y_pnp[batch_idx]

            def cost(weights: pnp.ndarray) -> pnp.ndarray:
                preds = self._predict_scaled_pnp(weights, x_batch)
                return pnp.mean((preds - y_batch) ** 2)

            self.weights = optimizer.step(cost, self.weights)

            should_log = epoch == 1 or epoch % self.log_interval == 0 or epoch == self.epochs
            should_full = epoch == 1 or epoch % self.full_loss_interval == 0 or epoch == self.epochs

            batch_loss = None
            full_loss = None
            if should_log:
                batch_loss = float(cost(self.weights))
            if should_full:
                full_preds = self._predict_scaled_pnp(self.weights, x_pnp)
                full_loss = float(pnp.mean((full_preds - y_pnp) ** 2))

                if full_loss < best_full_loss - self.min_delta:
                    best_full_loss = full_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += self.full_loss_interval

            early_stopped = False
            if self.patience > 0 and epochs_without_improvement >= self.patience:
                self.stop_epoch_ = epoch
                early_stopped = True

            if should_log or should_full or early_stopped:
                self.history_.append(
                    {
                        "epoch": epoch,
                        "batch_loss": batch_loss,
                        "full_train_loss": full_loss,
                        "best_full_train_loss": None if not onp.isfinite(best_full_loss) else best_full_loss,
                        "early_stopped": early_stopped,
                    }
                )

            if early_stopped:
                break

    def predict_scaled(self, x_data: onp.ndarray) -> onp.ndarray:
        """Predice en escala interna [-1, 1].

        Parameters
        ----------
        x_data:
            Features angulares en [0, pi].

        Returns
        -------
        numpy.ndarray
            Predicciones escaladas.
        """
        if x_data.ndim != 2 or x_data.shape[1] != self.params_config.n_qubits:
            raise ValueError(f"x_data debe tener {self.params_config.n_qubits} columnas.")
        x_pnp = pnp.array(x_data, requires_grad=False)
        return onp.asarray(self._predict_scaled_pnp(self.weights, x_pnp), dtype=float)


def parse_int_list(text: str) -> list[int]:
    """Parsea una lista de enteros separados por coma."""
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("La lista de enteros no puede estar vacía.")
    return values


def parse_float_list(text: str) -> list[float]:
    """Parsea una lista de floats separados por coma."""
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("La lista de floats no puede estar vacía.")
    return values


def parse_rotation_list(text: str) -> list[str]:
    """Parsea una lista de rotaciones X/Y/Z."""
    values = [item.strip().upper() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("La lista de rotaciones no puede estar vacía.")
    invalid = sorted(set(values) - {"X", "Y", "Z"})
    if invalid:
        raise ValueError(f"Rotaciones inválidas: {invalid}")
    return values


def build_hyperparameter_grid(
    n_qubits: Iterable[int],
    n_layers: Iterable[int],
    learning_rates: Iterable[float],
    rotations: Iterable[str],
    init_seeds: Iterable[int],
) -> list[VQRHyperParams]:
    """Construye la grilla cartesiana de hiperparámetros VQR."""
    grid = [
        VQRHyperParams(
            n_qubits=q,
            n_layers=l,
            learning_rate=lr,
            rotation=rot,
            init_seed=seed,
        )
        for q, l, lr, rot, seed in itertools.product(
            n_qubits, n_layers, learning_rates, rotations, init_seeds
        )
    ]
    if not grid:
        raise ValueError("La grilla de hiperparámetros quedó vacía.")
    return grid


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Optimización VQR para metalicidad SDSS.")
    parser.add_argument("--input", type=Path, default=INPUT_CSV, help="CSV de features físicas.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Directorio de salida.")
    parser.add_argument("--n-galaxies", type=int, default=500, help="Número máximo de galaxias.")
    parser.add_argument("--n-folds", type=int, default=3, help="Número de folds.")
    parser.add_argument("--split-seed", type=int, default=42, help="Semilla del muestreo y KFold.")
    parser.add_argument("--n-qubits", type=str, default="6", help="Lista de qubits. Ej: '4,6'.")
    parser.add_argument("--n-layers", type=str, default="4,6", help="Lista de capas. Ej: '2,4,6'.")
    parser.add_argument(
        "--learning-rates",
        type=str,
        default="0.01,0.03",
        help="Lista de learning rates. Ej: '0.01,0.03'.",
    )
    parser.add_argument(
        "--rotations",
        type=str,
        default="Y",
        help="Rotaciones de AngleEmbedding. Ej: 'X,Y,Z'.",
    )
    parser.add_argument(
        "--init-seeds",
        type=str,
        default="101,202,303",
        help="Semillas de inicialización VQR. Ej: '101,202,303'.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Épocas máximas.")
    parser.add_argument("--batch-size", type=int, default=16, help="Tamaño de minibatch.")
    parser.add_argument("--log-interval", type=int, default=20, help="Intervalo de logging.")
    parser.add_argument(
        "--full-loss-interval",
        type=int,
        default=20,
        help="Intervalo para evaluar pérdida full-train.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Paciencia de early stopping en épocas. 0 lo desactiva.",
    )
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Mejora mínima early stopping.")
    return parser.parse_args()


def load_dataset(input_csv: Path, n_galaxies: int, seed: int) -> pd.DataFrame:
    """Carga y valida el dataset de features físicas.

    Parameters
    ----------
    input_csv:
        Ruta al CSV de features físicas.
    n_galaxies:
        Número máximo de objetos.
    seed:
        Semilla para submuestreo reproducible.

    Returns
    -------
    pandas.DataFrame
        Dataset validado.
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


def prepare_fold_features(
    x_train: onp.ndarray,
    x_test: onp.ndarray,
    n_components: int,
    split_seed: int,
) -> tuple[onp.ndarray, onp.ndarray]:
    """Escala features, ajusta PCA en train y transforma a ángulos [0, pi]."""
    if n_components > x_train.shape[1]:
        raise ValueError(
            f"n_components={n_components} excede número de features={x_train.shape[1]}."
        )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    pca = PCA(n_components=n_components, random_state=split_seed)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    angle_scaler = MinMaxScaler(feature_range=(0.0, onp.pi))
    x_train_angles = angle_scaler.fit_transform(x_train_pca)
    x_test_angles = angle_scaler.transform(x_test_pca)
    x_test_angles = onp.clip(x_test_angles, 0.0, onp.pi)

    return x_train_angles, x_test_angles


def compute_metrics(y_true: onp.ndarray, y_pred: onp.ndarray) -> dict[str, float]:
    """Calcula R2, MAE, MSE y RMSE en unidades físicas."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(onp.sqrt(mse)),
    }


def run_single_vqr(
    hyperparams: VQRHyperParams,
    fold: int,
    x_train: onp.ndarray,
    x_test: onp.ndarray,
    y_train: onp.ndarray,
    y_test: onp.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], onp.ndarray, pd.DataFrame]:
    """Entrena y evalúa una configuración VQR en un fold."""
    x_train_angles, x_test_angles = prepare_fold_features(
        x_train=x_train,
        x_test=x_test,
        n_components=hyperparams.n_qubits,
        split_seed=args.split_seed,
    )

    y_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    model = OptimizedVQR(
        params=hyperparams,
        batch_size=args.batch_size,
        epochs=args.epochs,
        log_interval=args.log_interval,
        patience=args.patience,
        min_delta=args.min_delta,
        full_loss_interval=args.full_loss_interval,
    )

    start = time.perf_counter()
    model.fit(x_train_angles, y_train_scaled)
    y_pred_scaled = model.predict_scaled(x_test_angles)
    y_pred_scaled = onp.clip(y_pred_scaled, -1.0, 1.0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    elapsed = time.perf_counter() - start

    model_name = (
        f"vqr_q{hyperparams.n_qubits}_l{hyperparams.n_layers}_"
        f"lr{hyperparams.learning_rate:g}_rot{hyperparams.rotation}_seed{hyperparams.init_seed}"
    )
    metric_row = {
        "fold": fold,
        "model": model_name,
        "n_qubits": hyperparams.n_qubits,
        "n_layers": hyperparams.n_layers,
        "learning_rate": hyperparams.learning_rate,
        "rotation": hyperparams.rotation,
        "init_seed": hyperparams.init_seed,
        "fit_predict_time_sec": elapsed,
        "stop_epoch": model.stop_epoch_ if model.stop_epoch_ is not None else args.epochs,
        **compute_metrics(y_test, y_pred),
    }

    history = pd.DataFrame(model.history_)
    history["fold"] = fold
    history["model"] = model_name
    for key, value in asdict(hyperparams).items():
        history[key] = value

    return metric_row, y_pred, history


def run_optimization(
    data: pd.DataFrame,
    grid: list[VQRHyperParams],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ejecuta la grilla VQR en KFold."""
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

    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.split_seed)
    total_jobs = args.n_folds * len(grid)
    job_counter = 0

    for fold, (train_idx, test_idx) in enumerate(kfold.split(x_all), start=1):
        x_train, x_test = x_all[train_idx], x_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        print(f"\n--- Fold {fold}/{args.n_folds} ---")
        for hyperparams in grid:
            job_counter += 1
            print(f"[{job_counter}/{total_jobs}] {hyperparams}")

            metric_row, y_pred, history = run_single_vqr(
                hyperparams=hyperparams,
                fold=fold,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                args=args,
            )
            metrics_rows.append(metric_row)
            history_frames.append(history)

            pred_frame = ids.iloc[test_idx].copy()
            pred_frame["fold"] = fold
            pred_frame["model"] = metric_row["model"]
            pred_frame["n_qubits"] = hyperparams.n_qubits
            pred_frame["n_layers"] = hyperparams.n_layers
            pred_frame["learning_rate"] = hyperparams.learning_rate
            pred_frame["rotation"] = hyperparams.rotation
            pred_frame["init_seed"] = hyperparams.init_seed
            pred_frame["y_true"] = y_test
            pred_frame["y_pred"] = y_pred
            pred_frame["residual"] = y_pred - y_test
            prediction_frames.append(pred_frame)

            print(
                f"  R2={metric_row['r2']:+.4f} "
                f"MAE={metric_row['mae']:.4f} "
                f"RMSE={metric_row['rmse']:.4f} "
                f"t={metric_row['fit_predict_time_sec']:.1f}s "
                f"stop={metric_row['stop_epoch']}"
            )

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    return metrics, predictions, history


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Resume la grilla por hiperparámetros y ordena por RMSE medio."""
    return (
        metrics.groupby(
            ["n_qubits", "n_layers", "learning_rate", "rotation", "init_seed", "model"],
            as_index=False,
        )
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            time_mean_sec=("fit_predict_time_sec", "mean"),
            time_std_sec=("fit_predict_time_sec", "std"),
            stop_epoch_mean=("stop_epoch", "mean"),
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
    run_config: dict[str, Any],
) -> None:
    """Guarda resultados de optimización."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    predictions.to_csv(output_dir / "predictions_by_fold.csv", index=False)
    history.to_csv(output_dir / "training_history.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    summary.head(1).to_csv(output_dir / "best_config.csv", index=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Ejecuta la optimización VQR."""
    args = parse_args()
    n_qubits = parse_int_list(args.n_qubits)
    n_layers = parse_int_list(args.n_layers)
    learning_rates = parse_float_list(args.learning_rates)
    rotations = parse_rotation_list(args.rotations)
    init_seeds = parse_int_list(args.init_seeds)
    grid = build_hyperparameter_grid(n_qubits, n_layers, learning_rates, rotations, init_seeds)

    start_total = time.perf_counter()
    data = load_dataset(args.input, args.n_galaxies, args.split_seed)

    print(f"Muestra cargada: {len(data)} galaxias")
    print(f"Tamaño de grilla: {len(grid)} configuraciones")
    print(f"Folds: {args.n_folds}; trabajos totales: {len(grid) * args.n_folds}")

    metrics, predictions, history = run_optimization(data, grid, args)
    summary = summarize_metrics(metrics)

    run_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "input_csv": str(args.input.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "n_galaxies": len(data),
        "n_folds": args.n_folds,
        "split_seed": args.split_seed,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "learning_rates": learning_rates,
        "rotations": rotations,
        "init_seeds": init_seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "log_interval": args.log_interval,
        "full_loss_interval": args.full_loss_interval,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "qml_embedding": "AngleEmbedding(rotation configurable)",
        "qml_ansatz": "StronglyEntanglingLayers",
        "qml_observable": "PauliZ(0)",
        "notes": "VQR hyperparameter grid with fold-local preprocessing and target scaling.",
    }
    save_outputs(args.output_dir, metrics, predictions, history, summary, run_config)

    print("\n" + "=" * 100)
    print("RESUMEN OPTIMIZACIÓN VQR")
    print("=" * 100)
    print(summary.head(20).to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("=" * 100)
    print(f"Tiempo total: {(time.perf_counter() - start_total) / 60:.2f} min")
    print(f"Resultados guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()
