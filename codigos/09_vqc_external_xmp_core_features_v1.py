"""VQC compacto para positivos XMP externos en subconjuntos físicos.

Este script agrega un Variational Quantum Classifier (VQC) como tercer enfoque
QML, complementario a los Quantum Kernel SVC de los Scripts 06--08. La corrida
por defecto está deliberadamente acotada para no repetir el coste del Script 08:
dos subconjuntos físicos, 5 folds, tres semillas CV, una capa variacional y tres
inicializaciones VQC.

La métrica primaria sigue siendo PR-AUC. El objetivo no es declarar ventaja
cuántica robusta, sino medir si el VQC es comparable a ML clásico y si aparece
una mejora ligera en `core_4` o `metallicity_5`.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "results_external_xmp_supervised_dataset_v1"
MATCHED_DATASET = DATASET_DIR / "dataset_external_xmp_matched_controls_v1.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_external_xmp_vqc_core_features_v1"
SCRIPT_VERSION = "v1"
LABEL_COLUMN = "external_xmp_label"
PRIMARY_METRIC = "pr_auc"

FEATURE_SUBSETS = {
    "core_4": ["log_nii_ha", "log_oiii_hb", "log_o32", "delta_N2_O3N2"],
    "metallicity_5": ["log_nii_ha", "log_oiii_hb", "log_r23", "log_o32", "delta_N2_O3N2"],
}

ID_COLUMNS = ["specobjid", "ra", "dec", "external_class", "sample_source"]


@dataclass(frozen=True)
class VQCConfig:
    """Configuración del clasificador variacional cuántico.

    Parameters
    ----------
    n_qubits:
        Número de qubits; igual al número de features del subconjunto.
    n_layers:
        Capas de ``StronglyEntanglingLayers``.
    seed:
        Semilla de inicialización de pesos y minibatches.
    learning_rate:
        Paso del optimizador Adam de PennyLane.
    epochs:
        Número de épocas de entrenamiento.
    batch_size:
        Tamaño del minibatch usado en cada época.
    rotation:
        Eje de ``AngleEmbedding`` para codificar features escaladas a [0, pi].
    """

    n_qubits: int
    n_layers: int
    seed: int
    learning_rate: float
    epochs: int
    batch_size: int
    rotation: str = "Y"


class VariationalQuantumClassifier:
    """VQC binario con embedding angular y capas fuertemente entrelazantes.

    El circuito devuelve ``<Z_0>``. Este valor se transforma a probabilidad de
    clase positiva como ``p = (1 - <Z_0>) / 2``. La función de pérdida es binary
    cross-entropy ponderada para compensar el desbalance 1:5 del dataset.
    """

    def __init__(self, config: VQCConfig, fold_seed: int) -> None:
        if config.n_qubits <= 0:
            raise ValueError("n_qubits debe ser positivo.")
        if config.n_layers <= 0:
            raise ValueError("n_layers debe ser positivo.")
        if config.learning_rate <= 0.0:
            raise ValueError("learning_rate debe ser positivo.")
        if config.epochs <= 0:
            raise ValueError("epochs debe ser positivo.")
        if config.batch_size <= 0:
            raise ValueError("batch_size debe ser positivo.")
        if config.rotation not in {"X", "Y", "Z"}:
            raise ValueError("rotation debe ser 'X', 'Y' o 'Z'.")

        try:
            import pennylane as qml
            from pennylane import numpy as pnp
        except ImportError as exc:
            raise ImportError(f"No se pudo cargar PennyLane para ejecutar VQC: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(
                "PennyLane está instalado pero falló al importarse. Esto suele indicar "
                "incompatibilidad entre la versión de Python y la versión de PennyLane. "
                "Ejecuta este script con el mismo entorno Python usado para los Scripts 07/08 "
                "o con Python >= 3.9 si tu instalación de PennyLane lo requiere. "
                f"Error original: {type(exc).__name__}: {exc}"
            ) from exc

        self.qml = qml
        self.pnp = pnp
        self.config = config
        self.rng = np.random.default_rng(config.seed + 10_000 * fold_seed)
        self.history_: list[dict[str, float | int]] = []
        dev = qml.device("default.qubit", wires=config.n_qubits)

        @qml.qnode(dev)
        def circuit(weights: pnp.ndarray, x_row: pnp.ndarray) -> pnp.ndarray:
            qml.AngleEmbedding(x_row, wires=range(config.n_qubits), rotation=config.rotation)
            qml.StronglyEntanglingLayers(weights, wires=range(config.n_qubits))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        initial = self.rng.uniform(
            low=-0.1,
            high=0.1,
            size=(config.n_layers, config.n_qubits, 3),
        )
        self.weights = pnp.array(initial, requires_grad=True)

    def _positive_probabilities(self, weights: Any, x_data: Any) -> Any:
        """Calcula probabilidades VQC de clase positiva."""
        expvals = self.pnp.array([self.circuit(weights, row) for row in x_data])
        probs = 0.5 * (1.0 - expvals)
        return self.pnp.clip(probs, 1.0e-6, 1.0 - 1.0e-6)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Entrena el VQC sobre features angulares.

        Parameters
        ----------
        x_train:
            Matriz de features ya escaladas a ángulos en [0, pi].
        y_train:
            Etiquetas binarias 0/1.
        """
        if x_train.ndim != 2 or x_train.shape[1] != self.config.n_qubits:
            raise ValueError(f"x_train debe tener forma (n_samples, {self.config.n_qubits}).")
        if len(x_train) != len(y_train):
            raise ValueError("x_train e y_train deben tener la misma longitud.")
        if not np.isin(y_train, [0, 1]).all():
            raise ValueError("y_train debe contener sólo etiquetas 0/1.")

        x_pnp = self.pnp.array(x_train, requires_grad=False)
        y_pnp = self.pnp.array(y_train.astype(float), requires_grad=False)
        positives = float(np.sum(y_train == 1))
        negatives = float(np.sum(y_train == 0))
        if positives == 0.0 or negatives == 0.0:
            raise ValueError("Cada fold de entrenamiento debe contener ambas clases.")
        positive_weight = negatives / positives
        sample_weights = self.pnp.where(y_pnp > 0.5, positive_weight, 1.0)
        effective_batch = min(self.config.batch_size, len(x_train))
        optimizer = self.qml.AdamOptimizer(stepsize=self.config.learning_rate)

        for epoch in range(1, self.config.epochs + 1):
            batch_idx = self.rng.choice(len(x_train), size=effective_batch, replace=False)
            x_batch = x_pnp[batch_idx]
            y_batch = y_pnp[batch_idx]
            w_batch = sample_weights[batch_idx]

            def cost(weights: Any) -> Any:
                probs = self._positive_probabilities(weights, x_batch)
                loss = -(y_batch * self.pnp.log(probs) + (1.0 - y_batch) * self.pnp.log(1.0 - probs))
                return self.pnp.mean(w_batch * loss)

            self.weights = optimizer.step(cost, self.weights)
            if epoch == 1 or epoch == self.config.epochs or epoch % max(10, self.config.epochs // 2) == 0:
                self.history_.append({"epoch": epoch, "batch_loss": float(cost(self.weights))})

    def predict_scores(self, x_data: np.ndarray) -> np.ndarray:
        """Devuelve scores/probabilidades de clase positiva para ranking."""
        if x_data.ndim != 2 or x_data.shape[1] != self.config.n_qubits:
            raise ValueError(f"x_data debe tener {self.config.n_qubits} columnas.")
        x_pnp = self.pnp.array(x_data, requires_grad=False)
        return np.asarray(self._positive_probabilities(self.weights, x_pnp), dtype=float).ravel()


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="VQC compacto para benchmark externo XMP.")
    parser.add_argument("--input", type=Path, default=MATCHED_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--cv-seeds", type=str, default="11,22,33")
    parser.add_argument("--feature-subsets", type=str, default="core_4,metallicity_5")
    parser.add_argument("--classical-seed", type=int, default=42)
    parser.add_argument("--vqc-layers", type=str, default="1")
    parser.add_argument("--vqc-seeds", type=str, default="101,202,303")
    parser.add_argument("--vqc-epochs", type=int, default=80)
    parser.add_argument("--vqc-batch-size", type=int, default=32)
    parser.add_argument("--vqc-learning-rate", type=float, default=0.02)
    parser.add_argument("--vqc-rotation", type=str, default="Y", choices=["X", "Y", "Z"])
    parser.add_argument("--top-k", type=str, default="20,50")
    parser.add_argument("--skip-vqc", action="store_true")
    parser.add_argument("--skip-classical", action="store_true")
    return parser.parse_args()


def parse_str_list(text: str) -> list[str]:
    """Convierte una lista separada por comas en strings no vacíos."""
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("La lista no puede estar vacía.")
    return values


def parse_int_list(text: str) -> list[int]:
    """Convierte una lista separada por comas en enteros."""
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("La lista de enteros no puede estar vacía.")
    return values


def validate_output_dir(output_dir: Path) -> Path:
    """Valida que la salida quede dentro de Parte_3 y crea el directorio."""
    resolved = output_dir.expanduser().resolve()
    allowed_root = SCRIPT_DIR.resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        raise ValueError(f"Por seguridad, --output-dir debe estar dentro de Parte_3: {allowed_root}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_feature_subsets(names: list[str]) -> dict[str, list[str]]:
    """Valida y resuelve nombres de subconjuntos físicos."""
    invalid = sorted(set(names) - set(FEATURE_SUBSETS))
    if invalid:
        valid = ", ".join(FEATURE_SUBSETS)
        raise ValueError(f"Subconjuntos inválidos: {invalid}. Opciones válidas: {valid}")
    return {name: FEATURE_SUBSETS[name] for name in names}


def load_dataset(input_csv: Path, feature_subsets: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Carga dataset, valida columnas, etiquetas y valores finitos."""
    if not input_csv.exists():
        raise FileNotFoundError(f"No existe dataset de entrada: {input_csv}")
    data = pd.read_csv(input_csv)
    n_initial = len(data)
    feature_columns = sorted({feature for features in feature_subsets.values() for feature in features})
    required = set(feature_columns + [LABEL_COLUMN])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")
    keep_columns = [column for column in ID_COLUMNS if column in data.columns] + feature_columns + [LABEL_COLUMN]
    data = data.loc[:, keep_columns].copy()
    for column in feature_columns + [LABEL_COLUMN]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    finite_features = np.isfinite(data[feature_columns]).all(axis=1)
    valid_label = data[LABEL_COLUMN].isin([0, 1])
    data = data.loc[finite_features & valid_label].reset_index(drop=True)
    n_removed = n_initial - len(data)
    if data[LABEL_COLUMN].nunique() != 2:
        raise ValueError("El dataset debe contener ambas clases tras el filtrado.")
    return data, {"n_rows_initial": int(n_initial), "n_rows_removed_invalid": int(n_removed)}


def build_classical_models(seed: int) -> dict[str, Any]:
    """Construye modelos clásicos comparativos con escalado dentro del fold."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=15,
            random_state=seed,
        ),
        "svc_rbf": Pipeline(
            [("scaler", StandardScaler()), ("model", SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced"))]
        ),
        "svc_poly": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="poly", degree=3, C=3.0, gamma="scale", coef0=1.0, class_weight="balanced")),
            ]
        ),
    }


def decision_scores(model: Any, x_test: np.ndarray) -> np.ndarray:
    """Devuelve scores continuos donde mayor valor implica clase positiva."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x_test), dtype=float).ravel()
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x_test)[:, 1], dtype=float).ravel()
    return np.asarray(model.predict(x_test), dtype=float).ravel()


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Calcula precision@k para ranking binario."""
    if k <= 0:
        raise ValueError("k debe ser positivo.")
    top = np.argsort(-scores)[: min(k, len(scores))]
    return float(np.mean(y_true[top])) if len(top) else np.nan


def recall_at_fraction(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    """Calcula recall en el top fraction de objetos rankeados."""
    k = max(1, int(np.ceil(fraction * len(scores))))
    top = np.argsort(-scores)[:k]
    positives = np.sum(y_true == 1)
    return float(np.sum(y_true[top] == 1) / positives) if positives else np.nan


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, top_ks: list[int]) -> dict[str, float]:
    """Calcula métricas de ranking para clase rara."""
    if not np.isfinite(scores).all():
        raise ValueError("Los scores contienen NaN o inf.")
    metrics = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "recall_at_top5percent": recall_at_fraction(y_true, scores, 0.05),
    }
    for k in top_ks:
        metrics[f"precision_at_{k}"] = precision_at_k(y_true, scores, k)
    return metrics


def evaluate_classical_models(
    subset_name: str,
    features: list[str],
    data: pd.DataFrame,
    y: np.ndarray,
    cv_seed: int,
    n_folds: int,
    top_ks: list[int],
    classical_seed: int,
) -> list[dict[str, Any]]:
    """Evalúa todos los clásicos para un subconjunto y una partición CV."""
    x = data[features].to_numpy(dtype=float)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        for model_name, model_template in build_classical_models(classical_seed).items():
            start = time.perf_counter()
            model = clone(model_template)
            model.fit(x_train, y_train)
            scores = decision_scores(model, x_test)
            rows.append(
                {
                    "feature_subset": subset_name,
                    "n_features": len(features),
                    "features": ";".join(features),
                    "cv_seed": cv_seed,
                    "fold": fold,
                    "model": model_name,
                    "model_family": "classical",
                    "vqc_layers": np.nan,
                    "vqc_seed": np.nan,
                    "vqc_epochs": np.nan,
                    "fit_score_time_sec": time.perf_counter() - start,
                    **compute_metrics(y_test, scores, top_ks),
                }
            )
    return rows


def evaluate_vqc_models(
    subset_name: str,
    features: list[str],
    data: pd.DataFrame,
    y: np.ndarray,
    cv_seed: int,
    n_folds: int,
    layers: list[int],
    seeds: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    rotation: str,
    top_ks: list[int],
) -> list[dict[str, Any]]:
    """Evalúa VQC para un subconjunto y una partición CV."""
    x = data[features].to_numpy(dtype=float)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
    rows = []
    for n_layers in layers:
        for vqc_seed in seeds:
            for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
                x_train_raw, x_test_raw = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                scaler = MinMaxScaler(feature_range=(0.0, np.pi)).fit(x_train_raw)
                x_train = np.clip(scaler.transform(x_train_raw), 0.0, np.pi)
                x_test = np.clip(scaler.transform(x_test_raw), 0.0, np.pi)
                print(
                    f"Subset={subset_name} cv_seed={cv_seed} VQC layers={n_layers} "
                    f"seed={vqc_seed} fold={fold}",
                    flush=True,
                )
                start = time.perf_counter()
                config = VQCConfig(
                    n_qubits=len(features),
                    n_layers=n_layers,
                    seed=vqc_seed,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size,
                    rotation=rotation,
                )
                model = VariationalQuantumClassifier(config=config, fold_seed=cv_seed * 100 + fold)
                model.fit(x_train, y_train)
                scores = model.predict_scores(x_test)
                rows.append(
                    {
                        "feature_subset": subset_name,
                        "n_features": len(features),
                        "features": ";".join(features),
                        "cv_seed": cv_seed,
                        "fold": fold,
                        "model": f"vqc_layers{n_layers}_seed{vqc_seed}_epochs{epochs}",
                        "model_family": "quantum_variational",
                        "vqc_layers": n_layers,
                        "vqc_seed": vqc_seed,
                        "vqc_epochs": epochs,
                        "fit_score_time_sec": time.perf_counter() - start,
                        **compute_metrics(y_test, scores, top_ks),
                    }
                )
    return rows


def build_summary_by_model(metrics: pd.DataFrame) -> pd.DataFrame:
    """Resume métricas por subconjunto, familia, modelo e hiperparámetros."""
    group_cols = [
        "feature_subset",
        "n_features",
        "features",
        "model",
        "model_family",
        "vqc_layers",
        "vqc_seed",
        "vqc_epochs",
    ]
    metric_cols = [col for col in metrics.columns if col not in set(group_cols + ["cv_seed", "fold"])]
    rows = []
    for keys, subset in metrics.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_folds_total"] = int(len(subset))
        row["n_cv_seeds"] = int(subset["cv_seed"].nunique())
        for col in metric_cols:
            if pd.api.types.is_numeric_dtype(subset[col]):
                row[f"{col}_mean"] = float(subset[col].mean())
                row[f"{col}_std"] = float(subset[col].std(ddof=1)) if len(subset) > 1 else 0.0
                row[f"{col}_median"] = float(subset[col].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)


def build_delta_by_cv_seed(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calcula delta VQC-clásico por subconjunto y cv_seed."""
    rows = []
    output_columns = [
        "feature_subset",
        "cv_seed",
        "best_classical_model",
        "best_vqc_model",
        "best_vqc_layers",
        "best_vqc_seed",
        "best_vqc_epochs",
    ]
    metric_cols = [col for col in metrics.columns if col.startswith("precision_at_")]
    metric_cols += ["pr_auc", "roc_auc", "recall_at_top5percent"]
    for col in metric_cols:
        output_columns.extend(
            [f"best_classical_{col}", f"best_vqc_{col}", f"delta_vqc_minus_classical_{col}"]
        )
    for (subset_name, cv_seed), subset in metrics.groupby(["feature_subset", "cv_seed"]):
        group_cols = ["model", "model_family", "vqc_layers", "vqc_seed", "vqc_epochs"]
        averaged = subset.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
        classical = averaged.loc[averaged["model_family"] == "classical"].sort_values("pr_auc", ascending=False)
        vqc = averaged.loc[averaged["model_family"] == "quantum_variational"].sort_values("pr_auc", ascending=False)
        if classical.empty or vqc.empty:
            continue
        best_classical = classical.iloc[0]
        best_vqc = vqc.iloc[0]
        row: dict[str, Any] = {
            "feature_subset": subset_name,
            "cv_seed": int(cv_seed),
            "best_classical_model": best_classical["model"],
            "best_vqc_model": best_vqc["model"],
            "best_vqc_layers": best_vqc["vqc_layers"],
            "best_vqc_seed": best_vqc["vqc_seed"],
            "best_vqc_epochs": best_vqc["vqc_epochs"],
        }
        for col in metric_cols:
            row[f"best_classical_{col}"] = float(best_classical[col])
            row[f"best_vqc_{col}"] = float(best_vqc[col])
            row[f"delta_vqc_minus_classical_{col}"] = float(best_vqc[col] - best_classical[col])
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=output_columns)
    return pd.DataFrame(rows).sort_values(["feature_subset", "cv_seed"]).reset_index(drop=True)


def build_delta_distribution_summary(delta_by_seed: pd.DataFrame) -> pd.DataFrame:
    """Resume distribución de deltas VQC-clásico por subconjunto."""
    rows = []
    if delta_by_seed.empty:
        return pd.DataFrame()
    for subset_name, subset in delta_by_seed.groupby("feature_subset"):
        deltas = subset["delta_vqc_minus_classical_pr_auc"].to_numpy(dtype=float)
        row = {
            "feature_subset": subset_name,
            "n_cv_seeds": int(len(deltas)),
            "delta_pr_auc_mean": float(np.mean(deltas)),
            "delta_pr_auc_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "delta_pr_auc_median": float(np.median(deltas)),
            "delta_pr_auc_q16": float(np.quantile(deltas, 0.16)),
            "delta_pr_auc_q84": float(np.quantile(deltas, 0.84)),
            "fraction_positive_delta": float(np.mean(deltas > 0.0)),
            "fraction_delta_ge_0p02": float(np.mean(deltas >= 0.02)),
        }
        for col in ["roc_auc", "recall_at_top5percent"]:
            delta_col = f"delta_vqc_minus_classical_{col}"
            if delta_col in subset.columns:
                row[f"delta_{col}_mean"] = float(subset[delta_col].mean())
                row[f"delta_{col}_median"] = float(subset[delta_col].median())
        precision_cols = [col for col in subset.columns if col.startswith("delta_vqc_minus_classical_precision_at_")]
        for col in precision_cols:
            suffix = col.replace("delta_vqc_minus_classical_", "")
            row[f"delta_{suffix}_mean"] = float(subset[col].mean())
            row[f"delta_{suffix}_median"] = float(subset[col].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("delta_pr_auc_mean", ascending=False).reset_index(drop=True)


def build_vqc_interpretation(delta_summary: pd.DataFrame) -> pd.DataFrame:
    """Clasifica el resultado en comparable, ligeramente superior o inferior."""
    rows = []
    for _, row in delta_summary.iterrows():
        precision_loss = False
        precision_cols = [col for col in delta_summary.columns if col.startswith("delta_precision_at_") and col.endswith("_mean")]
        for col in precision_cols:
            if float(row[col]) < -0.02:
                precision_loss = True
        delta_mean = float(row["delta_pr_auc_mean"])
        fraction_positive = float(row["fraction_positive_delta"])
        comparable = delta_mean >= -0.02 and not precision_loss
        slightly_better = delta_mean > 0.0 and fraction_positive >= 0.5 and not precision_loss
        if slightly_better:
            interpretation = "VQC comparable y ligeramente favorable frente al mejor clásico"
        elif comparable:
            interpretation = "VQC comparable al mejor clásico, sin ventaja clara"
        else:
            interpretation = "VQC inferior o con pérdida relevante en ranking top-k"
        rows.append(
            {
                "feature_subset": row["feature_subset"],
                "vqc_comparable_to_classical": bool(comparable),
                "vqc_slightly_better_than_classical": bool(slightly_better),
                "criterion_delta_pr_auc_mean_ge_minus_0p02": bool(delta_mean >= -0.02),
                "criterion_delta_pr_auc_mean_gt_0": bool(delta_mean > 0.0),
                "criterion_fraction_positive_ge_0p50": bool(fraction_positive >= 0.5),
                "criterion_no_clear_precision_at_k_loss": bool(not precision_loss),
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def save_outputs(output_dir: Path, outputs: dict[str, pd.DataFrame], run_config: dict[str, Any]) -> None:
    """Guarda CSVs y configuración JSON."""
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Ejecuta el benchmark compacto VQC vs ML clásico."""
    args = parse_args()
    if args.skip_classical and args.skip_vqc:
        raise ValueError("No se puede usar --skip-classical y --skip-vqc al mismo tiempo.")
    output_dir = validate_output_dir(args.output_dir)
    feature_subsets = resolve_feature_subsets(parse_str_list(args.feature_subsets))
    cv_seeds = parse_int_list(args.cv_seeds)
    vqc_layers = parse_int_list(args.vqc_layers)
    vqc_seeds = parse_int_list(args.vqc_seeds)
    top_ks = parse_int_list(args.top_k)
    if any(layer <= 0 for layer in vqc_layers):
        raise ValueError("Todas las capas VQC deben ser positivas.")
    if args.n_folds < 2:
        raise ValueError("--n-folds debe ser >= 2.")
    if args.vqc_epochs <= 0 or args.vqc_batch_size <= 0 or args.vqc_learning_rate <= 0.0:
        raise ValueError("epochs, batch size y learning rate deben ser positivos.")

    data, load_metadata = load_dataset(args.input, feature_subsets)
    y = data[LABEL_COLUMN].to_numpy(dtype=int)
    class_counts = np.bincount(y, minlength=2)
    if args.n_folds > int(class_counts.min()):
        raise ValueError(f"--n-folds={args.n_folds} excede la clase minoritaria: {int(class_counts.min())}")

    print(f"Dataset: {args.input}", flush=True)
    print(f"Filas válidas: {len(data)}; positivos={int(class_counts[1])}; negativos={int(class_counts[0])}", flush=True)
    print(f"Subconjuntos: {', '.join(feature_subsets)}", flush=True)
    print(f"CV seeds: {cv_seeds}", flush=True)
    if not args.skip_vqc:
        n_trainings = len(feature_subsets) * len(cv_seeds) * len(vqc_layers) * len(vqc_seeds) * args.n_folds
        print(
            f"VQC compacto: layers={vqc_layers}, seeds={vqc_seeds}, epochs={args.vqc_epochs}; "
            f"entrenamientos VQC={n_trainings}",
            flush=True,
        )

    metric_rows: list[dict[str, Any]] = []
    for subset_name, features in feature_subsets.items():
        for cv_seed in cv_seeds:
            if not args.skip_classical:
                metric_rows.extend(
                    evaluate_classical_models(
                        subset_name=subset_name,
                        features=features,
                        data=data,
                        y=y,
                        cv_seed=cv_seed,
                        n_folds=args.n_folds,
                        top_ks=top_ks,
                        classical_seed=args.classical_seed,
                    )
                )
            if not args.skip_vqc:
                metric_rows.extend(
                    evaluate_vqc_models(
                        subset_name=subset_name,
                        features=features,
                        data=data,
                        y=y,
                        cv_seed=cv_seed,
                        n_folds=args.n_folds,
                        layers=vqc_layers,
                        seeds=vqc_seeds,
                        epochs=args.vqc_epochs,
                        batch_size=args.vqc_batch_size,
                        learning_rate=args.vqc_learning_rate,
                        rotation=args.vqc_rotation,
                        top_ks=top_ks,
                    )
                )

    metrics = pd.DataFrame(metric_rows)
    summary_by_model = build_summary_by_model(metrics)
    delta_by_cv_seed = build_delta_by_cv_seed(metrics)
    delta_summary = build_delta_distribution_summary(delta_by_cv_seed)
    interpretation = build_vqc_interpretation(delta_summary) if not delta_summary.empty else pd.DataFrame()
    best_vqc = delta_by_cv_seed.loc[
        :,
        [
            "feature_subset",
            "cv_seed",
            "best_vqc_model",
            "best_vqc_layers",
            "best_vqc_seed",
            "best_vqc_epochs",
            "best_vqc_pr_auc",
            "best_vqc_roc_auc",
        ],
    ].copy() if not delta_by_cv_seed.empty else pd.DataFrame()

    run_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "input_csv": str(args.input.resolve()),
        "output_dir": str(output_dir),
        "n_rows": int(len(data)),
        **load_metadata,
        "n_positive": int(class_counts[1]),
        "n_negative": int(class_counts[0]),
        "feature_subsets": feature_subsets,
        "n_folds": args.n_folds,
        "cv_seeds": cv_seeds,
        "classical_seed": args.classical_seed,
        "classical_models": list(build_classical_models(args.classical_seed)),
        "vqc_layers": vqc_layers,
        "vqc_seeds": vqc_seeds,
        "vqc_epochs": args.vqc_epochs,
        "vqc_batch_size": args.vqc_batch_size,
        "vqc_learning_rate": args.vqc_learning_rate,
        "vqc_rotation": args.vqc_rotation,
        "top_k": top_ks,
        "primary_metric": PRIMARY_METRIC,
        "skip_vqc": args.skip_vqc,
        "skip_classical": args.skip_classical,
        "notes": (
            "Compact VQC benchmark against the same classical baselines used in Scripts 06--08. "
            "The purpose is comparability and possible small QML advantage, not strict confirmation of quantum superiority."
        ),
    }
    save_outputs(
        output_dir,
        {
            "metrics_by_fold": metrics,
            "summary_by_model": summary_by_model,
            "delta_by_cv_seed": delta_by_cv_seed,
            "delta_distribution_summary": delta_summary,
            "best_vqc_configs_by_cv_seed": best_vqc,
            "vqc_interpretation": interpretation,
        },
        run_config,
    )
    print(f"Salida escrita en: {output_dir}")
    if not delta_summary.empty:
        print(delta_summary.to_string(index=False))
        print(interpretation.to_string(index=False))


if __name__ == "__main__":
    main()
