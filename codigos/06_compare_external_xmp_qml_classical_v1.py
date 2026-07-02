"""Benchmark supervisado ML clásico vs QML para XMP/metal-poor externas.

Este script compara modelos clásicos y Quantum Kernel SVC sobre el dataset
supervisado construido en ``05_build_external_xmp_supervised_dataset_v1.py``.
El dataset principal recomendado es la versión con controles pareados en
redshift y masa estelar, porque reduce sesgos triviales de selección.

La métrica principal es PR-AUC y recuperación top-k, no accuracy global. Todos
los escalados se ajustan dentro de cada fold para evitar leakage.
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
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "results_external_xmp_supervised_dataset_v1"
MATCHED_DATASET = DATASET_DIR / "dataset_external_xmp_matched_controls_v1.csv"
RANDOM_DATASET = DATASET_DIR / "dataset_external_xmp_random_controls_v1.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_external_xmp_qml_classical_v1"
SCRIPT_VERSION = "v1"

BASE_FEATURE_COLUMNS = [
    "log_nii_ha",
    "log_oiii_hb",
    "log_oii_hb",
    "log_sii_ha",
    "log_r23",
    "log_o32",
    "log_stellar_mass",
    "z",
    "o3n2",
    "delta_N2_O3N2",
]
LABEL_COLUMN = "external_xmp_label"
ID_COLUMNS = ["specobjid", "ra", "dec", "external_class", "sample_source"]


@dataclass(frozen=True)
class QuantumFeatureMapConfig:
    """Configuración de feature map cuántico para kernel de fidelidad."""

    name: str
    n_layers: int
    seed: int
    rotation: str = "Y"


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Benchmark ML vs QML para XMP externos.")
    parser.add_argument("--dataset", choices=["matched", "random"], default="matched")
    parser.add_argument("--input", type=Path, default=None, help="CSV de dataset supervisado. Sobrescribe --dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio de salida. Si se omite, se usa una carpeta separada por dataset.",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantum-feature-maps", type=str, default="reuploading,zz")
    parser.add_argument("--quantum-layers", type=int, default=2)
    parser.add_argument("--quantum-seeds", type=str, default="101,202,303")
    parser.add_argument("--skip-quantum", action="store_true")
    parser.add_argument("--top-k", type=str, default="20,50")
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    """Parsea lista CSV de strings."""
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("La lista no puede estar vacía.")
    return values


def parse_int_list(text: str) -> list[int]:
    """Parsea lista CSV de enteros."""
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("La lista de enteros no puede estar vacía.")
    return values


def resolve_input_path(args: argparse.Namespace) -> Path:
    """Resuelve el dataset de entrada."""
    if args.input is not None:
        return args.input
    return MATCHED_DATASET if args.dataset == "matched" else RANDOM_DATASET


def validate_output_dir(output_dir: Path) -> Path:
    """Valida que la salida quede dentro de Parte_3."""
    resolved = output_dir.expanduser().resolve()
    allowed_root = SCRIPT_DIR.resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        raise ValueError(f"Por seguridad, --output-dir debe estar dentro de Parte_3: {allowed_root}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_dataset(input_csv: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Carga y valida dataset supervisado."""
    if not input_csv.exists():
        raise FileNotFoundError(f"No existe dataset de entrada: {input_csv}")
    data = pd.read_csv(input_csv)
    n_initial = len(data)
    required = set(BASE_FEATURE_COLUMNS + [LABEL_COLUMN, "specobjid", "ra", "dec"])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")
    columns = [col for col in ID_COLUMNS if col in data.columns] + BASE_FEATURE_COLUMNS + [LABEL_COLUMN]
    data = data.loc[:, columns].copy()
    for column in BASE_FEATURE_COLUMNS + [LABEL_COLUMN, "ra", "dec"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    finite = np.isfinite(data[BASE_FEATURE_COLUMNS]).all(axis=1)
    valid_label = data[LABEL_COLUMN].isin([0, 1])
    data = data.loc[finite & valid_label].reset_index(drop=True)
    n_removed = n_initial - len(data)
    if n_removed:
        print(f"Filas descartadas por features no finitas o etiqueta inválida: {n_removed}/{n_initial}", flush=True)
    if data[LABEL_COLUMN].nunique() != 2:
        raise ValueError("El dataset debe contener ambas clases.")
    return data, {"n_rows_initial": int(n_initial), "n_rows_removed_invalid": int(n_removed)}


def resolve_output_dir(args: argparse.Namespace, input_csv: Path) -> tuple[Path, str]:
    """Resuelve salida y etiqueta de dataset evitando sobrescrituras por defecto."""
    dataset_tag = args.dataset if args.input is None else input_csv.stem
    if args.output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / dataset_tag
    else:
        output_dir = args.output_dir
    return validate_output_dir(output_dir), dataset_tag


def build_classical_models(seed: int) -> dict[str, Any]:
    """Construye modelos clásicos comparativos."""
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
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced", probability=False)),
            ]
        ),
        "svc_poly": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="poly", degree=3, C=3.0, gamma="scale", coef0=1.0, class_weight="balanced")),
            ]
        ),
    }


def decision_scores(model: Any, x_test: np.ndarray) -> np.ndarray:
    """Obtiene score continuo para ranking binario."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x_test), dtype=float).ravel()
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x_test)[:, 1], dtype=float).ravel()
    return np.asarray(model.predict(x_test), dtype=float).ravel()


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Precision@k para scores donde mayor implica más positivo."""
    if k <= 0:
        raise ValueError("k debe ser positivo.")
    top = np.argsort(-scores)[: min(k, len(scores))]
    return float(np.mean(y_true[top])) if len(top) else np.nan


def recall_at_fraction(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    """Recall en el top fraction de objetos rankeados."""
    k = max(1, int(np.ceil(fraction * len(scores))))
    top = np.argsort(-scores)[:k]
    positives = np.sum(y_true == 1)
    return float(np.sum(y_true[top] == 1) / positives) if positives else np.nan


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    """Calcula F1 y balanced accuracy con umbral optimizado en test fold."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        preds = np.zeros_like(y_true)
    else:
        f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        threshold = thresholds[int(np.nanargmax(f1_values))]
        preds = (scores >= threshold).astype(int)
    return (
        float(f1_score(y_true, preds, zero_division=0)),
        float(balanced_accuracy_score(y_true, preds)),
        float(np.mean(preds)),
    )


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, top_ks: list[int]) -> dict[str, float]:
    """Calcula métricas principales."""
    metrics: dict[str, float] = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "recall_at_top5percent": recall_at_fraction(y_true, scores, 0.05),
    }
    f1, balanced, pred_frac = threshold_metrics(y_true, scores)
    metrics["f1_test_threshold_oracle"] = f1
    metrics["balanced_accuracy_test_threshold_oracle"] = balanced
    metrics["predicted_positive_fraction_test_threshold_oracle"] = pred_frac
    for k in top_ks:
        metrics[f"precision_at_{k}"] = precision_at_k(y_true, scores, k)
    return metrics


def quantum_states_from_angles(x_angles: np.ndarray, config: QuantumFeatureMapConfig) -> np.ndarray:
    """Evalúa estados cuánticos desde ángulos ya escalados a [0, pi]."""
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except Exception as exc:  # pragma: no cover - depende del entorno local
        raise ImportError(f"No se pudo cargar PennyLane para ejecutar modelos QML: {type(exc).__name__}: {exc}") from exc

    n_qubits = x_angles.shape[1]
    rng = np.random.default_rng(config.seed + 97 * n_qubits)
    reupload_weights = pnp.array(rng.uniform(0.0, 2.0 * np.pi, size=(config.n_layers, n_qubits)), requires_grad=False)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x_row: pnp.ndarray) -> pnp.ndarray:
        if config.name == "reuploading":
            for layer in range(config.n_layers):
                qml.AngleEmbedding(x_row, wires=range(n_qubits), rotation=config.rotation)
                qml.BasicEntanglerLayers(reupload_weights[layer : layer + 1], wires=range(n_qubits))
            qml.AngleEmbedding(x_row, wires=range(n_qubits), rotation="Z")
        elif config.name == "zz":
            qml.AngleEmbedding(x_row, wires=range(n_qubits), rotation="Y")
            for _ in range(config.n_layers):
                for wire in range(n_qubits):
                    qml.RZ(float(x_row[wire] ** 2), wires=wire)
                for wire in range(n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
                    qml.RZ(float(x_row[wire] * x_row[wire + 1]), wires=wire + 1)
                    qml.CNOT(wires=[wire, wire + 1])
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits - 1, 0])
                    qml.RZ(float(x_row[-1] * x_row[0]), wires=0)
                    qml.CNOT(wires=[n_qubits - 1, 0])
                qml.AngleEmbedding(x_row, wires=range(n_qubits), rotation="X")
        else:
            raise ValueError(f"Feature map cuántico inválido: {config.name}")
        return qml.state()

    return np.vstack([np.asarray(circuit(pnp.array(row, requires_grad=False)), dtype=complex) for row in x_angles])


def kernel_from_states(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    """Kernel de fidelidad |<phi(a)|phi(b)>|^2."""
    return np.asarray(np.abs(states_a @ np.conjugate(states_b.T)) ** 2, dtype=float)


def fit_predict_quantum_kernel(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    config: QuantumFeatureMapConfig,
) -> np.ndarray:
    """Entrena SVC con kernel cuántico precomputado y devuelve scores test."""
    scaler = MinMaxScaler(feature_range=(0.0, np.pi)).fit(x_train)
    train_angles = np.clip(scaler.transform(x_train), 0.0, np.pi)
    test_angles = np.clip(scaler.transform(x_test), 0.0, np.pi)
    states_train = quantum_states_from_angles(train_angles, config)
    states_test = quantum_states_from_angles(test_angles, config)
    k_train = kernel_from_states(states_train, states_train)
    k_train = 0.5 * (k_train + k_train.T)
    np.fill_diagonal(k_train, 1.0)
    k_test = kernel_from_states(states_test, states_train)
    model = SVC(kernel="precomputed", C=3.0, class_weight="balanced")
    model.fit(k_train, y_train)
    return np.asarray(model.decision_function(k_test), dtype=float).ravel()


def evaluate_classical_models(
    models: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    data: pd.DataFrame,
    cv: StratifiedKFold,
    top_ks: list[int],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    """Evalúa modelos clásicos por validación cruzada."""
    metric_rows = []
    prediction_frames = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        for model_name, model_template in models.items():
            start = time.perf_counter()
            model = clone(model_template)
            model.fit(x_train, y_train)
            scores = decision_scores(model, x_test)
            elapsed = time.perf_counter() - start
            metrics = compute_metrics(y_test, scores, top_ks)
            metric_rows.append({"fold": fold, "model": model_name, "model_family": "classical", "fit_score_time_sec": elapsed, **metrics})
            frame = data.iloc[test_idx][[col for col in ID_COLUMNS if col in data.columns]].copy()
            frame["fold"] = fold
            frame["model"] = model_name
            frame["y_true"] = y_test
            frame["score"] = scores
            prediction_frames.append(frame)
    return metric_rows, prediction_frames


def evaluate_quantum_models(
    configs: list[QuantumFeatureMapConfig],
    x: np.ndarray,
    y: np.ndarray,
    data: pd.DataFrame,
    cv: StratifiedKFold,
    top_ks: list[int],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    """Evalúa modelos QK-SVC por validación cruzada."""
    metric_rows = []
    prediction_frames = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        for config in configs:
            model_name = f"qksvc_{config.name}_layers{config.n_layers}_seed{config.seed}"
            if config.name == "zz":
                model_name = f"qksvc_{config.name}_layers{config.n_layers}_deterministic"
            print(f"Fold {fold}: {model_name}", flush=True)
            start = time.perf_counter()
            scores = fit_predict_quantum_kernel(x_train, y_train, x_test, config)
            elapsed = time.perf_counter() - start
            metrics = compute_metrics(y_test, scores, top_ks)
            metric_rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "model_family": "quantum_kernel",
                    "quantum_feature_map": config.name,
                    "quantum_layers": config.n_layers,
                    "quantum_seed": config.seed,
                    "fit_score_time_sec": elapsed,
                    **metrics,
                }
            )
            frame = data.iloc[test_idx][[col for col in ID_COLUMNS if col in data.columns]].copy()
            frame["fold"] = fold
            frame["model"] = model_name
            frame["y_true"] = y_test
            frame["score"] = scores
            prediction_frames.append(frame)
    return metric_rows, prediction_frames


def build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Resume métricas por modelo."""
    metric_cols = [col for col in metrics.columns if col not in {"fold", "model", "model_family", "quantum_feature_map", "quantum_layers", "quantum_seed"}]
    rows = []
    for model, subset in metrics.groupby("model"):
        row: dict[str, Any] = {"model": model, "model_family": subset["model_family"].iloc[0], "n_folds": len(subset)}
        for col in metric_cols:
            if pd.api.types.is_numeric_dtype(subset[col]):
                row[f"{col}_mean"] = float(subset[col].mean())
                row[f"{col}_std"] = float(subset[col].std(ddof=1)) if len(subset) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)


def build_top_ranked(predictions: pd.DataFrame, top_ks: list[int]) -> pd.DataFrame:
    """Extrae candidatos top-k por modelo sobre predicciones out-of-fold."""
    rows = []
    for model, subset in predictions.groupby("model"):
        subset = subset.copy()
        subset["rank_within_fold"] = subset.groupby("fold")["score"].rank(ascending=False, method="first")
        subset["score_percentile_within_fold"] = subset.groupby("fold")["score"].rank(pct=True)
        ordered = subset.sort_values("score", ascending=False).reset_index(drop=True)
        for k in top_ks:
            top = ordered.head(min(k, len(ordered))).copy()
            top["top_k"] = k
            top["rank_within_model"] = np.arange(1, len(top) + 1)
            rows.append(top)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_top_ranked_by_fold(predictions: pd.DataFrame, top_ks: list[int]) -> pd.DataFrame:
    """Extrae top-k dentro de cada fold para evitar mezclar escalas de score."""
    rows = []
    for (model, fold), subset in predictions.groupby(["model", "fold"]):
        ordered = subset.sort_values("score", ascending=False).reset_index(drop=True)
        for k in top_ks:
            top = ordered.head(min(k, len(ordered))).copy()
            top["top_k"] = k
            top["rank_within_fold"] = np.arange(1, len(top) + 1)
            rows.append(top)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def save_outputs(output_dir: Path, outputs: dict[str, pd.DataFrame], run_config: dict[str, Any]) -> None:
    """Guarda tablas y configuración."""
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Ejecuta benchmark completo."""
    args = parse_args()
    input_csv = resolve_input_path(args)
    output_dir, dataset_tag = resolve_output_dir(args, input_csv)
    top_ks = parse_int_list(args.top_k)
    feature_maps = parse_csv_list(args.quantum_feature_maps)
    invalid_maps = sorted(set(feature_maps) - {"reuploading", "zz"})
    if invalid_maps:
        raise ValueError(f"Feature maps inválidos: {invalid_maps}")
    qseeds = parse_int_list(args.quantum_seeds)
    data, load_metadata = load_dataset(input_csv)
    x = data[BASE_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = data[LABEL_COLUMN].to_numpy(dtype=int)
    class_counts = np.bincount(y, minlength=2)
    if args.n_folds > int(class_counts.min()):
        raise ValueError(f"--n-folds={args.n_folds} excede la clase minoritaria: {int(class_counts.min())}")
    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    metric_rows, prediction_frames = evaluate_classical_models(
        build_classical_models(args.seed), x, y, data, cv, top_ks
    )
    if not args.skip_quantum:
        quantum_configs = []
        for fmap in feature_maps:
            seeds_for_map = qseeds if fmap == "reuploading" else [0]
            for qseed in seeds_for_map:
                quantum_configs.append(QuantumFeatureMapConfig(name=fmap, n_layers=args.quantum_layers, seed=qseed))
        q_metrics, q_preds = evaluate_quantum_models(quantum_configs, x, y, data, cv, top_ks)
        metric_rows.extend(q_metrics)
        prediction_frames.extend(q_preds)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = build_summary(metrics)
    top_ranked = build_top_ranked(predictions, top_ks)
    top_ranked_by_fold = build_top_ranked_by_fold(predictions, top_ks)
    run_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "input_csv": str(input_csv.resolve()),
        "output_dir": str(output_dir),
        "dataset_arg": args.dataset,
        "dataset_resolved": dataset_tag,
        "n_rows": int(len(data)),
        **load_metadata,
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "feature_columns": BASE_FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "top_k": top_ks,
        "skip_quantum": args.skip_quantum,
        "quantum_feature_maps": feature_maps,
        "quantum_layers": args.quantum_layers,
        "quantum_seeds": qseeds,
        "primary_metric": "pr_auc_mean",
        "notes": (
            "Primary dataset should be matched controls; random controls are a selection-bias check. "
            "All preprocessing is fold-local. Metrics ending in test_threshold_oracle choose the threshold "
            "on the test fold and are diagnostic only. top_ranked_candidates uses raw out-of-fold scores; "
            "top_ranked_candidates_by_fold avoids cross-fold score-scale mixing. The zz feature map is deterministic."
        ),
    }
    save_outputs(
        output_dir,
        {
            "metrics_by_fold": metrics,
            "summary_metrics": summary,
            "predictions_by_fold": predictions,
            "top_ranked_candidates": top_ranked,
            "top_ranked_candidates_by_fold": top_ranked_by_fold,
        },
        run_config,
    )
    print(f"Salida escrita en: {output_dir}")
    print(summary.loc[:, ["model", "model_family", "pr_auc_mean", "roc_auc_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
