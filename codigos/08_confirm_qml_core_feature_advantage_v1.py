"""Confirmación robusta de la señal QML en subconjuntos compactos.

Este script toma la señal exploratoria del Script 07 y la somete a repeated
stratified CV usando sólo los subconjuntos ``core_4`` y ``metallicity_5``. El
espacio QML está predefinido y acotado: feature map reuploading, capas 1--2,
C=10--30 y múltiples semillas del feature map. La métrica primaria es PR-AUC.

No usa el dataset con controles aleatorios por defecto, porque ese control ya
mostró sesgo de selección fuerte. La comparación confirmatoria se hace contra
el mejor clásico disponible dentro de cada ``cv_seed``.
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_external_xmp_qml_confirm_core_features_v1"
SCRIPT_VERSION = "v1"
LABEL_COLUMN = "external_xmp_label"

FEATURE_SUBSETS = {
    "core_4": ["log_nii_ha", "log_oiii_hb", "log_o32", "delta_N2_O3N2"],
    "metallicity_5": ["log_nii_ha", "log_oiii_hb", "log_r23", "log_o32", "delta_N2_O3N2"],
}

ID_COLUMNS = ["specobjid", "ra", "dec", "external_class", "sample_source"]
PRIMARY_METRIC = "pr_auc"


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Confirmación robusta QML para core_4/metallicity_5.")
    parser.add_argument("--input", type=Path, default=MATCHED_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--cv-seeds", type=str, default="11,22,33,44,55,66,77,88,99,111")
    parser.add_argument("--feature-subsets", type=str, default="core_4,metallicity_5")
    parser.add_argument("--classical-seed", type=int, default=42)
    parser.add_argument("--quantum-layers", type=str, default="1,2")
    parser.add_argument("--quantum-c-values", type=str, default="10,30")
    parser.add_argument("--quantum-seeds", type=str, default="101,202,303,404,505,606,707,808,909,1001")
    parser.add_argument("--top-k", type=str, default="20,50")
    parser.add_argument("--skip-quantum", action="store_true")
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


def parse_float_list(text: str) -> list[float]:
    """Convierte una lista separada por comas en floats positivos."""
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("La lista de floats no puede estar vacía.")
    if any(value <= 0.0 for value in values):
        raise ValueError("Todos los valores de C deben ser positivos.")
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
    """Carga dataset, valida columnas, etiquetas y valores finitos.

    Parameters
    ----------
    input_csv:
        CSV supervisado producido por el Script 05.
    feature_subsets:
        Diccionario de subconjuntos de features a evaluar.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        Tabla filtrada y metadatos de filas descartadas.
    """
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
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced")),
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
    metrics = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "recall_at_top5percent": recall_at_fraction(y_true, scores, 0.05),
    }
    for k in top_ks:
        metrics[f"precision_at_{k}"] = precision_at_k(y_true, scores, k)
    return metrics


def quantum_states_from_angles(x_angles: np.ndarray, n_layers: int, seed: int) -> np.ndarray:
    """Evalúa estados cuánticos del feature map reuploading.

    El embedding transforma features escaladas a ángulos en [0, pi]. Cada capa
    aplica AngleEmbedding-Y, BasicEntanglerLayers con pesos fijos por semilla y
    un AngleEmbedding-Z final antes de medir el estado completo.
    """
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except ImportError as exc:
        raise ImportError(f"No se pudo cargar PennyLane para ejecutar QML: {exc}") from exc

    n_qubits = x_angles.shape[1]
    rng = np.random.default_rng(seed + 97 * n_qubits + 1009 * n_layers)
    weights = pnp.array(rng.uniform(0.0, 2.0 * np.pi, size=(n_layers, n_qubits)), requires_grad=False)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x_row: pnp.ndarray) -> pnp.ndarray:
        for layer in range(n_layers):
            qml.AngleEmbedding(x_row, wires=range(n_qubits), rotation="Y")
            qml.BasicEntanglerLayers(weights[layer : layer + 1], wires=range(n_qubits))
        qml.AngleEmbedding(x_row, wires=range(n_qubits), rotation="Z")
        return qml.state()

    return np.vstack([np.asarray(circuit(pnp.array(row, requires_grad=False)), dtype=complex) for row in x_angles])


def kernel_from_states(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    """Construye kernel de fidelidad |<phi(a)|phi(b)>|^2."""
    return np.asarray(np.abs(states_a @ np.conjugate(states_b.T)) ** 2, dtype=float)


def build_quantum_kernels(
    x_train: np.ndarray,
    x_test: np.ndarray,
    n_layers: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Escala features dentro del fold y construye kernels train/test."""
    scaler = MinMaxScaler(feature_range=(0.0, np.pi)).fit(x_train)
    train_angles = np.clip(scaler.transform(x_train), 0.0, np.pi)
    test_angles = np.clip(scaler.transform(x_test), 0.0, np.pi)
    states_train = quantum_states_from_angles(train_angles, n_layers=n_layers, seed=seed)
    states_test = quantum_states_from_angles(test_angles, n_layers=n_layers, seed=seed)
    k_train = kernel_from_states(states_train, states_train)
    k_train = 0.5 * (k_train + k_train.T)
    np.fill_diagonal(k_train, 1.0)
    k_test = kernel_from_states(states_test, states_train)
    return k_train, k_test


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
                    "quantum_layers": np.nan,
                    "quantum_seed": np.nan,
                    "c_value": np.nan,
                    "fit_score_time_sec": time.perf_counter() - start,
                    **compute_metrics(y_test, scores, top_ks),
                }
            )
    return rows


def evaluate_quantum_models(
    subset_name: str,
    features: list[str],
    data: pd.DataFrame,
    y: np.ndarray,
    cv_seed: int,
    n_folds: int,
    layers: list[int],
    seeds: list[int],
    c_values: list[float],
    top_ks: list[int],
) -> list[dict[str, Any]]:
    """Evalúa QK-SVC reuploading para un subconjunto y una partición CV."""
    x = data[features].to_numpy(dtype=float)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
    rows = []
    for n_layers in layers:
        for q_seed in seeds:
            for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                print(
                    f"Subset={subset_name} cv_seed={cv_seed} QK layers={n_layers} seed={q_seed} fold={fold}",
                    flush=True,
                )
                start_kernel = time.perf_counter()
                k_train, k_test = build_quantum_kernels(x_train, x_test, n_layers=n_layers, seed=q_seed)
                kernel_elapsed = time.perf_counter() - start_kernel
                for c_value in c_values:
                    start = time.perf_counter()
                    model = SVC(kernel="precomputed", C=c_value, class_weight="balanced")
                    model.fit(k_train, y_train)
                    scores = np.asarray(model.decision_function(k_test), dtype=float).ravel()
                    rows.append(
                        {
                            "feature_subset": subset_name,
                            "n_features": len(features),
                            "features": ";".join(features),
                            "cv_seed": cv_seed,
                            "fold": fold,
                            "model": f"qksvc_reuploading_layers{n_layers}_seed{q_seed}_C{c_value:g}",
                            "model_family": "quantum_kernel",
                            "quantum_layers": n_layers,
                            "quantum_seed": q_seed,
                            "c_value": c_value,
                            "fit_score_time_sec": kernel_elapsed + (time.perf_counter() - start),
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
        "quantum_layers",
        "quantum_seed",
        "c_value",
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
    """Calcula delta QML-clásico por subconjunto y cv_seed.

    Dentro de cada cv_seed se promedia por folds, se elige el mejor clásico y el
    mejor QML según PR-AUC medio, y se calcula el delta en métricas primarias y
    secundarias. Esto mide estabilidad frente a cambios de partición.
    """
    rows = []
    metric_cols = [col for col in metrics.columns if col.startswith("precision_at_")]
    metric_cols += ["pr_auc", "roc_auc", "recall_at_top5percent"]
    for (subset_name, cv_seed), subset in metrics.groupby(["feature_subset", "cv_seed"]):
        group_cols = ["model", "model_family", "quantum_layers", "quantum_seed", "c_value"]
        averaged = subset.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
        classical = averaged.loc[averaged["model_family"] == "classical"].sort_values("pr_auc", ascending=False)
        quantum = averaged.loc[averaged["model_family"] == "quantum_kernel"].sort_values("pr_auc", ascending=False)
        if classical.empty or quantum.empty:
            continue
        best_classical = classical.iloc[0]
        best_quantum = quantum.iloc[0]
        row: dict[str, Any] = {
            "feature_subset": subset_name,
            "cv_seed": int(cv_seed),
            "best_classical_model": best_classical["model"],
            "best_quantum_model": best_quantum["model"],
            "best_quantum_layers": best_quantum["quantum_layers"],
            "best_quantum_seed": best_quantum["quantum_seed"],
            "best_quantum_c_value": best_quantum["c_value"],
        }
        for col in metric_cols:
            row[f"best_classical_{col}"] = float(best_classical[col])
            row[f"best_quantum_{col}"] = float(best_quantum[col])
            row[f"delta_qml_minus_classical_{col}"] = float(best_quantum[col] - best_classical[col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["feature_subset", "cv_seed"]).reset_index(drop=True)


def build_best_qml_configs_by_cv_seed(delta_by_seed: pd.DataFrame) -> pd.DataFrame:
    """Extrae la mejor configuración QML seleccionada en cada cv_seed."""
    columns = [
        "feature_subset",
        "cv_seed",
        "best_quantum_model",
        "best_quantum_layers",
        "best_quantum_seed",
        "best_quantum_c_value",
        "best_quantum_pr_auc",
        "best_quantum_roc_auc",
    ]
    available = [column for column in columns if column in delta_by_seed.columns]
    return delta_by_seed.loc[:, available].copy()


def build_delta_distribution_summary(delta_by_seed: pd.DataFrame) -> pd.DataFrame:
    """Resume distribución de deltas QML-clásico por subconjunto."""
    rows = []
    for subset_name, subset in delta_by_seed.groupby("feature_subset"):
        deltas = subset["delta_qml_minus_classical_pr_auc"].to_numpy(dtype=float)
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
            delta_col = f"delta_qml_minus_classical_{col}"
            if delta_col in subset.columns:
                row[f"delta_{col}_mean"] = float(subset[delta_col].mean())
                row[f"delta_{col}_median"] = float(subset[delta_col].median())
        precision_cols = [col for col in subset.columns if col.startswith("delta_qml_minus_classical_precision_at_")]
        for col in precision_cols:
            suffix = col.replace("delta_qml_minus_classical_", "")
            row[f"delta_{suffix}_mean"] = float(subset[col].mean())
            row[f"delta_{suffix}_median"] = float(subset[col].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("delta_pr_auc_mean", ascending=False).reset_index(drop=True)


def build_confirmatory_decision(delta_summary: pd.DataFrame) -> pd.DataFrame:
    """Aplica criterios predefinidos de confirmación de señal QML."""
    rows = []
    for _, row in delta_summary.iterrows():
        precision_loss = False
        precision_cols = [col for col in delta_summary.columns if col.startswith("delta_precision_at_") and col.endswith("_mean")]
        for col in precision_cols:
            if float(row[col]) < -0.02:
                precision_loss = True
        confirmed = (
            float(row["delta_pr_auc_median"]) > 0.0
            and float(row["delta_pr_auc_mean"]) >= 0.02
            and float(row["fraction_positive_delta"]) >= 0.70
            and not precision_loss
        )
        rows.append(
            {
                "feature_subset": row["feature_subset"],
                "qml_signal_confirmed": bool(confirmed),
                "criterion_delta_pr_auc_median_gt_0": bool(float(row["delta_pr_auc_median"]) > 0.0),
                "criterion_delta_pr_auc_mean_ge_0p02": bool(float(row["delta_pr_auc_mean"]) >= 0.02),
                "criterion_fraction_positive_ge_0p70": bool(float(row["fraction_positive_delta"]) >= 0.70),
                "criterion_no_clear_precision_at_k_loss": bool(not precision_loss),
                "interpretation": (
                    "señal QML confirmada bajo criterios predefinidos"
                    if confirmed
                    else "señal QML no confirmada; tratar como exploratoria o no robusta"
                ),
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
    """Ejecuta la validación confirmatoria robusta."""
    args = parse_args()
    if args.skip_classical and args.skip_quantum:
        raise ValueError("No se puede usar --skip-classical y --skip-quantum al mismo tiempo.")
    output_dir = validate_output_dir(args.output_dir)
    feature_subsets = resolve_feature_subsets(parse_str_list(args.feature_subsets))
    cv_seeds = parse_int_list(args.cv_seeds)
    quantum_layers = parse_int_list(args.quantum_layers)
    quantum_seeds = parse_int_list(args.quantum_seeds)
    quantum_c_values = parse_float_list(args.quantum_c_values)
    top_ks = parse_int_list(args.top_k)
    if any(layer <= 0 for layer in quantum_layers):
        raise ValueError("Todas las capas cuánticas deben ser positivas.")
    if args.n_folds < 2:
        raise ValueError("--n-folds debe ser >= 2.")

    data, load_metadata = load_dataset(args.input, feature_subsets)
    y = data[LABEL_COLUMN].to_numpy(dtype=int)
    class_counts = np.bincount(y, minlength=2)
    if args.n_folds > int(class_counts.min()):
        raise ValueError(f"--n-folds={args.n_folds} excede la clase minoritaria: {int(class_counts.min())}")

    print(f"Dataset: {args.input}", flush=True)
    print(f"Filas válidas: {len(data)}; positivos={int(class_counts[1])}; negativos={int(class_counts[0])}", flush=True)
    print(f"Subconjuntos: {', '.join(feature_subsets)}", flush=True)
    print(f"CV seeds: {cv_seeds}", flush=True)
    if not args.skip_quantum:
        n_kernels = len(feature_subsets) * len(cv_seeds) * len(quantum_layers) * len(quantum_seeds) * args.n_folds
        print(
            f"QML robusto: layers={quantum_layers}, C={quantum_c_values}, "
            f"q_seeds={quantum_seeds}; kernels a calcular={n_kernels}",
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
            if not args.skip_quantum:
                metric_rows.extend(
                    evaluate_quantum_models(
                        subset_name=subset_name,
                        features=features,
                        data=data,
                        y=y,
                        cv_seed=cv_seed,
                        n_folds=args.n_folds,
                        layers=quantum_layers,
                        seeds=quantum_seeds,
                        c_values=quantum_c_values,
                        top_ks=top_ks,
                    )
                )

    metrics = pd.DataFrame(metric_rows)
    summary_by_model = build_summary_by_model(metrics)
    delta_by_cv_seed = build_delta_by_cv_seed(metrics)
    best_qml_by_cv_seed = build_best_qml_configs_by_cv_seed(delta_by_cv_seed)
    delta_summary = build_delta_distribution_summary(delta_by_cv_seed)
    confirmatory_decision = build_confirmatory_decision(delta_summary)
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
        "quantum_feature_map": "reuploading",
        "quantum_layers": quantum_layers,
        "quantum_c_values": quantum_c_values,
        "quantum_seeds": quantum_seeds,
        "top_k": top_ks,
        "primary_metric": PRIMARY_METRIC,
        "confirmation_criteria": {
            "delta_pr_auc_median_gt_0": True,
            "delta_pr_auc_mean_ge_0p02": True,
            "fraction_positive_delta_ge_0p70": True,
            "no_clear_precision_at_k_loss": "mean delta precision@k >= -0.02 for every reported k",
        },
        "skip_quantum": args.skip_quantum,
        "skip_classical": args.skip_classical,
        "notes": (
            "Confirmatory script for the exploratory QML signal in core_4 and metallicity_5. "
            "Deltas are computed by cv_seed against the best classical model selected within the same repeated CV split. "
            "The matched-control dataset is the scientifically relevant benchmark."
        ),
    }
    save_outputs(
        output_dir,
        {
            "metrics_by_fold": metrics,
            "summary_by_model": summary_by_model,
            "delta_by_cv_seed": delta_by_cv_seed,
            "delta_distribution_summary": delta_summary,
            "best_qml_configs_by_cv_seed": best_qml_by_cv_seed,
            "confirmatory_decision": confirmatory_decision,
        },
        run_config,
    )
    print(f"Salida escrita en: {output_dir}")
    print(delta_summary.to_string(index=False))
    print(confirmatory_decision.to_string(index=False))


if __name__ == "__main__":
    main()
