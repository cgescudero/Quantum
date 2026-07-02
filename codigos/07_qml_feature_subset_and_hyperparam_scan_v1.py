"""Barrido QML vs ML clásico por subconjuntos físicos de features.

Este script profundiza el benchmark del Script 06 usando sólo el dataset
pareado por defecto. El objetivo es probar si Quantum Kernel SVC mejora al
ML clásico cuando el espacio de entrada se reduce a subconjuntos físicos más
compactos y se barren hiperparámetros QML relevantes.

La métrica primaria es PR-AUC. Las métricas top-k se calculan fold-a-fold.
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_external_xmp_qml_feature_scan_v1"
SCRIPT_VERSION = "v1"
LABEL_COLUMN = "external_xmp_label"

FEATURE_SUBSETS = {
    "core_4": ["log_nii_ha", "log_oiii_hb", "log_o32", "delta_N2_O3N2"],
    "metallicity_5": ["log_nii_ha", "log_oiii_hb", "log_r23", "log_o32", "delta_N2_O3N2"],
    "oxygen_6": ["log_oiii_hb", "log_oii_hb", "log_r23", "log_o32", "o3n2", "delta_N2_O3N2"],
    "nebular_no_mass_z_8": [
        "log_nii_ha",
        "log_oiii_hb",
        "log_oii_hb",
        "log_sii_ha",
        "log_r23",
        "log_o32",
        "o3n2",
        "delta_N2_O3N2",
    ],
    "all_10": [
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
    ],
}


@dataclass(frozen=True)
class QuantumConfig:
    """Configuración del kernel cuántico reuploading."""

    n_layers: int
    seed: int
    c_value: float
    rotation: str = "Y"


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Barrido QML por subsets físicos para XMP externos.")
    parser.add_argument("--input", type=Path, default=MATCHED_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-subsets", type=str, default=",".join(FEATURE_SUBSETS))
    parser.add_argument("--quantum-layers", type=str, default="1,2,3,4,6")
    parser.add_argument("--quantum-c-values", type=str, default="0.3,1.0,3.0,10.0,30.0")
    parser.add_argument("--quantum-seeds", type=str, default="101,202,303")
    parser.add_argument("--top-k", type=str, default="20,50")
    parser.add_argument("--skip-quantum", action="store_true")
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument(
        "--max-quantum-configs",
        type=int,
        default=None,
        help="Límite opcional de configuraciones QML para pruebas rápidas.",
    )
    return parser.parse_args()


def parse_str_list(text: str) -> list[str]:
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


def parse_float_list(text: str) -> list[float]:
    """Parsea lista CSV de floats."""
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("La lista de floats no puede estar vacía.")
    if any(value <= 0 for value in values):
        raise ValueError("Todos los valores de C deben ser positivos.")
    return values


def validate_output_dir(output_dir: Path) -> Path:
    """Valida que la salida quede dentro de Parte_3."""
    resolved = output_dir.expanduser().resolve()
    allowed_root = SCRIPT_DIR.resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        raise ValueError(f"Por seguridad, --output-dir debe estar dentro de Parte_3: {allowed_root}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_feature_subsets(names: list[str]) -> dict[str, list[str]]:
    """Resuelve y valida subconjuntos de features."""
    invalid = sorted(set(names) - set(FEATURE_SUBSETS))
    if invalid:
        valid = ", ".join(FEATURE_SUBSETS)
        raise ValueError(f"Subconjuntos inválidos: {invalid}. Opciones válidas: {valid}")
    return {name: FEATURE_SUBSETS[name] for name in names}


def load_dataset(input_csv: Path, feature_subsets: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Carga dataset y filtra filas válidas para la unión de features usadas."""
    if not input_csv.exists():
        raise FileNotFoundError(f"No existe dataset de entrada: {input_csv}")
    data = pd.read_csv(input_csv)
    n_initial = len(data)
    feature_columns = sorted({feature for subset in feature_subsets.values() for feature in subset})
    required = set(feature_columns + [LABEL_COLUMN])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")
    keep_columns = [column for column in ["specobjid", "ra", "dec", "external_class", "sample_source"] if column in data.columns]
    keep_columns += feature_columns + [LABEL_COLUMN]
    data = data.loc[:, keep_columns].copy()
    for column in feature_columns + [LABEL_COLUMN]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    finite = np.isfinite(data[feature_columns]).all(axis=1)
    valid_label = data[LABEL_COLUMN].isin([0, 1])
    data = data.loc[finite & valid_label].reset_index(drop=True)
    n_removed = n_initial - len(data)
    if data[LABEL_COLUMN].nunique() != 2:
        raise ValueError("El dataset debe contener ambas clases tras el filtrado.")
    return data, {"n_rows_initial": int(n_initial), "n_rows_removed_invalid": int(n_removed)}


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
    """Obtiene score continuo para ranking binario."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x_test), dtype=float).ravel()
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x_test)[:, 1], dtype=float).ravel()
    return np.asarray(model.predict(x_test), dtype=float).ravel()


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Precision@k para scores donde mayor implica más positivo."""
    top = np.argsort(-scores)[: min(k, len(scores))]
    return float(np.mean(y_true[top])) if len(top) else np.nan


def recall_at_fraction(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    """Recall en el top fraction de objetos rankeados."""
    k = max(1, int(np.ceil(fraction * len(scores))))
    top = np.argsort(-scores)[:k]
    positives = np.sum(y_true == 1)
    return float(np.sum(y_true[top] == 1) / positives) if positives else np.nan


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, top_ks: list[int]) -> dict[str, float]:
    """Calcula métricas primarias de ranking."""
    metrics: dict[str, float] = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "recall_at_top5percent": recall_at_fraction(y_true, scores, 0.05),
    }
    for k in top_ks:
        metrics[f"precision_at_{k}"] = precision_at_k(y_true, scores, k)
    return metrics


def quantum_states_from_angles(x_angles: np.ndarray, n_layers: int, seed: int, rotation: str = "Y") -> np.ndarray:
    """Evalúa estados cuánticos del feature map reuploading."""
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except Exception as exc:  # pragma: no cover - depende del entorno local
        raise ImportError(f"No se pudo cargar PennyLane para ejecutar modelos QML: {type(exc).__name__}: {exc}") from exc

    n_qubits = x_angles.shape[1]
    rng = np.random.default_rng(seed + 97 * n_qubits + 1009 * n_layers)
    weights = pnp.array(rng.uniform(0.0, 2.0 * np.pi, size=(n_layers, n_qubits)), requires_grad=False)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x_row: pnp.ndarray) -> pnp.ndarray:
        for layer in range(n_layers):
            qml.AngleEmbedding(x_row, wires=range(n_qubits), rotation=rotation)
            qml.BasicEntanglerLayers(weights[layer : layer + 1], wires=range(n_qubits))
        qml.AngleEmbedding(x_row, wires=range(n_qubits), rotation="Z")
        return qml.state()

    return np.vstack([np.asarray(circuit(pnp.array(row, requires_grad=False)), dtype=complex) for row in x_angles])


def kernel_from_states(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    """Kernel de fidelidad |<phi(a)|phi(b)>|^2."""
    return np.asarray(np.abs(states_a @ np.conjugate(states_b.T)) ** 2, dtype=float)


def build_quantum_kernels(
    x_train: np.ndarray,
    x_test: np.ndarray,
    n_layers: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construye kernels train/test una sola vez para reutilizarlos con varios C."""
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
    cv: StratifiedKFold,
    top_ks: list[int],
    seed: int,
) -> list[dict[str, Any]]:
    """Evalúa modelos clásicos para un subconjunto de features."""
    x = data[features].to_numpy(dtype=float)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        for model_name, model_template in build_classical_models(seed).items():
            start = time.perf_counter()
            model = clone(model_template)
            model.fit(x_train, y_train)
            scores = decision_scores(model, x_test)
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "feature_subset": subset_name,
                    "n_features": len(features),
                    "features": ";".join(features),
                    "fold": fold,
                    "model": model_name,
                    "model_family": "classical",
                    "quantum_layers": np.nan,
                    "quantum_seed": np.nan,
                    "c_value": np.nan,
                    "fit_score_time_sec": elapsed,
                    **compute_metrics(y_test, scores, top_ks),
                }
            )
    return rows


def evaluate_quantum_models(
    subset_name: str,
    features: list[str],
    data: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
    layers: list[int],
    seeds: list[int],
    c_values: list[float],
    top_ks: list[int],
    max_configs: int | None,
) -> list[dict[str, Any]]:
    """Evalúa QK-SVC reuploading para un subconjunto de features."""
    x = data[features].to_numpy(dtype=float)
    rows = []
    config_counter = 0
    for n_layers in layers:
        for seed in seeds:
            config_counter += 1
            if max_configs is not None and config_counter > max_configs:
                return rows
            for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                print(
                    f"Subset {subset_name}: QK reuploading layers={n_layers} seed={seed} fold={fold}",
                    flush=True,
                )
                start_kernel = time.perf_counter()
                k_train, k_test = build_quantum_kernels(x_train, x_test, n_layers=n_layers, seed=seed)
                kernel_elapsed = time.perf_counter() - start_kernel
                for c_value in c_values:
                    start = time.perf_counter()
                    model = SVC(kernel="precomputed", C=c_value, class_weight="balanced")
                    model.fit(k_train, y_train)
                    scores = np.asarray(model.decision_function(k_test), dtype=float).ravel()
                    elapsed = kernel_elapsed + (time.perf_counter() - start)
                    rows.append(
                        {
                            "feature_subset": subset_name,
                            "n_features": len(features),
                            "features": ";".join(features),
                            "fold": fold,
                            "model": f"qksvc_reuploading_layers{n_layers}_seed{seed}_C{c_value:g}",
                            "model_family": "quantum_kernel",
                            "quantum_layers": n_layers,
                            "quantum_seed": seed,
                            "c_value": c_value,
                            "fit_score_time_sec": elapsed,
                            **compute_metrics(y_test, scores, top_ks),
                        }
                    )
    return rows


def build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Resume métricas por subset y configuración de modelo."""
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
    metric_cols = [col for col in metrics.columns if col not in set(group_cols + ["fold"])]
    rows = []
    for keys, subset in metrics.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_folds"] = int(len(subset))
        for col in metric_cols:
            if pd.api.types.is_numeric_dtype(subset[col]):
                row[f"{col}_mean"] = float(subset[col].mean())
                row[f"{col}_std"] = float(subset[col].std(ddof=1)) if len(subset) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)


def build_best_qml(summary: pd.DataFrame) -> pd.DataFrame:
    """Selecciona la mejor configuración QML por subset."""
    qml = summary.loc[summary["model_family"] == "quantum_kernel"].copy()
    if qml.empty:
        return pd.DataFrame()
    return qml.sort_values("pr_auc_mean", ascending=False).groupby("feature_subset", as_index=False).head(1).reset_index(drop=True)


def build_classical_vs_qml(summary: pd.DataFrame) -> pd.DataFrame:
    """Compara el mejor clásico y el mejor QML por subset."""
    rows = []
    for subset_name, subset in summary.groupby("feature_subset"):
        classical = subset.loc[subset["model_family"] == "classical"].sort_values("pr_auc_mean", ascending=False)
        quantum = subset.loc[subset["model_family"] == "quantum_kernel"].sort_values("pr_auc_mean", ascending=False)
        if classical.empty and quantum.empty:
            continue
        row: dict[str, Any] = {"feature_subset": subset_name}
        if not classical.empty:
            best_classical = classical.iloc[0]
            row.update(
                {
                    "best_classical_model": best_classical["model"],
                    "best_classical_pr_auc_mean": best_classical["pr_auc_mean"],
                    "best_classical_roc_auc_mean": best_classical["roc_auc_mean"],
                }
            )
        if not quantum.empty:
            best_quantum = quantum.iloc[0]
            row.update(
                {
                    "best_quantum_model": best_quantum["model"],
                    "best_quantum_pr_auc_mean": best_quantum["pr_auc_mean"],
                    "best_quantum_roc_auc_mean": best_quantum["roc_auc_mean"],
                    "best_quantum_layers": best_quantum["quantum_layers"],
                    "best_quantum_seed": best_quantum["quantum_seed"],
                    "best_quantum_c_value": best_quantum["c_value"],
                }
            )
        if "best_classical_pr_auc_mean" in row and "best_quantum_pr_auc_mean" in row:
            row["delta_qml_minus_classical_pr_auc"] = row["best_quantum_pr_auc_mean"] - row["best_classical_pr_auc_mean"]
            row["delta_qml_minus_classical_roc_auc"] = row["best_quantum_roc_auc_mean"] - row["best_classical_roc_auc_mean"]
        rows.append(row)
    result = pd.DataFrame(rows)
    if "delta_qml_minus_classical_pr_auc" in result.columns:
        return result.sort_values("delta_qml_minus_classical_pr_auc", ascending=False, na_position="last")
    return result


def save_outputs(output_dir: Path, outputs: dict[str, pd.DataFrame], run_config: dict[str, Any]) -> None:
    """Guarda tablas y configuración."""
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Ejecuta el barrido completo."""
    args = parse_args()
    output_dir = validate_output_dir(args.output_dir)
    subset_names = parse_str_list(args.feature_subsets)
    feature_subsets = resolve_feature_subsets(subset_names)
    layers = parse_int_list(args.quantum_layers)
    if any(layer <= 0 for layer in layers):
        raise ValueError("Todas las capas cuánticas deben ser positivas.")
    c_values = parse_float_list(args.quantum_c_values)
    quantum_seeds = parse_int_list(args.quantum_seeds)
    top_ks = parse_int_list(args.top_k)
    data, load_metadata = load_dataset(args.input, feature_subsets)
    y = data[LABEL_COLUMN].to_numpy(dtype=int)
    class_counts = np.bincount(y, minlength=2)
    if args.n_folds > int(class_counts.min()):
        raise ValueError(f"--n-folds={args.n_folds} excede la clase minoritaria: {int(class_counts.min())}")
    if args.skip_classical and args.skip_quantum:
        raise ValueError("No se puede usar --skip-classical y --skip-quantum al mismo tiempo.")
    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    n_qml_configs = len(feature_subsets) * len(layers) * len(quantum_seeds) * len(c_values)
    if args.max_quantum_configs is not None:
        n_qml_configs = min(n_qml_configs, args.max_quantum_configs * len(feature_subsets) * len(c_values))
    print(f"Dataset: {args.input}", flush=True)
    print(f"Subconjuntos: {', '.join(feature_subsets)}", flush=True)
    if not args.skip_quantum:
        print(
            f"QML: layers={layers}, C={c_values}, seeds={quantum_seeds}; configs modelo aprox={n_qml_configs}",
            flush=True,
        )

    metric_rows: list[dict[str, Any]] = []
    for subset_name, features in feature_subsets.items():
        if not args.skip_classical:
            metric_rows.extend(evaluate_classical_models(subset_name, features, data, y, cv, top_ks, args.seed))
        if not args.skip_quantum:
            metric_rows.extend(
                evaluate_quantum_models(
                    subset_name,
                    features,
                    data,
                    y,
                    cv,
                    layers,
                    quantum_seeds,
                    c_values,
                    top_ks,
                    args.max_quantum_configs,
                )
            )

    metrics = pd.DataFrame(metric_rows)
    summary = build_summary(metrics)
    best_qml = build_best_qml(summary)
    classical_vs_qml = build_classical_vs_qml(summary)
    run_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "input_csv": str(args.input.resolve()),
        "output_dir": str(output_dir),
        "n_rows": int(len(data)),
        **load_metadata,
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "feature_subsets": feature_subsets,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "quantum_feature_map": "reuploading",
        "quantum_layers": layers,
        "quantum_c_values": c_values,
        "quantum_seeds": quantum_seeds,
        "top_k": top_ks,
        "skip_quantum": args.skip_quantum,
        "skip_classical": args.skip_classical,
        "max_quantum_configs": args.max_quantum_configs,
        "primary_metric": "pr_auc_mean",
        "notes": (
            "Use matched controls as the primary benchmark. Quantum kernels are computed once per "
            "subset/fold/layer/seed and reused across C values. A QML gain should be judged mainly by "
            "PR-AUC and top-k metrics, not ROC-AUC alone."
        ),
    }
    save_outputs(
        output_dir,
        {
            "metrics_by_fold": metrics,
            "summary_by_feature_subset": summary,
            "best_qml_configs": best_qml,
            "matched_best_classical_vs_qml": classical_vs_qml,
        },
        run_config,
    )
    print(f"Salida escrita en: {output_dir}")
    columns = ["feature_subset", "model", "model_family", "pr_auc_mean", "roc_auc_mean"]
    print(summary.loc[:, columns].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
