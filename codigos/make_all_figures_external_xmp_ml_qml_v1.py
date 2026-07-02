"""Ejecuta las figuras 01--08 de la Parte 3.

Comando:
    # python3 make_all_figures_external_xmp_ml_qml_v1.py --overwrite

Nota: plot_09 se excluye deliberadamente porque la prueba VQC fue negativa y
la carpeta de resultados fue eliminada del proyecto.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from plotting_external_xmp_ml_qml_utils_v1 import DEFAULT_FIGURE_DIR, SCRIPT_DIR, validate_output_dir

SCRIPTS = [
    "plot_01_pipeline_and_sdss_funnel_v1.py",
    "plot_02_external_catalog_composition_v1.py",
    "plot_03_matched_vs_random_controls_v1.py",
    "plot_04_main_matched_benchmark_v1.py",
    "plot_05_matched_vs_random_benchmark_bias_v1.py",
    "plot_06_qml_delta_by_feature_subset_v1.py",
    "plot_07_repeated_cv_qml_delta_v1.py",
    "plot_08_qml_topk_delta_v1.py",
]


def parse_args() -> argparse.Namespace:
    """Parsea argumentos del script maestro."""
    parser = argparse.ArgumentParser(description="Genera figuras 01--08 de Parte 3.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Ejecuta scripts secuencialmente y escribe log simple."""
    args = parse_args()
    output_dir = validate_output_dir(args.output_dir)
    log_lines = [f"Inicio UTC: {datetime.now(timezone.utc).isoformat()}", "plot_09 excluido por decisión del proyecto."]
    for script in SCRIPTS:
        path = SCRIPT_DIR / script
        if not path.exists():
            msg = f"MISSING {script}"
            print(msg)
            log_lines.append(msg)
            continue
        cmd = [sys.executable, str(path), "--output-dir", str(output_dir)]
        if args.overwrite:
            cmd.append("--overwrite")
        result = subprocess.run(cmd, cwd=SCRIPT_DIR, text=True, capture_output=True, check=False)
        status = "OK" if result.returncode == 0 else f"FAIL returncode={result.returncode}"
        print(f"{script}: {status}")
        log_lines.extend([f"\n## {script}: {status}", result.stdout.strip(), result.stderr.strip()])
    log_lines.append(f"Fin UTC: {datetime.now(timezone.utc).isoformat()}")
    log_path = output_dir / "figure_generation_log.txt"
    if log_path.exists() and not args.overwrite:
        raise FileExistsError(f"Ya existe {log_path}. Usa --overwrite para reemplazar.")
    log_path.write_text("\n".join(line for line in log_lines if line), encoding="utf-8")
    print(f"Generado: {log_path}")


if __name__ == "__main__":
    main()
