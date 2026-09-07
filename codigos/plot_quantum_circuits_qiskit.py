#!/usr/bin/env python3
"""Generador y visualizador de circuitos cuánticos con Qiskit.

Este script construye y grafica los circuitos cuánticos evaluados en la suite
del proyecto:
  1. Data Re-uploading Feature Map (QK-SVC, Script 06/07/08)
  2. ZZ Feature Map con interacciones no lineales tipo Ising (QK-SVC, Script 06/07/08)
  3. Variational Quantum Classifier / Regressor (VQC/VQR, Scripts 02 y 09)

Permite guardar los diagramas en alta resolución (.png y .pdf) y mostrarlos
en consola en formato texto/ASCII.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "graficos" / "circuitos"

FEATURE_NAMES_CORE4 = [
    r"x_0: [NII]/H\alpha",
    r"x_1: [OIII]/H\beta",
    r"x_2: O_{32}",
    r"x_3: \Delta N2-O3N2",
]


def create_param_labels(n_qubits: int, use_astrophysical_labels: bool) -> Sequence[Parameter]:
    """Crea los parámetros de entrada simbólicos para el circuito."""
    if use_astrophysical_labels and n_qubits == 4:
        return [Parameter(f"x_{i}") for i in range(4)]
    return ParameterVector("x", length=n_qubits)


def build_reuploading_circuit(
    n_qubits: int = 4,
    n_layers: int = 1,
    use_astrophysical_labels: bool = True,
) -> QuantumCircuit:
    """Construye el circuito cuántico de Data Re-uploading usado en QK-SVC.

    Estructura por capa:
      1. R_Y(x_i)
      2. R_X(theta_{l, i}) (rotación fija con pesos de semilla)
      3. Anillo de CNOTs (0->1, 1->2, ..., n-1->0)
    Capa final:
      R_Z(x_i)
    """
    x = create_param_labels(n_qubits, use_astrophysical_labels)
    qc = QuantumCircuit(n_qubits, name=f"Reuploading_L{n_layers}_Q{n_qubits}") 

    for layer in range(n_layers):
        #qc.barrier(label=f"Layer {layer+1}:  Y-Encoding")# +  X-Parametrized Rotations + CNOT Entanglement")
        for i in range(n_qubits):
            x_param = Parameter(rf"$x_{{{i}}}$")
            qc.ry(x_param, i)

        qc.barrier()

        # Rotaciones fijas parametrizadas por la semilla
        for i in range(n_qubits):
            theta_param = Parameter(rf"$\theta_{{{layer},{i}}}$")
            qc.rx(theta_param, i)

        qc.barrier()

        # Entrelazamiento circular básico
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        if n_qubits > 2:
            qc.cx(n_qubits - 1, 0)

    #qc.barrier(label="Final Z-Re-embedding")
    qc.barrier()
    for i in range(n_qubits):
        x_param = Parameter(rf"$x_{{{i}}}$")
        qc.rz(x[i], i)

    return qc


def build_zz_feature_map_circuit(
    n_qubits: int = 4,
    n_layers: int = 1,
    use_astrophysical_labels: bool = True,
) -> QuantumCircuit:
    """Construye el circuito ZZ Feature Map (Havlíček et al.) usado en QK-SVC.

    Estructura:
      - Inicialización: R_Y(x_i)
      - Por capa:
          * Monomios cuadráticos: R_Z(x_i^2)
          * Interacción ZZ adyacente: CNOT(i, i+1) -> R_Z(x_i * x_{i+1}) -> CNOT(i, i+1)
          * Cierre cíclico ZZ: CNOT(n-1, 0) -> R_Z(x_{n-1} * x_0) -> CNOT(n-1, 0)
      - Capa de mezcla final: R_X(x_i)
    """
    x = create_param_labels(n_qubits, use_astrophysical_labels)
    qc = QuantumCircuit(n_qubits, name=f"ZZ_FeatureMap_L{n_layers}_Q{n_qubits}")

    #qc.barrier(label="Initial Y-Embedding")
    for i in range(n_qubits):
        #qc.ry(x[i], i)
        x_param = Parameter(rf"$x_{{{i}}}$")
        qc.ry(x_param, i)

    for layer in range(n_layers):
        #qc.barrier(label=f"Layer {layer+1}: Quadratic Phases R_Z(x_i^2)")
        qc.barrier()
        for i in range(n_qubits):
            # Fase monomio x_i^2
            sq_param = Parameter(rf"$x_{{{i}}}^2$")
            qc.rz(sq_param, i)

        #qc.barrier(label=f"Layer {layer+1}: ZZ Interacciones: exp(-i x_i x_j Z_i Z_j)")
        qc.barrier()
        # Entrelazamiento ZZ adyacente
        for i in range(n_qubits - 1):
            cross_param = Parameter(rf"$x_{{{i}}} x_{{{i+1}}}$")
            qc.cx(i, i + 1)
            qc.rz(cross_param, i + 1)
            qc.cx(i, i + 1)

        # Cierre cíclico ZZ
        if n_qubits > 2:
            cyclic_param = Parameter(rf"$x_{{{n_qubits-1}}} x_{{0}}$")
            qc.cx(n_qubits - 1, 0)
            qc.rz(cyclic_param, 0)
            qc.cx(n_qubits - 1, 0)

    #qc.barrier(label="Final X-Mixing")
    qc.barrier()
    for i in range(n_qubits):
        qc.rx(x[i], i)

    return qc


def build_vqc_vqr_circuit(
    n_qubits: int = 4,
    n_layers: int = 1,
    use_astrophysical_labels: bool = True,
) -> QuantumCircuit:
    """Construye el circuito Variacional (VQC / VQR) con Strongly Entangling Layers.

    Estructura:
      - Carga de datos: R_Y(x_i)
      - Capas fuertemente entrelazantes:
          * Rotaciones de Euler U3(phi, theta, omega)
          * Entrelazamiento CNOT circular
      - Medición observable: Medición Pauli Z sobre q_0 (<Z_0>)
    """
    x = create_param_labels(n_qubits, use_astrophysical_labels)
    qc = QuantumCircuit(n_qubits, 1, name=f"VQC_VQR_L{n_layers}_Q{n_qubits}")

    #qc.barrier(label="Initial Y-Embedding")
    for i in range(n_qubits):
        x_param = Parameter(rf"$x_{{{i}}}$")
        qc.ry(x[i], i)

    for layer in range(n_layers):
        #qc.barrier(label=f"Layer {layer+1}: Variational Rotations R_U3(\\phi, \\theta, \\omega)")
        qc.barrier()
        for i in range(n_qubits):
            phi = Parameter(rf"$\phi_{{{layer},{i}}}$")
            theta = Parameter(rf"$\theta_{{{layer},{i}}}$")
            omega = Parameter(rf"$\omega_{{{layer},{i}}}$")
            qc.u(theta, phi, omega, i)

        # CNOT circular
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        if n_qubits > 2:
            qc.cx(n_qubits - 1, 0)

    #qc.barrier(label="Final Observable Measurement <Z_0>")
    qc.barrier()   
    qc.measure(0, 0)

    return qc


def save_and_plot_circuit(
    qc: QuantumCircuit,
    filename_base: str,
    output_dir: Path,
    style: str = "iqp",
) -> None:
    """Grafica el circuito usando Matplotlib y lo guarda en PNG y PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{filename_base}.png"
    pdf_path = output_dir / f"{filename_base}.pdf"


    ###print(f"\n========================================================")
    print(f"Circuito: {qc.name}")
    print(f"========================================================")
    # Visualización en consola (ASCII)
    print(qc.draw(output="text", fold=120))

    try:
        import matplotlib.pyplot as plt

        fig = qc.draw(
            output="mpl",
            style=style,
            fold=25,
            plot_barriers=True,
            initial_state=True,
        )
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"-> Imagen guardada exitosamente:")
        print(f"   PNG: {png_path}")
        print(f"   PDF: {pdf_path}")
    except Exception as exc:
        print(f"Nota: No se pudo renderizar con Matplotlib ({exc}). El circuito se mostró en modo texto.")

###
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graficador de Circuitos Cuánticos con Qiskit.")
    parser.add_argument("--qubits", type=int, default=4, help="Número de qubits (por defecto: 4 para core_4).")
    parser.add_argument("--layers", type=int, default=1, help="Número de capas para visualizar (1 o 2 recomendado para gráficos claros).")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Carpeta destino para los diagramas.")
    parser.add_argument("--style", type=str, default="iqp", choices=["iqp", "clifford", "default"], help="Estilo visual de Qiskit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    n_qubits = args.qubits
    n_layers = args.layers

    print(f"Generando circuitos con N={n_qubits} qubits y L={n_layers} capas...")

    # 1. Circuito Data Re-uploading (QK-SVC)
    qc_reup = build_reuploading_circuit(n_qubits=n_qubits, n_layers=n_layers)
    save_and_plot_circuit(qc_reup, f"circuito_01_reuploading_q{n_qubits}_l{n_layers}", output_dir, style=args.style)

    # 2. Circuito ZZ Feature Map (QK-SVC)
    qc_zz = build_zz_feature_map_circuit(n_qubits=n_qubits, n_layers=n_layers)
    save_and_plot_circuit(qc_zz, f"circuito_02_zz_feature_map_q{n_qubits}_l{n_layers}", output_dir, style=args.style)

    # 3. Circuito Variacional VQC / VQR
    qc_vqc = build_vqc_vqr_circuit(n_qubits=n_qubits, n_layers=n_layers)
    save_and_plot_circuit(qc_vqc, f"circuito_03_variacional_vqc_vqr_q{n_qubits}_l{n_layers}", output_dir, style=args.style)

    print(f"\n¡Listo! Todos los circuitos fueron generados en: {output_dir}")


if __name__ == "__main__":
    main()
