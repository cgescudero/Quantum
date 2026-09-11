import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# Qiskit Aer noise
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit.providers.aer.noise import thermal_relaxation_error

import os

# 1. Data
csv_path = 'sdss_galaxies_spectroscopy.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

data = pd.read_csv(csv_path)
target_column = 'metallicity'
feature_columns = ['h_alpha_flux', 'h_beta_flux', 'oiii_5007_flux', 'nii_6584_flux',
                   'oii_3726_flux', 'oii_3729_flux', 'sii_6717_flux', 'sii_6731_flux',
                   'log_stellar_mass', 'z']
data = data.dropna(subset=feature_columns + [target_column]).reset_index(drop=True)

# scaling
scaler_f = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler_f.fit_transform(data[feature_columns])
features_reduced = PCA(n_components=4).fit_transform(features_scaled)
features_reduced = features_reduced * 2 * np.pi

scaler_t = MinMaxScaler()
target = scaler_t.fit_transform(data[[target_column]]).flatten()

# 2. Noise model (approximate realistic device noise)
noise_model = NoiseModel()
# Depolarizing errors
single_qubit_error = depolarizing_error(0.005, 1)
two_qubit_error = depolarizing_error(0.02, 2)
noise_model.add_all_qubit_quantum_error(single_qubit_error, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'x', 'sx'])
noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx', 'cz'])
# Readout error (confusion matrix)
readout_error = ReadoutError([[0.98, 0.02], [0.03, 0.97]])
noise_model.add_all_qubit_readout_error(readout_error)

# 3. Model
n_qubits = 4
# Use Qiskit Aer device with noise
dev = qml.device("qiskit.aer", wires=n_qubits, backend='aer_simulator', shots=2048, noise_model=noise_model)

@qml.qnode(dev)
def quantum_circuit(params, x):
    qml.AngleEmbedding(x[:4], wires=range(n_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

class VQR_Model:
    def __init__(self, n_qubits, learning_rate=0.005, n_layers=3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = np.random.uniform(0, 0.5, (self.n_layers, n_qubits, 3), requires_grad=True)
        self.learning_rate = learning_rate

    def fit(self, X, y, epochs=50):
        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        X_t = np.array(X)
        y_t = np.array(y)
        batch_size = 16

        for epoch in range(epochs):
            idx = np.random.choice(len(X_t), batch_size, replace=True)
            X_batch = X_t[idx]
            y_batch = y_t[idx]

            def cost(p):
                preds = [quantum_circuit(p, x) for x in X_batch]
                return np.mean((np.array(preds) - y_batch) ** 2)

            self.params = opt.step(cost, self.params)

            if (epoch+1) % 10 == 0:
                current_loss = cost(self.params)
                print(f"  -> Época {epoch+1}: Loss = {current_loss:.6f}")

    def predict(self, X):
        preds = np.array([quantum_circuit(self.params, x) for x in X])
        return preds.flatten()

# 4. Run a single promising config (Config 2 from tuning)
config = {"lr": 0.005, "layers": 3, "epochs": 50}

kf = KFold(n_splits=3, shuffle=True, random_state=42)
print("Running noise-aware simulation for Config 2 (shortened epochs for speed)")

r2_quantum = []
r2_classical = []

for fold, (train_idx, test_idx) in enumerate(kf.split(features_reduced)):
    print(f"\n--- Fold {fold+1} ---")

    model = VQR_Model(n_qubits=n_qubits, learning_rate=config['lr'], n_layers=config['layers'])
    model.fit(features_reduced[train_idx], target[train_idx], epochs=config['epochs'])
    y_pred_q = model.predict(features_reduced[test_idx])

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(features_reduced[train_idx], target[train_idx])
    y_pred_rf = rf.predict(features_reduced[test_idx])

    r2_q = r2_score(target[test_idx], y_pred_q)
    r2_c = r2_score(target[test_idx], y_pred_rf)

    r2_quantum.append(r2_q)
    r2_classical.append(r2_c)

    print(f"  R2 Quantum : {r2_q:.4f}")
    print(f"  R2 Classical: {r2_c:.4f}")

print("\nAverages:")
print(f"  R2 Quantum (mean) : {np.mean(r2_quantum):.4f}")
print(f"  R2 Classical(mean): {np.mean(r2_classical):.4f}")
