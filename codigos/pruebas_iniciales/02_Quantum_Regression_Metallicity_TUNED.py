import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import os

# 1. Preparación de datos
csv_path = 'sdss_galaxies_spectroscopy.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

data = pd.read_csv(csv_path)
target_column = 'metallicity'
feature_columns = ['h_alpha_flux', 'h_beta_flux', 'oiii_5007_flux', 'nii_6584_flux',
                   'oii_3726_flux', 'oii_3729_flux', 'sii_6717_flux', 'sii_6731_flux',
                   'log_stellar_mass', 'z']
data = data.dropna(subset=feature_columns + [target_column]).reset_index(drop=True)

# Scale features to [0, 1] for angle embedding
scaler_f = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler_f.fit_transform(data[feature_columns])
features_reduced = PCA(n_components=4).fit_transform(features_scaled)
# Scale to [0, 2*pi] for quantum angles
features_reduced = features_reduced * 2 * np.pi

scaler_t = MinMaxScaler()
target = scaler_t.fit_transform(data[[target_column]]).flatten()

# 2. Modelo Cuántico Avanzado - OPTIMIZADO
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(params, x):
    qml.AngleEmbedding(x[:4], wires=range(n_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

class VQR_Model:
    def __init__(self, n_qubits, learning_rate=0.01, n_layers=3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Initialize params with smaller values for stability
        self.params = np.random.uniform(0, 0.5, (self.n_layers, n_qubits, 3), requires_grad=True)
        self.learning_rate = learning_rate

    def fit(self, X, y, epochs=200):
        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        X_t = np.array(X)
        y_t = np.array(y)
        batch_size = 16  # Smaller batches for better learning

        print(f"  -> Learning rate: {self.learning_rate}, Layers: {self.n_layers}")
        best_loss = float('inf')
        for epoch in range(epochs):
            # Mini-batch training
            idx = np.random.choice(len(X_t), batch_size, replace=True)
            X_batch = X_t[idx]
            y_batch = y_t[idx]

            def cost(p):
                preds = [quantum_circuit(p, x) for x in X_batch]
                return np.mean((np.array(preds) - y_batch) ** 2)

            self.params = opt.step(cost, self.params)

            if epoch == 0:
                current_loss = cost(self.params)
                best_loss = current_loss

            if (epoch+1) % 20 == 0:
                current_loss = cost(self.params)
                print(f"  -> Época {epoch+1}: Loss = {current_loss:.6f}")

    def predict(self, X):
        preds = np.array([quantum_circuit(self.params, x) for x in X])
        return preds.flatten()

# 3. Entrenamiento y Comparativa - CON TUNING
kf = KFold(n_splits=3, shuffle=True, random_state=42)
print("Iniciando validación cruzada (3 Folds) - VERSIÓN OPTIMIZADA\n")

# Test different hyperparameters
configs = [
    {"lr": 0.01, "layers": 2, "epochs": 150},  # Lower learning rate
    {"lr": 0.005, "layers": 3, "epochs": 200}, # Even lower LR, more layers
    {"lr": 0.02, "layers": 2, "epochs": 150},  # Slightly higher LR
]

run_only = [1]   # run only config 2

for config_idx, config in enumerate(configs, 1):

    if config_idx not in run_only:
        continue
    
    print(f"\n{'='*60}")
    print(f"CONFIGURACIÓN {config_idx}: lr={config['lr']}, layers={config['layers']}, epochs={config['epochs']}")
    print(f"{'='*60}")

    r2_quantum = []
    r2_classical = []
    r2_mlp_list = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(features_reduced)):
        print(f"\n--- Fold {fold+1} ---")

        # Quantum model with tuned hyperparameters
        model = VQR_Model(n_qubits=n_qubits, learning_rate=config['lr'], n_layers=config['layers'])
        model.fit(features_reduced[train_idx], target[train_idx], epochs=config['epochs'])
        y_pred_q = model.predict(features_reduced[test_idx])

        # Classical baseline
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(features_reduced[train_idx], target[train_idx])
        y_pred_rf = rf.predict(features_reduced[test_idx])

        # =========================
        # Classical Neural Network
        # =========================
        mlp = MLPRegressor(hidden_layer_sizes=(32, 16),activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        mlp.fit(features_reduced[train_idx], target[train_idx])
        y_pred_mlp = mlp.predict(features_reduced[test_idx])

        r2_q = r2_score(target[test_idx], y_pred_q)
        r2_c = r2_score(target[test_idx], y_pred_rf)
        r2_mlp = r2_score(target[test_idx], y_pred_mlp)

        r2_quantum.append(r2_q)
        r2_classical.append(r2_c)
        r2_mlp_list.append(r2_mlp)

        print(f"  R2 Quantum : {r2_q:.4f}")
        print(f"  R2 Classical: {r2_c:.4f}")
        print(f"  R2 MLP: {r2_mlp:.4f}")

    print(f"\nPromedio para Config {config_idx}:")
    print(f"  R2 Quantum (promedio) : {np.mean(r2_quantum):.4f}")
    print(f"  R2 RandomForest (promedio): {np.mean(r2_classical):.4f}")
    print(f"  R2 Neural Network (promedio): {np.mean(r2_mlp_list):.4f}")
