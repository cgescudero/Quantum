import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# 1. Preparación de datos
data = pd.read_csv('sdss_galaxies_spectroscopy.csv')
target_column = 'metallicity'
feature_columns =['h_alpha_flux', 'h_beta_flux', 'oiii_5007_flux', 'nii_6584_flux', 
                   'oii_3726_flux', 'oii_3729_flux', 'sii_6717_flux', 'sii_6731_flux',
                   'log_stellar_mass', 'z']
data = data.dropna(subset=feature_columns + [target_column]).reset_index(drop=True)

scaler_f = MinMaxScaler(feature_range=(0, np.pi))
features_reduced = PCA(n_components=4).fit_transform(scaler_f.fit_transform(data[feature_columns]))
scaler_t = MinMaxScaler()
target = scaler_t.fit_transform(data[[target_column]]).flatten()

# 2. Modelo Cuántico Avanzado
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(params, x):
    # Codificación de datos
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
    # Capa variacional compleja
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

class VQR_Model:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.n_layers = 2
        self.params = np.random.uniform(0, 2*np.pi, (self.n_layers, n_qubits, 3), requires_grad=True)

    def fit(self, X, y, epochs=100):
        opt = qml.AdamOptimizer(stepsize=0.05)
        X_t = np.array(X, requires_grad=False)
        y_t = np.array(y, requires_grad=False)
        batch_size = 32  # <-- ¡ESTO ES LO QUE FALTABA PARA QUE SEA RÁPIDO!
        
        print("  -> Iniciando cálculo de gradientes...")
        for epoch in range(epochs):
            # Tomamos un lote aleatorio pequeño
            idx = np.random.choice(len(X_t), batch_size)
            X_batch = X_t[idx]
            y_batch = y_t[idx]
            
            def cost(p):
                preds =[quantum_circuit(p, x) for x in X_batch]
                return np.mean((np.array(preds) - y_batch) ** 2)
            
            self.params = opt.step(cost, self.params)
            
            if (epoch+1) % 10 == 0: 
                print(f"  -> Época {epoch+1} completada")

    def predict(self, X):
        return np.array([quantum_circuit(self.params, x) for x in X])

# 3. Entrenamiento y Comparativa
kf = KFold(n_splits=3, shuffle=True, random_state=42)
print("Iniciando validación cruzada (3 Folds)...")

for fold, (train_idx, test_idx) in enumerate(kf.split(features_reduced)):
    print(f"\n--- Entrenando Fold {fold+1} ---")
    
    # Entrenamiento Cuántico
    model = VQR_Model(n_qubits=n_qubits)
    model.fit(features_reduced[train_idx], target[train_idx], epochs=100)
    y_pred_q = model.predict(features_reduced[test_idx])
    
    # Entrenamiento Clásico
    rf = RandomForestRegressor().fit(features_reduced[train_idx], target[train_idx])
    y_pred_rf = rf.predict(features_reduced[test_idx])
    
    print(f"Resultados Fold {fold+1}:")
    print(f"  R2 Cuántico: {r2_score(target[test_idx], y_pred_q):.4f}")
    print(f"  R2 Clásico : {r2_score(target[test_idx], y_pred_rf):.4f}")