import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURACIÓN PARA CORRIDA CIENTÍFICA ---
N_GALAXIES = 2500  # Balance entre estadística y tiempo
N_QUBITS = 6       
N_LAYERS = 4       
N_FOLDS = 5        
LEARNING_RATE = 0.01 
EPOCHS = 150       
BATCH_SIZE = 16    

# 1. Preparación de datos
print(f"Cargando y preparando {N_GALAXIES} galaxias...")
try:
    data = pd.read_csv('sdss_galaxies_spectroscopy.csv')
except FileNotFoundError:
    # Intento buscar en el directorio superior si no está en el actual
    data = pd.read_csv('01_Proyectos_Personales/02_Quantum/023_trabajo/sdss_galaxies_spectroscopy.csv')

target_column = 'metallicity'
feature_columns = ['h_alpha_flux', 'h_beta_flux', 'oiii_5007_flux', 'nii_6584_flux', 
                   'oii_3726_flux', 'oii_3729_flux', 'sii_6717_flux', 'sii_6731_flux',
                   'log_stellar_mass', 'z']

data = data.dropna(subset=feature_columns + [target_column]).reset_index(drop=True)
if len(data) > N_GALAXIES:
    data = data.sample(n=N_GALAXIES, random_state=42).reset_index(drop=True)

scaler_f = MinMaxScaler(feature_range=(0, np.pi))
features_scaled = scaler_f.fit_transform(data[feature_columns])
pca = PCA(n_components=N_QUBITS)
features_reduced = pca.fit_transform(features_scaled)

scaler_t = MinMaxScaler()
target = scaler_t.fit_transform(data[[target_column]]).flatten()

# 2. Modelo Cuántico Avanzado (V2)
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def quantum_circuit(params, x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation='Y')
    qml.StronglyEntanglingLayers(params, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

class VQR_Model:
    def __init__(self, n_qubits, n_layers):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = np.random.uniform(0, 2*np.pi, (n_layers, n_qubits, 3), requires_grad=True)

    def fit(self, X, y, epochs=EPOCHS):
        opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)
        X_t = np.array(X, requires_grad=False)
        y_t = np.array(y, requires_grad=False)
        
        for epoch in range(epochs):
            idx = np.random.choice(len(X_t), BATCH_SIZE)
            X_batch = X_t[idx]
            y_batch = y_t[idx]
            
            def cost(p):
                preds = [quantum_circuit(p, x) for x in X_batch]
                return np.mean((np.array(preds) - y_batch) ** 2)
            
            self.params = opt.step(cost, self.params)
            if (epoch+1) % 50 == 0: 
                print(f"    -> Epoca {epoch+1}/{epochs} completada")

    def predict(self, X):
        return np.array([quantum_circuit(self.params, x) for x in X])

# 3. Entrenamiento y Comparativa
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
results = []
print(f"\nIniciando Validacion Cruzada ({N_FOLDS} Folds)...")
start_total = time.time()

for fold, (train_idx, test_idx) in enumerate(kf.split(features_reduced)):
    print(f"\n--- [FOLD {fold+1}/{N_FOLDS}] ---")
    
    start_q = time.time()
    model_q = VQR_Model(n_qubits=N_QUBITS, n_layers=N_LAYERS)
    model_q.fit(features_reduced[train_idx], target[train_idx])
    y_pred_q = model_q.predict(features_reduced[test_idx])
    time_q = time.time() - start_q
    
    start_rf = time.time()
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    rf.fit(features_reduced[train_idx], target[train_idx])
    y_pred_rf = rf.predict(features_reduced[test_idx])
    time_rf = time.time() - start_rf
    
    metrics = {
        'fold': fold + 1,
        'q_r2': r2_score(target[test_idx], y_pred_q),
        'q_mae': mean_absolute_error(target[test_idx], y_pred_q),
        'q_mse': mean_squared_error(target[test_idx], y_pred_q),
        'q_time': time_q,
        'rf_r2': r2_score(target[test_idx], y_pred_rf),
        'rf_mae': mean_absolute_error(target[test_idx], y_pred_rf),
        'rf_mse': mean_squared_error(target[test_idx], y_pred_rf),
        'rf_time': time_rf
    }
    results.append(metrics)
    print(f"  Resultados Fold {fold+1}: Q-R2: {metrics['q_r2']:.4f} | RF-R2: {metrics['rf_r2']:.4f}")

# 4. Resumen Final
df_res = pd.DataFrame(results)
print("\n" + "="*50)
print("RESUMEN FINAL DE LA CORRIDA CIENTIFICA")
print("="*50)
print(f"Tiempo Total: {(time.time() - start_total)/60:.2f} minutos")
print(f"VQR R2: {df_res['q_r2'].mean():.4f} +/- {df_res['q_r2'].std():.4f}")
print(f"RF  R2: {df_res['rf_r2'].mean():.4f} +/- {df_res['rf_r2'].std():.4f}")
print(f"VQR MAE: {df_res['q_mae'].mean():.4f}")
print(f"RF  MAE: {df_res['rf_mae'].mean():.4f}")
print(f"Ratio Tiempo (VQR/RF): {df_res['q_time'].mean() / df_res['rf_time'].mean():.1f}x")
print("="*50)
