# Metodología: QML vs ML clásico para metalicidad galáctica en SDSS

Archivo vivo del proyecto. Registra la secuencia metodológica, scripts, comandos, productos intermedios, resultados y advertencias científicas para una charla de congreso sobre el uso de ML clásico y QML en la estimación de metalicidad nebular de galaxias SDSS.

---

## 1. Objetivo

Comparar algoritmos clásicos de Machine Learning y algoritmos cuánticos variacionales para estimar:

```text
metallicity = oh_p50 ≈ 12 + log(O/H)
```

en galaxias star-forming del SDSS, usando ratios de líneas de emisión y propiedades globales simples.

La pregunta central no es asumir que QML ya debe superar al ML clásico. La pregunta útil para congreso es:

> ¿Qué tan cerca puede llegar un circuito cuántico variacional, bajo restricciones NISQ/simulador, a benchmarks clásicos fuertes o compactos en un problema astrofísico realista?

Resultado negativo también es resultado: si QML no supera a ML clásico, se cuantifica la brecha, el coste computacional y qué configuraciones empiezan a aproximarse.

---

## 2. Diagrama de flujo actual

```text
1) Descarga observacional SDSS
   datos_sdss_v2.py
        |
        v
   sdss_galaxies_spectroscopy_v2.csv
   - flujos, errores, z, masa, metallicity
   - 10000 galaxias descargadas desde SDSS DR18

        |
        v

2) Construcción de features físicas
   preprocess_sdss_features.py
        |
        v
   sdss_galaxies_features_physical.csv
   sdss_galaxies_features_physical_metadata.json
   - ratios nebulares logarítmicos
   - sin escalado, sin PCA, sin train/test split

        |
        v

3) Benchmark clásico amplio
   compare_ml_qml_metallicity_v1.py
        |
        v
   results_ml_qml_v1/
   - ML clásico full-feature
   - controles clásicos con PCA(4, 6, 8)
   - establece el techo clásico y referencias compactas

        |
        v

4) Primera comparación QML controlada
   compare_qml_vqr_metallicity_v1.py
        |
        v
   results_qml_vqr_v1/
   - VQR vs Ridge/RF con la misma PCA
   - prueba 4q/2l, 6q/2l, 6q/4l

        |
        v

5) Optimización inicial VQR
   optimize_vqr_metallicity_v1.py
        |
        v
   results_vqr_optimization_v1/
   - grilla sobre capas, learning rate y semillas
   - identifica configuración VQR más prometedora

        |
        v

6) Próximos pasos
   - visualizaciones
   - análisis de residuos
   - repetir mejores VQR con más semillas/folds
   - probar arquitecturas QML mejoradas si corresponde
```

---

## 3. Dataset y features

### 3.1. Fuente observacional

Archivo generado:

```text
sdss_galaxies_spectroscopy_v2.csv
```

Script:

```text
datos_sdss_v2.py
```

Fuente usada en la ejecución válida:

```text
SDSS DR18
```

Nota operacional: `data_release = 17` produjo errores de servidor para esta consulta; con `data_release = 18` la descarga funcionó correctamente. Por trazabilidad, cualquier presentación debe reportar DR18 si usa este CSV.

Columnas descargadas:

```text
specobjid, ra, dec, z,
h_alpha_flux, h_alpha_flux_err,
h_beta_flux, h_beta_flux_err,
oiii_5007_flux, oiii_5007_flux_err,
nii_6584_flux, nii_6584_flux_err,
oii_3726_flux, oii_3726_flux_err,
oii_3729_flux, oii_3729_flux_err,
sii_6717_flux, sii_6717_flux_err,
sii_6731_flux, sii_6731_flux_err,
bptclass,
log_stellar_mass,
metallicity
```

Cortes aplicados en la consulta/validación:

```text
class = GALAXY
0.02 <= z <= 0.10
bptclass = 1
7.0 <= metallicity <= 10.0
6.0 <= log_stellar_mass <= 13.0
flujos > 0
S/N >= 3 en todas las líneas usadas
```

Implementación relevante:

- descarga objetivo: `N_GALAXIES = 10000`;
- descarga por lotes: `BATCH_SIZE = 1000`;
- orden reproducible por `specobjid`;
- condición incremental:

```sql
s.specobjid > CAST(last_specobjid AS BIGINT)
```

Esto evita errores de servidor asociados a `TOP 10000` con `ORDER BY` global.

---

### 3.2. Features físicas

Script:

```text
preprocess_sdss_features.py
```

Entrada:

```text
sdss_galaxies_spectroscopy_v2.csv
```

Salidas:

```text
sdss_galaxies_features_physical.csv
sdss_galaxies_features_physical_metadata.json
```

Resultado de ejecución:

```text
Filas iniciales: 10000
Filas finales: 10000
Filas descartadas: 0
```

Features generadas:

```text
log_nii_ha       = log10([NII]6584 / Halpha)
log_oiii_hb      = log10([OIII]5007 / Hbeta)
log_oii_hb       = log10(([OII]3726 + [OII]3729) / Hbeta)
log_sii_ha       = log10(([SII]6717 + [SII]6731) / Halpha)
log_r23          = log10(([OII]3726 + [OII]3729 + [OIII]5007) / Hbeta)
log_o32          = log10([OIII]5007 / ([OII]3726 + [OII]3729))
log_stellar_mass = lgm_tot_p50
z                = redshift espectroscópico
```

Target:

```text
metallicity = oh_p50
```

Decisión metodológica: este script **no escala**, **no aplica PCA** y **no divide train/test**. Esas operaciones se hacen dentro de cada fold de validación cruzada para evitar leakage.

Justificación física resumida:

- `log_nii_ha`: índice tipo N2, sensible a metalicidad/N/O.
- `log_oiii_hb`: excitación/ionización; participa en BPT y O3N2.
- `log_oii_hb`: componente de baja ionización, necesaria para R23/O32.
- `log_sii_ha`: sensible a condiciones de ionización y posible DIG/LI(N)ER.
- `log_r23`: diagnóstico clásico de abundancia de oxígeno.
- `log_o32`: proxy del parámetro de ionización.
- `log_stellar_mass`: relación masa-metalicidad.
- `z`: controla posibles sesgos de selección/apertura/evolución residual.

Bibliografía de trabajo para verificar definiciones: Pettini & Pagel (2004), Tremonti et al. (2004), Kobulnicky & Kewley (2004), Kewley et al. (2019), Bian et al. (2018), Patricio et al. (2018), Zhang et al. (2016), Kumari et al. (2019), Nakajima et al. (2022).

---

## 4. Scripts del pipeline

### 4.1. `datos_sdss_v2.py` — descarga SDSS robusta

Función: construir la muestra observacional base desde SDSS.

Genera:

```text
sdss_galaxies_spectroscopy_v2.csv
```

Características:

- consulta `SpecObj`, `galSpecLine` y `galSpecExtra`;
- descarga flujos y errores de líneas nebulares;
- aplica cortes físicos y de S/N;
- usa descarga por lotes ordenados;
- guarda CSV en el directorio de trabajo.

Comando típico:

```bash
python3 datos_sdss_v2.py
```

---

### 4.2. `preprocess_sdss_features.py` — ratios físicos

Función: transformar flujos limpios en features físicas interpretables.

Genera:

```text
sdss_galaxies_features_physical.csv
sdss_galaxies_features_physical_metadata.json
```

Características:

- valida columnas;
- valida valores finitos;
- valida flujos positivos;
- construye ratios nebulares logarítmicos;
- conserva `specobjid`, `ra`, `dec` para trazabilidad;
- no introduce leakage.

Comando:

```bash
python3 preprocess_sdss_features.py
```

---

### 4.3. `compare_ml_qml_metallicity_v1.py` — benchmark clásico

Función: establecer el techo clásico y controles PCA antes de usar QML.

Entrada:

```text
sdss_galaxies_features_physical.csv
```

Genera:

```text
results_ml_qml_v1/metrics_by_fold.csv
results_ml_qml_v1/predictions_by_fold.csv
results_ml_qml_v1/summary_metrics.csv
results_ml_qml_v1/run_config.json
```

Modelos con features completas:

```text
DummyRegressor(mean)
Ridge
ElasticNet
RandomForestRegressor
SVR-RBF
MLPRegressor
XGBoost si está instalado; si no, HistGradientBoostingRegressor
```

Controles PCA:

```text
Ridge + PCA(4, 6, 8)
RandomForest + PCA(4, 6, 8)
```

Comando de corrida completa usado:

```bash
python3 compare_ml_qml_metallicity_v1.py \
  --n-galaxies 5000 \
  --n-folds 5 \
  --output-dir results_ml_qml_v1
```

Configuración registrada:

```text
n_galaxies = 5000
n_folds = 5
seed = 42
SVR = activado
MLP = activado
XGBoost = no disponible; se usó HistGradientBoostingRegressor
```

Resultados principales:

```text
modelo                         R2 medio    MAE medio    RMSE medio
mlp_full                       0.98036     0.01951     0.02961
svr_rbf_full                   0.97779     0.01963     0.03138
random_forest_full             0.97534     0.02002     0.03308
hist_gradient_boosting_full    0.97300     0.02001     0.03445
random_forest_pca8             0.97248     0.02300     0.03503
random_forest_pca6             0.97219     0.02317     0.03523
ridge_pca6                     0.95309     0.03317     0.04580
ridge_full                     0.95297     0.03307     0.04586
random_forest_pca4             0.93128     0.03471     0.05540
ridge_pca4                     0.89340     0.04781     0.06906
dummy_mean_full               -0.00091     0.16743     0.21178
```

Lectura:

- ML clásico full-feature alcanza RMSE ~0.03 dex.
- Ridge ya funciona muy bien: la relación entre features y `oh_p50` es en gran parte suave/casi lineal.
- Los controles PCA son esenciales para comparar justamente contra VQR, porque el VQR queda limitado a `n_qubits` dimensiones.

---

### 4.4. `compare_qml_vqr_metallicity_v1.py` — primera comparación VQR

Función: comparar VQR contra Ridge/RF entrenados sobre la misma PCA en cada fold.

Genera:

```text
results_qml_vqr_v1/metrics_by_fold.csv
results_qml_vqr_v1/predictions_by_fold.csv
results_qml_vqr_v1/training_history.csv
results_qml_vqr_v1/summary_metrics.csv
results_qml_vqr_v1/run_config.json
```

Arquitectura QML:

```text
Embedding:   AngleEmbedding(rotation='Y')
Ansatz:      StronglyEntanglingLayers
Observable:  <PauliZ(0)>
Optimizer:   qml.AdamOptimizer
Loss:        MSE en target escalado [-1, 1]
```

Preprocesamiento por fold:

```text
StandardScaler fit en train
PCA(n_components = n_qubits) fit en train
MinMaxScaler a [0, pi] para ángulos fit en train
target scaler a [-1, 1] fit en train solo para VQR
```

Comando ejecutado:

```bash
python3 compare_qml_vqr_metallicity_v1.py \
  --n-galaxies 500 \
  --n-folds 3 \
  --vqr-configs 4:2,6:2,6:4 \
  --epochs 120 \
  --batch-size 16 \
  --output-dir results_qml_vqr_v1
```

Resultados:

```text
modelo                 familia              qubits  capas  R2 medio   MAE medio   RMSE medio   tiempo medio
ridge_pca6             classical_control    6       0      0.94841   0.03520    0.04838      0.0008 s
random_forest_pca6     classical_control    6       0      0.93799   0.03563    0.05308      0.5033 s
vqr_q6_l4              vqr                  6       4      0.89249   0.04561    0.06875      173.26 s
ridge_pca4             classical_control    4       0      0.88622   0.05040    0.07190      0.0008 s
random_forest_pca4     classical_control    4       0      0.87372   0.04670    0.07548      0.4853 s
vqr_q6_l2              vqr                  6       2      0.75958   0.07391    0.10389      88.25 s
vqr_q4_l2              vqr                  4       2      0.53146   0.09938    0.13435      60.29 s
```

Lectura:

- El mejor VQR inicial fue `vqr_q6_l4`.
- Aumentar profundidad de 2 a 4 capas mejoró mucho el VQR de 6 qubits.
- No hay ventaja cuántica frente a controles clásicos PCA6.
- `vqr_q6_l4` queda cerca del régimen de `ridge_pca4`, lo cual ya es una línea base QML no trivial.

---

### 4.5. `optimize_vqr_metallicity_v1.py` — optimización inicial VQR

Función: explorar hiperparámetros del VQR para mejorar la configuración `6 qubits / 4 capas`.

Genera:

```text
results_vqr_optimization_v1/metrics_by_fold.csv
results_vqr_optimization_v1/predictions_by_fold.csv
results_vqr_optimization_v1/training_history.csv
results_vqr_optimization_v1/summary_metrics.csv
results_vqr_optimization_v1/best_config.csv
results_vqr_optimization_v1/run_config.json
```

Hiperparámetros explorados:

```text
n_qubits       = 6
n_layers       = 4, 6
learning_rates = 0.01, 0.03
rotations      = Y
init_seeds     = 101, 202, 303
epochs         = 200
batch_size     = 16
n_folds        = 3
n_galaxies     = 500
```

Comando ejecutado:

```bash
python3 optimize_vqr_metallicity_v1.py \
  --n-galaxies 500 \
  --n-folds 3 \
  --n-qubits 6 \
  --n-layers 4,6 \
  --learning-rates 0.01,0.03 \
  --rotations Y \
  --init-seeds 101,202,303 \
  --epochs 200 \
  --batch-size 16 \
  --output-dir results_vqr_optimization_v1
```

Tamaño de la grilla:

```text
1 n_qubits × 2 n_layers × 2 learning rates × 1 rotation × 3 seeds × 3 folds = 36 entrenamientos VQR
```

Mejor configuración:

```text
model       = vqr_q6_l6_lr0.03_rotY_seed202
n_qubits    = 6
n_layers    = 6
learning_rate = 0.03
rotation    = Y
init_seed   = 202
R2 medio    = 0.91875
MAE medio   = 0.04058 dex
RMSE medio  = 0.06025 dex
tiempo medio = 501.66 s por fold
```

Top de configuraciones:

```text
modelo                               R2 medio   MAE medio   RMSE medio   tiempo medio
vqr_q6_l6_lr0.03_rotY_seed202        0.91875   0.04058    0.06025      501.66 s
vqr_q6_l6_lr0.03_rotY_seed303        0.91704   0.04026    0.06140      514.03 s
vqr_q6_l6_lr0.01_rotY_seed101        0.91002   0.04096    0.06327      478.82 s
vqr_q6_l6_lr0.01_rotY_seed303        0.90824   0.04079    0.06402      519.25 s
vqr_q6_l6_lr0.03_rotY_seed101        0.90605   0.04250    0.06418      507.52 s
vqr_q6_l4_lr0.03_rotY_seed303        0.90069   0.04425    0.06497      324.96 s
vqr_q6_l4_lr0.03_rotY_seed101        0.90205   0.04337    0.06675      326.54 s
```

Lectura:

- La optimización mejoró el mejor VQR de RMSE ~0.06875 a RMSE ~0.06025 dex.
- Pasar de 4 a 6 capas ayuda, pero el coste sube de ~325 s a ~500 s por fold/configuración.
- El mejor VQR sigue por debajo de `ridge_pca6` y `random_forest_pca6` de la comparación controlada.
- El resultado es científicamente útil: QML todavía no supera a ML clásico, pero una configuración VQR compacta se aproxima al régimen de controles clásicos reducidos.

Comparación clave:

```text
MLP full clásico         RMSE ≈ 0.02961 dex
RandomForest PCA6        RMSE ≈ 0.05308 dex
Ridge PCA6               RMSE ≈ 0.04838 dex
VQR optimizado 6q/6l     RMSE ≈ 0.06025 dex
```

Conclusión provisional:

```text
No hay ventaja cuántica en esta etapa.
Sí hay aprendizaje QML significativo y mejorable.
La brecha actual frente al mejor control PCA6 es ~0.012 dex en RMSE respecto a Ridge PCA6 y ~0.007 dex respecto a RF PCA6.
```

---

## 5. Tabla rápida de resultados acumulados

```text
Etapa                          Mejor modelo/configuración              RMSE medio
ML clásico full                 MLP full                                0.02961 dex
ML clásico PCA6                 RandomForest PCA6                       0.03523 dex  (corrida 5000 gal)
ML clásico PCA6 control QML      Ridge PCA6                              0.04838 dex  (corrida 500 gal)
QML inicial                     VQR 6q/4l                               0.06875 dex
QML optimizado                  VQR 6q/6l, lr=0.03, seed=202            0.06025 dex
Baseline nulo                   Dummy mean                              0.21178 dex
```

Nota: no comparar de forma ingenua números de corridas con distinto `n_galaxies` y distinto KFold. Para figuras finales habrá que ejecutar modelos seleccionados bajo una misma partición y tamaño de muestra.

---

## 6. Decisiones metodológicas clave

### 6.1. Por qué ratios y no flujos crudos

Los flujos absolutos dependen de distancia, luminosidad, apertura de fibra, extinción y selección observacional. Los ratios nebulares están conectados con diagnósticos físicos de abundancia, ionización y excitación.

Limitación: aún no se corrigió extinción interna con decremento de Balmer. Ratios como [OII]/Hbeta, R23 y O32 pueden estar afectados por reddening.

### 6.2. Por qué PCA dentro de cada fold

El VQR necesita que la dimensionalidad de entrada coincida con `n_qubits`. El PCA permite comprimir las 8 features físicas a 4 o 6 dimensiones. Debe ajustarse solo con train; hacerlo antes del KFold sería leakage.

### 6.3. Por qué comparar VQR contra controles PCA

Comparar VQR de 6 qubits contra MLP/RF usando 8 features completas es útil como techo clásico, pero no es una comparación dimensionalmente justa. La comparación justa es:

```text
VQR(n_qubits=k)
vs
Ridge/RF/PCA(n_components=k)
```

### 6.4. Por qué escalar target a [-1, 1] en VQR

El observable `PauliZ(0)` devuelve valores en `[-1, 1]`. Por eso `metallicity` se escala a ese rango dentro de cada fold y luego se transforma de vuelta a dex antes de calcular métricas.

---

## 7. Próximos pasos recomendados

1. **Visualización de resultados actuales**
   - predicho vs real;
   - residuos vs metalicidad;
   - residuos vs masa;
   - residuos vs redshift;
   - curva de pérdida VQR.

2. **Re-ejecutar una comparación final homogénea**
   - mismos 500 objetos o subir a 1000 si el tiempo lo permite;
   - mismos folds;
   - modelos: Ridge PCA6, RF PCA6, MLP full, mejor VQR 6q/6l.

3. **Probar mejoras QML seleccionadas, no una explosión de modelos**
   - data re-uploading;
   - múltiples observables (`PauliZ` en varios wires);
   - `BasicEntanglerLayers` vs `StronglyEntanglingLayers`;
   - rotaciones X/Y/Z si el tiempo computacional lo permite.

4. **Preparar narrativa de congreso**
   - ML clásico aún domina;
   - VQR aprende una relación astrofísica realista;
   - la brecha baja con profundidad/optimización;
   - coste computacional en simulador es el principal cuello de botella;
   - esto marca una línea base para futuras pruebas en hardware/ruido.

---

## 8. Advertencias científicas pendientes

- [!] Confirmar exactamente cómo SDSS/MPA-JHU define `oh_p50` en el data release usado.
- [!] `oh_p50` probablemente fue inferida usando información espectral relacionada con nuestras features. El ejercicio es válido como benchmark de regresión/calibración ML/QML, pero no debe presentarse como predicción independiente de una abundancia medida externamente.
- [!] Evaluar corrección por extinción usando Halpha/Hbeta.
- [!] Evaluar sesgo de selección inducido por exigir S/N >= 3 en todas las líneas.
- [!] Confirmar interpretación operativa de `bptclass = 1` en DR18.
- [!] Para resultados finales, ejecutar comparaciones seleccionadas bajo exactamente el mismo dataset/folds.

---

## 9. Bibliografía de trabajo para leer/verificar

No es bibliografía final; sirve para revisar definiciones y contexto físico.

1. Tremonti et al. (2004), relación masa-metalicidad en SDSS.
2. Pettini & Pagel (2004), calibraciones N2 y O3N2.
3. Kobulnicky & Kewley (2004), calibraciones R23.
4. Kewley & Dopita (2002), modelos fotoionizados y diagnósticos de abundancia.
5. Kewley et al. (2019): [Understanding Galaxy Evolution Through Emission Lines](https://consensus.app/papers/details/982d3c985be65ebf99f22fd1ec7e46ca/?utm_source=unknown).
6. Bian et al. (2018): [“Direct” Gas-phase Metallicity in Local Analogs of High-redshift Galaxies](https://consensus.app/papers/details/a751a4f9ae16583887acc5778d246e99/?utm_source=unknown).
7. Patricio et al. (2018): [Testing strong line metallicity diagnostics at z ∼ 2](https://consensus.app/papers/details/9756527798365db7b6f138d6cc714a66/?utm_source=unknown).
8. Zhang et al. (2016): [SDSS-IV MaNGA: the impact of diffuse ionized gas on emission-line ratios, interpretation of diagnostic diagrams and gas metallicity measurements](https://consensus.app/papers/details/f0a2d0d0230c5025b6dd624ef0287353/?utm_source=unknown).
9. Kumari et al. (2019): [Metallicity calibrations for diffuse ionized gas and low-ionization emission regions](https://consensus.app/papers/details/3b73081d4bcb5d8090cf806db9bec7bb/?utm_source=unknown).
10. Nakajima et al. (2022), EMPRESS V, diagnósticos ópticos de metalicidad.

---

## 10. Registro cronológico

### 2026-06-15

- Se creó `datos_sdss_v2.py`.
- Se descargaron 10000 galaxias usando SDSS DR18.
- Se creó `preprocess_sdss_features.py`.
- Se generó `sdss_galaxies_features_physical.csv` con 10000 galaxias válidas.
- Se creó `compare_ml_qml_metallicity_v1.py` y se ejecutó benchmark clásico.
- Se creó `compare_qml_vqr_metallicity_v1.py` y se ejecutó primera comparación VQR.

### 2026-06-16

- Se creó `optimize_vqr_metallicity_v1.py`.
- Se ejecutó grilla VQR con 6 qubits, 4/6 capas, dos learning rates y tres semillas.
- Mejor configuración encontrada: `vqr_q6_l6_lr0.03_rotY_seed202`, RMSE medio ≈ 0.06025 dex.
