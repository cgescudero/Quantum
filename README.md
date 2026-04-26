# Proyecto de Regresión Cuántica para Espectroscopía

Este proyecto implementa un modelo de **Regresor Cuántico Variacional (VQR)** para predecir la metalicidad de galaxias utilizando datos espectroscópicos.

## Estructura de Datos
Los datos necesarios para ejecutar el código se encuentran en la carpeta:
`/datos/sdss_galaxies_spectroscopy.csv`

## Descripción del Código
El script principal realiza el procesamiento de datos y la implementación del modelo cuántico siguiendo estas etapas:

1.  **Preprocesamiento:** 
    *   Carga y limpieza de datos espectroscópicos (flujos de líneas de emisión, masa estelar y redshift).
    *   Reducción de dimensionalidad mediante **PCA** (de múltiples características a 4 componentes principales).
    *   Normalización de los datos para su correcta codificación en el circuito cuántico.

2.  **Modelo Cuántico (VQR):**
    *   Se define un circuito cuántico utilizando **PennyLane** con 4 qubits.
    *   **Codificación:** Utiliza `AngleEmbedding` para mapear los datos clásicos al espacio cuántico.
    *   **Procesamiento:** Emplea `StronglyEntanglingLayers` para aprender las relaciones no lineales.
    *   **Optimización:** Implementa un optimizador `Adam` con entrenamiento por lotes (*mini-batch training*) para mejorar la eficiencia del cálculo de gradientes.

3.  **Evaluación y Comparativa:**
    *   El modelo es evaluado mediante **Validación Cruzada (K-Fold)** con 3 divisiones.
    *   Compara el rendimiento predictivo (usando la métrica $R^2$) del **modelo cuántico** frente a un modelo clásico de **Random Forest Regressor**, sirviendo como base para medir la capacidad de aprendizaje del circuito cuántico.
