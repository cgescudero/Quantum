
# Guía de Instalación: Entorno para Quantum

### 1. Crear el entorno base
Recomendamos **Python 3.10** porque garantiza la mejor compatibilidad con los drivers de PennyLane y las librerías de astrofísica.

```bash
# Crear el entorno
conda create --name quantum python=3.10 -y

# Activar el entorno
conda activate quantum
```

### 2. Instalar el Stack de Ciencia de Datos (vía Conda)
Instalamos primero las librerías "pesadas" a través de Conda para asegurar que utilicen las librerías de aceleración matemática (MKL/BLAS) de tu procesador.

```bash
conda install pandas scikit-learn matplotlib scipy ipykernel -y
```

### 3. Instalar el Stack Cuántico y Astro (vía Pip)
Instalamos PennyLane y sus aceleradores, junto con las herramientas de acceso a datos del SDSS.

```bash
pip install pennylane pennylane-lightning astroquery astropy
```

### 4. Verificación de la instalación
Para asegurar que todo funcione correctamente, ejecutar este pequeño comando en la terminal:

```bash
python -c "import pennylane as qml; import astroquery; print('PennyLane:', qml.__version__); print('Simulador Lightning listo.');"
```

