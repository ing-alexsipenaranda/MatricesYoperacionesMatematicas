# Sistema de Ecuaciones Lineales - Comparación de Métodos Numéricos

## 📘 Descripción General

Este proyecto implementa un **solucionador completo de sistemas de ecuaciones lineales Ax = b**, desarrollado en **Python**, con el propósito de **comparar la eficiencia, precisión y desempeño temporal** de distintos métodos matriciales.  
El trabajo hace parte de un ejercicio académico orientado al análisis y comparación de métodos numéricos utilizados en la ingeniería.

El enfoque principal está en analizar cómo se comportan tres métodos clásicos al resolver sistemas lineales de distintos tamaños, desde matrices pequeñas (3x3) hasta más grandes (100x100).

---

## 🎯 Objetivos

- Desarrollar un programa en Python que resuelva sistemas de ecuaciones lineales mediante tres enfoques:
  1. **Eliminación Gaussiana**
  2. **Factorización LU (usando `scipy.linalg`)**
  3. **Método iterativo de Gauss–Seidel**
- Incluir dos métodos adicionales de referencia:
  - **np.linalg.solve()**
  - **Cálculo con matriz inversa (A⁻¹·b)**
- Comparar **tiempos de ejecución**, **precisión** y **residuos** obtenidos por cada método.
- Guardar los **datasets de prueba**, **resultados** y **gráficos comparativos**.

---

## 🧠 Contexto del Proyecto

Este desarrollo forma parte del curso de **Métodos Numéricos aplicados a la Ingeniería de Software**, con el fin de entender cómo los distintos métodos matriciales se comportan en entornos computacionales reales, evaluando tanto el rendimiento como la estabilidad numérica.

El programa está diseñado para ser **modular, reproducible y automatizado**, ideal para pruebas experimentales o comparación con librerías numéricas más avanzadas.

---

## ⚙️ Requisitos y Librerías

### 📦 Requisitos mínimos
- **Python 3.9+**
- Sistema operativo compatible (Windows, macOS o Linux)
- Espacio disponible para almacenar resultados y datasets

### 🧩 Librerías necesarias
Instalar dependencias con:
```bash
pip install numpy matplotlib scipy
```

Librerías usadas:
- `numpy` → operaciones matriciales y algebra lineal
- `scipy.linalg` → factorización LU
- `matplotlib` → generación de gráficos comparativos
- `time` → medición de desempeño
- `pathlib` → manejo de archivos y directorios

---

## 🧮 Métodos Implementados

### 1. Eliminación Gaussiana
Resuelve Ax = b transformando la matriz aumentada [A|b] a una forma triangular superior y aplicando sustitución regresiva.

### 2. Factorización LU
Descompone A en el producto de dos matrices (L y U) para resolver el sistema de manera más eficiente, especialmente útil para resolver múltiples sistemas con la misma A.

### 3. Gauss–Seidel
Método iterativo que aproxima la solución de forma sucesiva, ideal para matrices dispersas o grandes. Incluye control de convergencia y número máximo de iteraciones.

### 4. Métodos de Referencia
- **`np.linalg.solve()`** — método de referencia optimizado de NumPy.
- **Matriz inversa (`A⁻¹·b`)** — útil para contrastar resultados, aunque menos eficiente.

---

## 🧪 Dataset y Pruebas

Se realizaron pruebas con matrices de distintos tamaños (3x3, 10x10, 100x100), tanto aleatorias como cargadas desde archivos Matrix Market (`.mtx`).  
Cada conjunto de prueba se guarda automáticamente en la carpeta `datasets/`.

Los resultados de cada ejecución se almacenan en:
```
results/linear_solver_results.csv
```

Y los gráficos generados:
```
results/linear_solver_time_comparison.png
results/linear_solver_residual_comparison.png
```

---

## 📊 Visualización y Análisis

El programa genera comparaciones visuales de:
- **Tiempo de ejecución (segundos)**
- **Residual numérico (‖Ax - b‖)**

Estos gráficos permiten evaluar cuál método es más rápido y cuál conserva mejor la precisión en distintos tamaños de matrices.

---

## 💾 Estructura del Proyecto

```
📂 sistema_ecuaciones_lineales/
├── main.py                      # Script principal del proyecto
├── datasets/                    # Almacena matrices A y vectores b usados en las pruebas
├── results/                     # Contiene CSV y gráficos comparativos
├── README.md                    # Este documento
└── requirements.txt             # Librerías necesarias
```

---

## 🚀 Ejecución

### Ejemplo básico (matriz 5x5)
```bash
python3 main.py --example5
```

### Pruebas de rendimiento (3x3, 10x10, 100x100)
```bash
python3 main.py --sizes 3 10 100 --repeats 3
```

### Cargar una matriz personalizada desde archivo Matrix Market
```bash
python3 main.py --mm path/to/matriz.mtx
```

---

## 📈 Resultados Esperados

Al finalizar la ejecución, el sistema genera:
- Un CSV con resultados detallados
- Gráficos comparativos (tiempo vs residual)
- Datasets almacenados (.npy)
- Un análisis fácil de interpretar para comparar los métodos

---

## 👨‍💻 Autor

**William Alexis Peñaranda Castro**  
Estudiante de Ingeniería de Software  
Proyecto académico - Métodos Numéricos  
FESC - 2025

---

## 🧾 Licencia

Uso académico y educativo.  
Libre distribución citando al autor.
