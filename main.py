import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.matrix_generator import generate_diagonally_dominant_matrix
from utils.timer import measure_time
from methods.elimination_gauss import gaussian_elimination
from methods.factorization_lu import lu_method
from methods.gauss_seidel import gauss_seidel

def main():
    sizes = [3, 10, 100]
    results = []

    for n in sizes:
        print(f"\n🧮 Resolviendo sistema de tamaño {n}x{n}")
        A, b = generate_diagonally_dominant_matrix(n)

        # Método 1: Eliminación Gaussiana
        (x_gauss), t_gauss = measure_time(gaussian_elimination, A, b)
        residual_gauss = np.linalg.norm(np.dot(A, x_gauss) - b)

        # Método 2: Factorización LU
        (x_lu), t_lu = measure_time(lu_method, A, b)
        residual_lu = np.linalg.norm(np.dot(A, x_lu) - b)

        # Método 3: Gauss-Seidel
        (res_gs, iter_gs, conv_gs), t_gs = measure_time(gauss_seidel, A, b)
        residual_gs = np.linalg.norm(np.dot(A, res_gs) - b)

        results.append({
            "n": n,
            "Método": "Eliminación Gaussiana",
            "Tiempo (s)": t_gauss,
            "Residuo": residual_gauss
        })
        results.append({
            "n": n,
            "Método": "LU",
            "Tiempo (s)": t_lu,
            "Residuo": residual_lu
        })
        results.append({
            "n": n,
            "Método": "Gauss-Seidel",
            "Tiempo (s)": t_gs,
            "Residuo": residual_gs
        })

    # Convertir a DataFrame
    df = pd.DataFrame(results)
    print("\n📊 Resultados:\n", df)

    # Guardar CSV
    df.to_csv("results/linear_solver_results.csv", index=False)

    # Gráficos
    plt.figure()
    for method in df["Método"].unique():
        plt.plot(df[df["Método"] == method]["n"], df[df["Método"] == method]["Tiempo (s)"], label=method)
    plt.xlabel("Tamaño de matriz (n)")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title("Comparación de tiempos de los métodos")
    plt.legend()
    plt.savefig("results/time_comparison.png")

    plt.figure()
    for method in df["Método"].unique():
        plt.plot(df[df["Método"] == method]["n"], df[df["Método"] == method]["Residuo"], label=method)
    plt.xlabel("Tamaño de matriz (n)")
    plt.ylabel("Residuo ||Ax - b||")
    plt.title("Comparación de precisión (residuo)")
    plt.legend()
    plt.savefig("results/residual_comparison.png")

    print("\n✅ Archivos guardados en carpeta 'results/'")

if __name__ == "__main__":
    main()