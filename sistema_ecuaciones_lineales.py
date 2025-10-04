#!/usr/bin/env python3
"""
main.py - SISTEMA DE ECUACIONES LINEALES
Soluciona Ax=b con varios métodos, compara eficiencia y precisión,
guarda CSV, gráficos y datasets generados.
"""
import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Intentos de import de SciPy (opcional)
# -------------------------
try:
    import scipy.linalg as sla
    SCIPY_AVAILABLE = True
except Exception:
    sla = None
    SCIPY_AVAILABLE = False

try:
    from scipy.io import mmread
    MMREAD_AVAILABLE = True
except Exception:
    mmread = None
    MMREAD_AVAILABLE = False

# -------------------------
# Utils
# -------------------------
def measure_time(fn, *args, **kwargs):
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return res, (t1 - t0)

def generate_diagonally_dominant_matrix(n, seed=None, scale=1.0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) * scale
    for i in range(n):
        A[i, i] += np.sum(np.abs(A[i])) + 1.0
    b = rng.standard_normal(n) * scale
    return A, b

def example_5x5():
    A = np.array([
        [10.0, -1.0, 2.0, 0.0, 0.0],
        [-1.0, 11.0, -1.0, 3.0, 0.0],
        [2.0, -1.0, 10.0, -1.0, 0.0],
        [0.0, 3.0, -1.0, 8.0, -2.0],
        [0.0, 0.0, 0.0, -2.0, 5.0]
    ])
    b = np.array([6.0, 25.0, -11.0, 15.0, 6.0])
    return A, b

def load_matrix_market(path):
    if not MMREAD_AVAILABLE:
        raise RuntimeError("scipy.io.mmread no disponible. Instala scipy para usar Matrix Market files.")
    M = mmread(path)
    if hasattr(M, "toarray"):
        A = M.toarray()
    else:
        A = np.array(M)
    return A

# -------------------------
# Métodos numéricos
# -------------------------
def numpy_solve(A, b):
    return np.linalg.solve(A, b)

def inverse_solve(A, b):
    return np.linalg.inv(A).dot(b)

def lu_solve(A, b):
    if SCIPY_AVAILABLE:
        lu, piv = sla.lu_factor(A)
        x = sla.lu_solve((lu, piv), b)
        return x
    else:
        return np.linalg.solve(A, b)

def gaussian_elimination(A_in, b_in):
    A = A_in.astype(float).copy()
    b = b_in.astype(float).copy()
    n = A.shape[0]
    for k in range(n - 1):
        max_row = np.argmax(np.abs(A[k:, k])) + k
        if np.isclose(A[max_row, k], 0.0):
            raise ValueError("Matriz singular (pivote cero).")
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        if np.isclose(A[i, i], 0.0):
            raise ValueError("Matriz singular en back substitution.")
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x

def gauss_seidel(A, b, x0=None, tol=1e-10, maxiter=10000):
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.astype(float).copy()
    converged = False
    for k in range(1, maxiter + 1):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            denom = A[i, i]
            if np.isclose(denom, 0.0):
                raise ValueError("A[i,i] = 0 en Gauss-Seidel.")
            x[i] = (b[i] - s1 - s2) / denom
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            converged = True
            break
    return x, k, converged

# -------------------------
# Ejecutar y recolectar resultados
# -------------------------
def run_all_methods(A, b):
    condA = np.linalg.cond(A)
    results = []
    def append(method_name, x=None, t=None, residual=None, iters=None, converged=None, error=None):
        results.append({
            "method": method_name,
            "time_s": float(t) if t is not None else None,
            "residual": float(residual) if residual is not None else None,
            "iters": int(iters) if iters is not None else None,
            "converged": bool(converged) if converged is not None else None,
            "cond": float(condA),
            "error": str(error) if error is not None else None
        })

    try:
        x, t = measure_time(numpy_solve, A, b)
        append("numpy_solve", x=x, t=t, residual=np.linalg.norm(A.dot(x) - b))
    except Exception as e:
        append("numpy_solve", error=e)

    try:
        x, t = measure_time(inverse_solve, A, b)
        append("inverse", x=x, t=t, residual=np.linalg.norm(A.dot(x) - b))
    except Exception as e:
        append("inverse", error=e)

    try:
        x, t = measure_time(lu_solve, A, b)
        append("lu", x=x, t=t, residual=np.linalg.norm(A.dot(x) - b))
    except Exception as e:
        append("lu", error=e)

    try:
        x, t = measure_time(gaussian_elimination, A, b)
        append("gaussian_elimination", x=x, t=t, residual=np.linalg.norm(A.dot(x) - b))
    except Exception as e:
        append("gaussian_elimination", error=e)

    try:
        (xgs, iters, conv), t = measure_time(gauss_seidel, A, b)
        append("gauss_seidel", x=xgs, t=t, residual=np.linalg.norm(A.dot(xgs) - b), iters=iters, converged=conv)
    except Exception as e:
        append("gauss_seidel", error=e)

    return results

def run_benchmarks(sizes=(3, 10, 100), repeats=3, seed=None, dataset_dir=None):
    rows = []
    for rep in range(repeats):
        for n in sizes:
            s = None if seed is None else (seed + rep + n)
            A, b = generate_diagonally_dominant_matrix(n, seed=s)

            # 🔹 NUEVO: Guardar dataset generado
            if dataset_dir:
                dataset_dir.mkdir(parents=True, exist_ok=True)
                np.save(dataset_dir / f"A_{n}x{n}_rep{rep+1}.npy", A)
                np.save(dataset_dir / f"b_{n}x{n}_rep{rep+1}.npy", b)

            out = run_all_methods(A, b)
            for r in out:
                rows.append({
                    "n": n,
                    "method": r["method"],
                    "time_s": r["time_s"],
                    "residual": r["residual"],
                    "iters": r["iters"],
                    "converged": r["converged"],
                    "cond": r["cond"],
                    "error": r["error"],
                    "repeat": rep
                })
    df = pd.DataFrame(rows)
    return df

# -------------------------
# Guardar y graficar
# -------------------------
def save_and_plot(df, results_dir: Path, prefix="linear_solver"):
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{prefix}_results.csv"
    df.to_csv(csv_path, index=False)

    df_plot = df.dropna(subset=["time_s", "residual"]).copy()
    if df_plot.empty:
        print("No hay datos válidos para graficar.")
        return csv_path, None, None

    plt.figure(figsize=(8,4))
    methods = df_plot["method"].unique()
    for m in methods:
        sub = df_plot[df_plot["method"] == m]
        agg = sub.groupby("n")["time_s"].median().reset_index()
        plt.plot(agg["n"], agg["time_s"], marker="o", label=m)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n (matriz n x n)")
    plt.ylabel("Tiempo (s)")
    plt.title("Comparación de tiempos (mediana por n)")
    plt.grid(True)
    plt.legend()
    tpath = results_dir / f"{prefix}_time_comparison.png"
    plt.tight_layout()
    plt.savefig(tpath)
    plt.close()

    plt.figure(figsize=(8,4))
    for m in methods:
        sub = df_plot[df_plot["method"] == m]
        agg = sub.groupby("n")["residual"].median().reset_index()
        plt.plot(agg["n"], agg["residual"], marker="o", label=m)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n (matriz n x n)")
    plt.ylabel("Residual ||Ax - b||")
    plt.title("Comparación de precisión (residual mediano por n)")
    plt.grid(True)
    plt.legend()
    rpath = results_dir / f"{prefix}_residual_comparison.png"
    plt.tight_layout()
    plt.savefig(rpath)
    plt.close()

    return csv_path, tpath, rpath

# -------------------------
# CLI y ejecución principal
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SISTEMA DE ECUACIONES LINEALES - benchmarking")
    p.add_argument("--example5", action="store_true", help="Resolver el ejemplo 5x5 y guardar dataset.")
    p.add_argument("--mm", type=str, default=None, help="Ruta a archivo Matrix Market (.mtx)")
    p.add_argument("--sizes", type=int, nargs="+", default=[3,10,100], help="Tamaños a testear.")
    p.add_argument("--repeats", type=int, default=3, help="Número de repeticiones por tamaño.")
    p.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidad.")
    p.add_argument("--outdir", type=str, default="results", help="Carpeta de salida para CSV y gráficos.")
    p.add_argument("--datasetdir", type=str, default="datasets", help="Carpeta para guardar matrices y vectores generados.")  # 🔹 NUEVO
    p.add_argument("--show", action="store_true", help="Mostrar gráficos en pantalla además de guardarlos.")
    return p.parse_args()

def main():
    args = parse_args()
    results_dir = Path(args.outdir)
    dataset_dir = Path(args.datasetdir)  # 🔹 NUEVO

    print("Scipy disponible para LU:", SCIPY_AVAILABLE)
    print("Scipy.io.mmread disponible:", MMREAD_AVAILABLE)

    if args.example5:
        A5, b5 = example_5x5()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        np.save(dataset_dir / "A_5x5.npy", A5)
        np.save(dataset_dir / "b_5x5.npy", b5)
        print("\nEjemplo 5x5 guardado en carpeta datasets/")
        res5 = run_all_methods(A5, b5)
        print("\nResultados ejemplo 5x5:")
        for r in res5:
            print(f"{r['method']}: time={r['time_s']:.6f}s residual={r['residual']:.3e} cond={r['cond']:.3e}")

    print(f"\nEjecutando benchmarks: sizes={args.sizes} repeats={args.repeats}")
    df = run_benchmarks(sizes=tuple(args.sizes), repeats=args.repeats, seed=args.seed, dataset_dir=dataset_dir)
    print(df.head(10))

    csv_path, tpath, rpath = save_and_plot(df, results_dir, prefix="linear_solver")
    print(f"\n✅ CSV guardado en: {csv_path}")
    print(f"✅ Gráfico tiempos: {tpath}")
    print(f"✅ Gráfico residual: {rpath}")
    print(f"✅ Datasets guardados en: {dataset_dir.resolve()}")

if __name__ == "__main__":
    main()