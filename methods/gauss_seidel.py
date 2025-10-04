import numpy as np

def gauss_seidel(A, b, tol=1e-10, max_iter=1000):
    """
    Implementación del método iterativo de Gauss-Seidel
    Retorna la solución, número de iteraciones y convergencia.
    """
    n = len(b)
    x = np.zeros(n)
    converged = False

    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        # Condición de parada
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            converged = True
            break
        x = x_new
    return x, k+1, converged