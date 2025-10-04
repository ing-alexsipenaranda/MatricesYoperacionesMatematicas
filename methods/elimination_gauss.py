import numpy as np

def gaussian_elimination(A, b):
    """
    Implementación manual del método de eliminación Gaussiana con pivoteo parcial.
    A: matriz cuadrada (numpy array)
    b: vector columna
    Retorna: vector solución x
    """
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)

    for i in range(n):
        # Pivoteo parcial
        max_row = np.argmax(np.abs(A[i:, i])) + i
        if i != max_row:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]

        # Hacer ceros debajo del pivote
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - factor * A[i, i:]
            b[j] = b[j] - factor * b[i]

    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x