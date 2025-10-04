import numpy as np

def generate_diagonally_dominant_matrix(n):
    """
    Genera una matriz cuadrada A (n x n) diagonalmente dominante
    y un vector b con valores aleatorios.
    """
    A = np.random.rand(n, n)
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i])) + 1
    b = np.random.rand(n)
    return A, b