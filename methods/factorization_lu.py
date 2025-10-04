import numpy as np
from scipy.linalg import lu_factor, lu_solve

def lu_method(A, b):
    """
    Método de factorización LU usando scipy.linalg.lu_factor y lu_solve
    """
    lu, piv = lu_factor(A)
    x = lu_solve((lu, piv), b)
    return x