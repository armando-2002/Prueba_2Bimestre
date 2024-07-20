# -*- coding: utf-8 -*-

"""
Python 3
19 / 07 / 2024
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())

import numpy as np
import matplotlib.pyplot as plt


# ####################################################################
def gauss_jacobi(
    *, A: np.array, b: np.array, x0: np.array, tol: float, max_iter: int
) -> np.array:
    """Resuelve el sistema de ecuaciones lineales Ax = b mediante el método de Jacobi."""

    # --- Validación de los argumentos de la función ---
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A, dtype=float)
    assert A.shape[0] == A.shape[1], "La matriz A debe ser de tamaño n-by-(n)."

    if not isinstance(b, np.ndarray):
        logging.debug("Convirtiendo b a numpy array.")
        b = np.array(b, dtype=float)
    assert b.shape[0] == A.shape[0], "El vector b debe ser de tamaño n."

    if not isinstance(x0, np.ndarray):
        x0 = np.array(x0, dtype=float)
    assert x0.shape[0] == A.shape[0], "El vector x0 debe ser de tamaño n."

    # --- Algoritmo ---
    n = A.shape[0]
    x = x0.copy()
    trajectory = [x.copy().flatten()]
    logging.info(f"i= {0} x: {x.T}")

    for k in range(1, max_iter):
        x_new = np.zeros((n, 1))  # prealloc
        for i in range(n):
            suma = sum([A[i, j] * x[j] for j in range(n) if j != i])
            x_new[i] = (b[i] - suma) / A[i, i]

        trajectory.append(x_new.copy().flatten())

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new.copy()
        logging.info(f"i= {k} x: {x.T}")

    return np.array(trajectory)


# ####################################################################
def gauss_seidel(
    *, A: np.array, b: np.array, x0: np.array, tol: float, max_iter: int
) -> np.array:
    """Resuelve el sistema de ecuaciones lineales Ax = b mediante el método de Gauss-Seidel."""

    # --- Validación de los argumentos de la función ---
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A, dtype=float)
    assert A.shape[0] == A.shape[1], "La matriz A debe ser de tamaño n-by-(n)."

    if not isinstance(b, np.ndarray):
        logging.debug("Convirtiendo b a numpy array.")
        b = np.array(b, dtype=float)
    assert b.shape[0] == A.shape[0], "El vector b debe ser de tamaño n."

    if not isinstance(x0, np.ndarray):
        x0 = np.array(x0, dtype=float)
    assert x0.shape[0] == A.shape[0], "El vector x0 debe ser de tamaño n."

    # --- Algoritmo ---
    n = A.shape[0]
    x = x0.copy()
    trajectory = [x.copy().flatten()]
    logging.info(f"i= {0} x: {x.T}")

    for k in range(1, max_iter):
        x_new = np.zeros((n, 1))  # prealloc
        for i in range(n):
            suma = sum([A[i, j] * x_new[j] for j in range(i) if j != i]) + sum(
                [A[i, j] * x[j] for j in range(i, n) if j != i]
            )
            x_new[i] = (b[i] - suma) / A[i, i]

        trajectory.append(x_new.copy().flatten())

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new.copy()
        logging.info(f"i= {k} x: {x.T}")

    return np.array(trajectory)


# ####################################################################
def plot_trajectory(trajectory_jacobi, trajectory_seidel):
    """Dibuja la trayectoria generada por los métodos de Gauss-Jacobi y Gauss-Seidel."""

    plt.figure(figsize=(12, 6))

    # Trajectory for Gauss-Jacobi
    plt.subplot(1, 2, 1)
    plt.plot(trajectory_jacobi[:, 0], trajectory_jacobi[:, 1], 'o-', label='Gauss-Jacobi')
    plt.title('Gauss-Jacobi')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)

    # Trajectory for Gauss-Seidel
    plt.subplot(1, 2, 2)
    plt.plot(trajectory_seidel[:, 0], trajectory_seidel[:, 1], 'o-', label='Gauss-Seidel')
    plt.title('Gauss-Seidel')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
