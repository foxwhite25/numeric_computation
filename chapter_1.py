from __future__ import annotations

import inspect
import timeit
from typing import Any, Callable, List

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm

SIZE = 1000
MAX_ITER = 100


def cosine_similarity(a, b) -> float:
    return np.dot(a, b) / (norm(a) * norm(b))


def jacobi(a: np.ndarray, b: np.ndarray, n=MAX_ITER) -> (np.ndarray, List[float]):
    """Solves the equation Ax=b via the Jacobi iterative method"""
    a_o = a.copy()
    b_o = b.copy()
    a[[0, 1]] = a[[1, 0]]  # swap to avoid zeros on diag
    b[[0, 1]] = b[[1, 0]]

    sim_list = []
    x = np.zeros_like(b, dtype=np.double)
    t = a - np.diag(np.diagonal(a))
    for k in range(n):
        x[:] = (b - np.dot(t, x)) / np.diagonal(a)
        sim_list.append(cosine_similarity(a_o.dot(x), b_o))
    return x, sim_list


def gauss_seidel(a: np.ndarray, b: np.ndarray, n=MAX_ITER) -> (np.ndarray, List[float]):
    """Solves the equation Ax=b via the Gauss–Seidel method"""
    a_o = a.copy()
    b_o = b.copy()
    a[[0, 1]] = a[[1, 0]]  # swap to avoid zeros on diag
    b[[0, 1]] = b[[1, 0]]

    sim_list = []
    x = np.zeros(len(a[0]))

    for k in range(n):
        x_old = x.copy()
        for i in range(a.shape[0]):
            x[i] = (b[i] - np.dot(a[i, :i], x[:i]) - np.dot(a[i, (i + 1):], x_old[(i + 1):])) / a[i, i]
        sim_list.append(cosine_similarity(a_o.dot(x), b_o))
    return x, sim_list


def sor(a: np.ndarray, b: np.ndarray, n=MAX_ITER) -> (np.ndarray, List[float]):
    """Solves the equation Ax=b via the Successive over-relaxation method"""
    a_o = a.copy()
    b_o = b.copy()
    a[[0, 1]] = a[[1, 0]]  # swap to avoid zeros on diag
    b[[0, 1]] = b[[1, 0]]

    def _sor_solver(_a, _b, omega, _n):
        if omega <= 1 or omega > 2:
            raise ValueError('omega should be inside [1, 2]')
        x = x0 = np.zeros(len(_a[0]))
        sim_list = []

        for step in range(_n):
            for i in range(_b.shape[0]):
                new_values_sum = np.dot(_a[i, :i], x[:i])
                old_values_sum = np.dot(_a[i, i + 1:], x0[i + 1:])
                x[i] = (_b[i] - (old_values_sum + new_values_sum)) / _a[i, i]
                x[i] = np.dot(x[i], omega) + np.dot(x0[i], (1 - omega))
            x0 = x
            sim_list.append(cosine_similarity(a_o.dot(x), b_o))
        return x, sim_list

    return _sor_solver(a, b, 1.1, n)


def conjugate(a: np.ndarray, b: np.ndarray, n=MAX_ITER) -> (np.ndarray, List[float]):
    """Solves the equation Ax=b via the Conjugate gradient method"""
    a_o = a.copy()
    b_o = b.copy()

    x = np.zeros(len(a[0]))
    r = b - np.dot(a, x)
    p = r
    rs_old = np.dot(np.transpose(r), r)
    sim_list = []

    for i in range(n):
        ap = np.dot(a, p)
        alpha = rs_old / np.dot(np.transpose(p), ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, ap)
        rs_new = np.dot(np.transpose(r), r)
        if np.sqrt(rs_new) < 1e-8:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
        sim_list.append(cosine_similarity(a_o.dot(x), b_o))
    return x, sim_list


def conjugate_with_jacobi(a: np.ndarray, b: np.ndarray, n=MAX_ITER) -> (np.ndarray, List[float]):
    """Solves the equation Ax=b via the Conjugate gradient method with Jacobi preconditioner"""
    a_o = a.copy()
    b_o = b.copy()
    a[[0, 1]] = a[[1, 0]]  # swap to avoid zeros on diag
    b[[0, 1]] = b[[1, 0]]

    sim_list = []

    x = np.zeros(len(a[0]))
    r = b - np.dot(a, x)
    p = r
    rs_old = np.dot(np.transpose(r), r)

    # Compute diagonal matrix D from the coefficient matrix a
    d = np.diag(a)

    for i in range(n):
        ap = np.dot(a, p)

        # Apply Jacobi preconditioner
        z = r / d

        alpha = rs_old / np.dot(np.transpose(p), ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, ap)
        rs_new = np.dot(np.transpose(r), r)

        beta = rs_new / rs_old
        p = z + beta * p
        rs_old = rs_new
        sim_list.append(cosine_similarity(a_o.dot(x), b_o))
    return x, sim_list


def create_matrix() -> np.ndarray:
    """Create the A for Ax=b"""
    matrix = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        matrix[i][i] = i
        if i + 1 < SIZE:
            matrix[i][i + 1] = 0.5
            matrix[i + 1][i] = 0.5
        if i + 2 < SIZE:
            matrix[i][i + 2] = 0.5
            matrix[i + 2][i] = 0.5
    return matrix


def create_result() -> np.ndarray:
    """Create the b for Ax=b"""
    result = np.empty(SIZE)
    result.fill(1)
    return result


def run_method(fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray]) -> (float, float, List[float]):
    """Running solver to solve the Ax=b problem with different implementation"""
    a = create_matrix()
    b = create_result()
    name = inspect.getdoc(fn)
    print(f"{name}:")
    start = timeit.default_timer()
    result, sim_list = fn(a, b, MAX_ITER)
    end = timeit.default_timer()
    print("- Result:")
    print(result)
    print("- Result dot a:")
    print(a.dot(result))
    print("- Cosine Similarity:")
    print(cosine_similarity(a.dot(result), b))
    print("- Speed:")
    print(end - start)
    return end - start, cosine_similarity(a.dot(result), b), sim_list


if __name__ == '__main__':
    func = [
        jacobi, gauss_seidel, sor,
        conjugate, conjugate_with_jacobi
    ]
    speed_rank = {}
    sim_rank = {}
    sim_lists = {}
    for fn in func:
        speed, similarity, sim_list = run_method(fn)
        speed_rank[fn.__name__] = speed
        sim_rank[fn.__name__] = similarity
        sim_lists[fn.__name__] = sim_list

    fig = plt.figure()
    for (name, sim_list) in sim_lists.items():
        plt.plot(sim_list, label=name)
    plt.legend()
    plt.suptitle(f"Cosine Similarity of Each algorithm")
    plt.title(f"{MAX_ITER} iteration")
    import tikzplotlib
    tikzplotlib.save("mytikz.tex")
    print("==========================================")

    print("Speed Rank")
    print(dict(sorted(speed_rank.items(), key=lambda item: item[1])))
    print("Similarity Rank")
    print(dict(sorted(sim_rank.items(), key=lambda item: item[1], reverse=True)))
    print(sim_lists)
