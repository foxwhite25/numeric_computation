import numpy as np


def gram_schmidt(A):
    Q = np.zeros_like(A, dtype=float)
    R = np.zeros((A.shape[1], A.shape[1]))

    for i in range(A.shape[1]):
        v = A[:, i].astype(float)
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], A[:, i])
            v -= R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]

    return Q, R


def householder_reflection(A):
    (rows, cols) = np.shape(A)
    Q = np.eye(rows)
    R = np.copy(A)

    for i in range(cols):
        x = R[i:, i]
        e = np.zeros_like(x)
        e[0] = 1  # make e a unit vector
        u = x - np.linalg.norm(x) * e
        v = u / np.linalg.norm(u)

        Q_i = np.eye(rows)
        Q_i[i:, i:] -= 2.0 * np.outer(v, v)
        print(f"Householder reflector H_{i+1}:\n", np.around(Q_i, 2))

        R = np.dot(Q_i, R)
        Q = np.dot(Q, Q_i.T)

    return Q, R


n = 0


def run(A, b):
    global n
    n += 1
    print(f"Equation {n}:")
    print(f"A: {A}")
    print(f"b: {b}")
    for func in [householder_reflection, gram_schmidt]:
        print("==========" + func.__name__ + "==========")
        Q, R = func(A)
        print("Q:\n", np.around(Q, 2))
        print("R:\n", np.around(R, 2))
        print("QR =\n", np.around(Q @ R, 2))

        # Compute least squares solution
        x = np.linalg.solve(R[:A.shape[1]], np.dot(Q.T, b)[:A.shape[1]])

        # Compute the 2-norm error
        error = np.linalg.norm(np.dot(A, x) - b)

        print("Least squares solution:\n", x)
        print("2-norm error:\n", error)
    print()


def main():
    run(
        np.array([
            [3, -1, 2],
            [4, 1, 0],
            [-3, 2, 1],
            [1, 1, 5],
            [-2, 0, 3]
        ]),
        np.array([10, 10, -5, 15, 0])
    )
    run(
        np.array([
            [4, -2, 3, 0],
            [-2, 3, -1, 1],
            [1, 3, -4, 2],
            [1, 0, 1, -1],
            [3, 1, 3, -2]
        ]),
        np.array([10, 0, 2, 0, 5])
    )


if __name__ == '__main__':
    main()
