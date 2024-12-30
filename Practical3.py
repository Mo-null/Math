import numpy as np

def gauss_elimination(A, B):
 
    n = len(B)
    augmented_matrix = [A[i] + [B[i]] for i in range(n)]
    
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(augmented_matrix[r][i]))
        augmented_matrix[i], augmented_matrix[max_row] = augmented_matrix[max_row], augmented_matrix[i]

        pivot = augmented_matrix[i][i]
        if pivot == 0:
            raise ValueError("Matrix is singular and cannot be solved.")
        augmented_matrix[i] = [x / pivot for x in augmented_matrix[i]]

        for j in range(i + 1, n):
            factor = augmented_matrix[j][i]
            augmented_matrix[j] = [
                augmented_matrix[j][k] - factor * augmented_matrix[i][k] for k in range(n + 1)
            ]

    X = [0] * n
    for i in range(n - 1, -1, -1):
        X[i] = augmented_matrix[i][-1] - sum(
            augmented_matrix[i][j] * X[j] for j in range(i + 1, n)
        )
    return X

A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
B = [8, -11, -3]

solution = gauss_elimination(A, B)
print("Solution:", solution)

