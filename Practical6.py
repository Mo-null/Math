import numpy as np

def is_diagonalizable(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    for eigenvalue in eigenvalues:
        eigenspace_matrix = matrix - eigenvalue * np.eye(matrix.shape[0])
        if np.linalg.matrix_rank(eigenspace_matrix) < matrix.shape[0]:
            print("The matrix is diagonalizable.")
            return True

    print("The matrix is not diagonalizable.")
    return False

def find_eigenvalues(matrix):
    eigenvalues, _ = np.linalg.eig(matrix)
    return eigenvalues

matrix = np.array([[4, 1], [2, 3]])

is_diagonalizable(matrix)

eigenvalues = find_eigenvalues(matrix)
print("Eigenvalues:", eigenvalues)