import numpy as np

def check_linear_dependence(vectors):
    matrix = np.array(vectors).T 
    rank = np.linalg.matrix_rank(matrix)
    return rank < len(vectors)

def linear_combination(vectors, coefficients):
    result = np.dot(coefficients, np.array(vectors))
    return result.tolist()

vectors = [[1, 2, 3], [2, 4, 6], [3, 6, 9]] 
coefficients = [1, -2, 1]  

if check_linear_dependence(vectors):
    print("The vectors are linearly dependent.")
else:
    print("The vectors are linearly independent.")

linear_combo = linear_combination(vectors, coefficients)
print("Linear combination result:", linear_combo)
