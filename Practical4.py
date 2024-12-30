import numpy as np
A = np.array([[4, 3, 2], [-2, 2, 3], [3, -5, 2]])
B = np.array([25, -10, -4])
X2 = np.linalg.solve(A,B)

print(X2)
import numpy as np

A = np.array([[4, 3, 2],
              [-2, 2, 3],
              [3, -5, 2]], dtype=float)

def gauss_jordan(A):
    n = len(A)
    
    for i in range(n):

        A[i] = A[i] / A[i][i]
        
        for j in range(n):
            if i != j:
                A[j] = A[j] - A[i] * A[j][i]
                
    return A

result = gauss_jordan(A)

solution = result[:, -1]
print("Solution of the system is:")
print(solution)