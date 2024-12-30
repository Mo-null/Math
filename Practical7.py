import numpy as np

def gradient(f, x, h=1e-4):
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        x_plus, x_minus = np.copy(x), np.copy(x)
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

def divergence(f, x, h=1e-4):
    div = 0
    for i in range(x.shape[0]):
        x_plus, x_minus = np.copy(x), np.copy(x)
        x_plus[i] += h
        x_minus[i] -= h
        div += (f(x_plus)[i] - f(x_minus)[i]) / (2 * h)
    return div

def curl(f, x, h=1e-4):
    c = np.zeros(2)
    for i in range(x.shape[0]):
        x_plus, x_minus = np.copy(x), np.copy(x)
        x_plus[i] += h
        x_minus[i] -= h
        c[i] = (f(x_plus)[(i + 1) % 2] - f(x_minus)[(i + 1) % 2]) / (2 * h)
    c[1], c[0] = -c[0], c[1]
    return c

def f_scalar(x):
    return x[0]**2 + x[1]**2

def f_vector(x):
    return np.array([x[0] + x[1], x[0] - x[1]])

x = np.array([1, 2])
print("Gradient:", gradient(f_scalar, x))
print("Divergence:", divergence(f_vector, x))
print("Curl:", curl(f_vector, x))
