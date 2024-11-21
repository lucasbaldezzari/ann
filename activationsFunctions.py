"""Módulo con las funciones de activación más comunes."""
import numpy as np

def sigmoid(x, clipper=500):
    """Función sigmoide."""
    x = np.clip(x, -1*clipper, clipper)
    return 1 / (1 + np.exp(-x))

def sigmoid_prima(x):
    """Derivada de la función sigmoide."""
    s = sigmoid(x)
    return s* (1 - s)

def relu(x):
    """Función ReLU."""
    return np.maximum(0, x)

def relu_prima(x):
    """Derivada de la función ReLU."""
    return np.where(x >= 0, 1, 0)

def tanh(x):
    """Función tangente hiperbólica."""
    return np.tanh(x)

def tanh_prima(x):
    """Derivada de la función tangente hiperbólica."""
    return 1 - np.tanh(x)**2

def softmax(x, clipper=500):
    """Función softmax."""
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=1, keepdims=True)

def softmax_prima(x, clipper=500):
    """Derivada de la función softmax."""
    x = np.clip(x, -1*clipper, clipper)
    if x.shape[1] == 1:
        return x * (1 - x)
    else:
        probas = np.zeros(x.shape)
        for i in range(x.shape[0]):
            probas[i] = np.exp(x[i]) / np.sum(np.exp(x[i]))

        return probas

def linear(x):
    """Función lineal."""
    return x

def linear_prima(x):
    """Derivada de la función lineal."""
    return np.ones_like(x)

def leaky_relu(x, alpha=0.01):
    """Función Leaky ReLU."""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_prima(x, alpha=0.01):
    """Derivada de la función Leaky ReLU."""
    return np.where(x > 0, 1, alpha)

