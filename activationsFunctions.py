"""Módulo con las funciones de activación más comunes."""
import numpy as np

def sigmoid(x, derivative = False):
    """Función sigmoide."""
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def tanh(x, derivative = False):
    """Función tangente hiperbólica."""
    if derivative:
        return 1 - x ** 2
    return np.tanh(x)

def relu(x, derivative = False):
    """Función ReLU."""
    if derivative:
        return np.where(x > 0, 1, 0)
    return np.maximum(0, x)

def leaky_relu(x, derivative = False):
    """Función Leaky ReLU."""
    if derivative:
        return np.where(x > 0, 1, 0.01)
    return np.maximum(0.01 * x, x)

def linear(x, derivative = False):
    """Función lineal."""
    if derivative:
        return np.ones_like(x)
    return x

def softmax(x, derivative = False):
    """Función softmax."""
    if derivative:
        return x * (1 - x)
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)

def step(x, derivative = False):
    """Función escalón."""
    if derivative:
        return 0
    return np.where(x >= 0, 1, 0)

def identity(x, derivative = False):
    """Función identidad."""
    if derivative:
        return np.ones_like(x)
    return x

def binary_step(x, derivative = False):
    """Función escalón binario."""
    if derivative:
        return 0
    return np.where(x >= 0, 1, 0)