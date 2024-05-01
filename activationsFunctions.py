"""Módulo con las funciones de activación más comunes."""
import numpy as np

class sigmoid:
    """Función sigmoide."""
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        return x * (1 - x)
    
class tanh:
    """Función tangente hiperbólica."""
    def __call__(self, x):
        return np.tanh(x)
    
    def derivative(self, x):
        return 1 - x ** 2
    
class relu:
    """Función ReLU."""
    def __call__(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return np.where(x > 0, 1, 0)

class leaky_relu:
    """Función Leaky ReLU."""
    def __call__(self, x):
        return np.maximum(0.01 * x, x)
    
    def derivative(self, x):
        return np.where(x > 0, 1, 0.01)
    
class linear:
    """Función lineal."""
    def __call__(self, x):
        return x
    
    def derivative(self, x):
        return np.ones_like(x)
    
class softmax:
    """Función softmax."""
    def __call__(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)
    
    def derivative(self, x):
        return x * (1 - x)
    
class step:
    """Función escalón."""
    def __call__(self, x):
        return np.where(x >= 0, 1, 0)
    
    def derivative(self, x):
        return 0
    
class identity:
    """Función identidad."""
    def __call__(self, x):
        return x
    
    def derivative(self, x):
        return np.ones_like(x)
    
class binary_step:
    """Función escalón binario."""
    def __call__(self, x):
        return np.where(x >= 0, 1, 0)
    
    def derivative(self, x):
        return 0
