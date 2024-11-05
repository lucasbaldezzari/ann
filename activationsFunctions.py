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


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4])
    s = sigmoid()
    print(s(x))
    print(s.derivative(x))
    
    t = tanh()
    print(t(x))
    print(t.derivative(x))
    
    r = relu()
    print(r(x))
    print(r.derivative(x))
    
    lr = leaky_relu()
    print(lr(x))
    print(lr.derivative(x))
    
    l = linear()
    print(l(x))
    print(l.derivative(x))
    
    so = softmax()
    print(so(x))
    print(so.derivative(x))
    
    st = step()
    print(st(x))
    print(st.derivative(x))
    
    i = identity()
    print(i(x))
    print(i.derivative(x))
    
    bs = binary_step()
    print(bs(x))
    print(bs.derivative(x))