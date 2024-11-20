import numpy as np

def mse(y_true, y_pred):
    ##chequeo si las dimensiones son compatibles
    if y_true.shape != y_pred.shape:
        raise ValueError("Las dimensiones de y_true y y_pred no son compatibles")
    ##si las dimensiones son compatibles, calculo el error. Tener en cuenta que y_true e y_pred son arrays de numpy
    ##y pueden ser de una o dos dimensiones
    ##si tienen una dimensión, calculo el error como si fueran vectores
    if y_true.ndim == 1:
        return np.mean(np.power(y_true - y_pred, 2))
    ##si tienen dos dimensiones, calculo el error como si fueran matrices
    else:
        return np.mean(np.power(y_true - y_pred, 2), axis=0)
    

def mse_prima(y_true, y_pred):
    """
    Calcula 2*(y_pred-y_true)/y_true.size
    """
    ##chequeo si las dimensiones son compatibles
    if y_true.shape != y_pred.shape:
        raise ValueError("Las dimensiones de y_true y y_pred no son compatibles")
    ##si las dimensiones son compatibles, calculo el gradiente de la función de pérdida. Tener en cuenta que y_true e y_pred son arrays de numpy
    ##y pueden ser de una o dos dimensiones
    ##si tienen una dimensión, calculo el gradiente como si fueran vectores
    if y_true.ndim == 1:
        return 2*(y_pred-y_true)/y_true.size
    ##si tienen dos dimensiones, calculo el gradiente como si fueran matrices
    else:
        return 2*(y_pred-y_true)/y_true.shape[0]