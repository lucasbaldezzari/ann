import numpy as np
from activationsFunctions import *

class Input:
    """Clase para representar una capa de entrada de una red neuronal.
    La clase sólo recibe los inputs y los pasa a la siguiente capa.
    """
    def __init__(self, input_shape:tuple, name = "InputLayer"):
        """Inicializa una capa de entrada con la forma de los inputs y la forma esperada de los inputs.
        
        Args:
        - input_shape (tupla): tupla con forma de los inputs. Máximo dos dimensiones.
        - name: nombre de la capa.
        """
        if len(input_shape) > 2:
            raise ValueError("La forma de los inputs no puede ser mayor a dos dimensiones")
        self.input_shape = input_shape
        self.name = name

    def predict(self, inputs):
        """Toma los inputs y los pasa a la siguiente capa
        Args:
        - inputs (np.ndarray): array con los inputs. No puede ser mayor a dos dimensiones.

        Returns:
        - np.ndarray: array con los inputs.
        """
        ##check if the input shape is the same as the target shape
        if inputs.shape != self.input_shape:
            raise ValueError(f"La forma de los inputs ({inputs.shape}) no coincide con la forma esperada ({self.input_shape})")
        return inputs
    
    def train(self, error):
        """Actualiza pesos y bias usando backpropagation"""
        pass
    
    def __str__(self):
        return f"InputLayer({self.input_shape})"
    
    def __repr__(self) -> str:
        return f"InputLayer({self.input_shape})"
    
class Dense:
    """Clase para representar una capa oculta de una red neuronal.
    La clase recibe los inputs y los pasa a los perceptrones de la capa.
    """
    def __init__(self, ninputs:int, noutputs:int, activation:str = "sigmoid", weigths_init:str = "uniform",bias_init:str = "zeros", name = "DenseLayer"):
        """Inicializa una capa oculta con el número de nodos, función de activación y tipo de inicialización de pesos y bias.
        
        Args:
        - inputs (int): número de nodos de la capa anterior
        - outputs (int): número de nodos de la capa actual.
        - activation (str): función de activación.
        - weigths_init (str): tipo de inicialización de pesos.
        - bias_init (str): tipo de inicialización de bias.
        - name: nombre de la capa.
        """
        self._inputs = ninputs
        self._outputs = noutputs
        self._activation = activation
        activation_names = ["sigmoid", "tanh", "relu", "leaky_relu", "softmax", "linear", "step", "identity", "binary_step"]
        activation_functions_list = [sigmoid, tanh, relu, leaky_relu, softmax, linear, step, identity, binary_step]
        if activation in activation_names:
            self._activation_function = activation_functions_list[activation_names.index(activation)]()
        else:
            raise ValueError(f"La función de activación {activation} no es una función válida. Utilizar: ", *activation_names)
        self._weigths_init = weigths_init
        self._weights = self.__set_weights()
        self._bias_init = bias_init
        self._bias = self.__set_bias()
        self.name = name

    def __set_weights(self):
        """Inicializa los pesos de la capa"""
        if self._weigths_init == "uniform":
            weights = np.random.uniform(-1,1,(self._inputs, self._outputs))
        if self._weigths_init == "zeros":
            weights = np.zeros((self._inputs, self._outputs))
        return weights

    def __set_bias(self):
        """Inicializa los bias de la capa"""
        if self._bias_init == "zeros":
            bias = np.zeros(self._outputs)
        if self._bias_init == "random":
            bias = np.random.random(self._outputs)
        return bias
    
    def predict(self, inputs):
        """Toma los inputs y los pasa a los perceptrones de la capa.
        Args:
        - inputs (np.ndarray): array con los inputs.

        Returns:
        - np.ndarray: array con los outputs de los perceptrones.
        """
        return self._activation_function(np.dot(inputs, self._weights)+self.bias)
    
    def train(self, error):
        """Actualiza pesos y bias usando backpropagation"""
        pass

    @property
    def perceptrons(self):
        return self._perceptrons
    
    @property
    def weights(self):
        return self._weights
    
    @property
    def bias(self):
        return self._bias
    
    def __str__(self):
        return f"{self.name}({self._inputs},{self._outputs})"
    
    def __repr__(self) -> str:
        return f"{self.name}({self._inputs},{self._outputs})"
