"""Módulo para crear redes neuronales artificiales.

Las clases son:

- Perceptron
- InputLayer
- HiddenLayer
- OutputLayer
- NeuralNetwork
"""

import numpy as np
import random
import math
import copy

from activationsFunctions import *

class Perceptron:
    """Clase que representa un perceptrón.

    Atributos:
    - weights: array con los pesos de las conexiones de entrada.
    - bias: sesgo del perceptrón.
    - activation_function: función de activación.
    - learning_rate: tasa de aprendizaje.
    """

    def __init__(self, weights:np.ndarray = None, bias:float = None, learning_rate:float = 0.1, activation_function:str = "sigmoid"):
        """Inicializa un perceptrón con los valores dados."""
        self._weights = weights
        self._bias = bias
        self.learning_rate = learning_rate
        activation_names = ["sigmoid", "tanh", "relu", "leaky_relu", "softmax", "linear", "step", "identity", "binary_step"]
        activation_functions_list = [sigmoid, tanh, relu, leaky_relu, softmax, linear, step, identity, binary_step]
        if activation_function in activation_names:
            self.activation_function = activation_functions_list[activation_names.index(activation_function)]
        else:
            raise ValueError(f"La función de activación {activation_function} no es una función válida. Utilizar: ", *activation_names)
        
    def predict(self, inputs):
        """Toma los pesos y bias y genera una salida con los inputs dados"""
        return self.activation_function(np.dot(inputs, self._weights)+self.bias)
    
    def train(self, error):
        """Actualiza pesos y bias"""
        pass

    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        if weights is None:
            self._weights = np.random.random(10)
        else:
            self._weights = weights

class InputLayer:
    """Clase para representar una capa de entrada de una red neuronal.
    La clase sólo recibe los inputs y los pasa a la siguiente capa.
    """
    def __init__(self, inputs:np.ndarray):
        #chequeamos que los inputs sean un array de numpy de 1 dimensión
        if not isinstance(inputs, np.ndarray):
            raise ValueError("Los inputs deben ser un array de numpy")
        if len(inputs.shape) != 1:
            raise ValueError("Los inputs deben ser un array de 1 dimensión")
        self.inputs = inputs

    def forward(self):
        return self.inputs
    
    def __str__(self):
        return f"InputLayer({self.inputs})"
    
    def __repr__(self) -> str:
        return f"InputLayer({self.inputs})"
    
class HiddenLayer:
    """Clase para representar una capa oculta de una red neuronal.
    La clase recibe los inputs y los pasa a los perceptrones de la capa.
    """
    def __init__(self, perceptrons:list):
        self.perceptrons = perceptrons
    
    def forward(self, inputs):
        outputs = []
        for perceptron in self.perceptrons:
            outputs.append(perceptron.predict(inputs))
        return np.array(outputs)
    
    def __str__(self):
        return f"HiddenLayer({self.perceptrons})"
    
    def __repr__(self) -> str:
        return f"HiddenLayer({self.perceptrons})"

if __name__ == "__main__":
    np.random.seed(4)
    w1 = np.random.random(10)
    b1 = np.random.random()
    inputs = np.array([-1,-1,-0.5,0.1,-1,0,-1,0.5,-0.1,0.0])

    p1 = Perceptron(w1,b1, activation_function="sigmoid")
    p2 = Perceptron(w1,b1, activation_function="sigmoid")

    entrada = InputLayer(inputs)
    entrada

    matrix = np.vstack((p1.weights,p2.weights))
    np.dot(matrix, inputs)

    p1.predict(inputs)

    