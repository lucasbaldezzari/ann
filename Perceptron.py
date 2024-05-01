
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

    def __init__(self, weights:np.ndarray, bias:float, learning_rate:float = 0.1, activation_function:str = "sigmoid", name = "Perceptron"):
        """Inicializa un perceptrón con los valores dados."""

        self._weights = weights
        self._bias = bias
        self._learning_rate = learning_rate
        self.name = name
        activation_names = ["sigmoid", "tanh", "relu", "leaky_relu", "softmax", "linear", "step", "identity", "binary_step"]
        activation_functions_list = [sigmoid, tanh, relu, leaky_relu, softmax, linear, step, identity, binary_step]
        if activation_function in activation_names:
            self._activation_function = activation_functions_list[activation_names.index(activation_function)]()
            self._activation_function_name = activation_function
        else:
            raise ValueError(f"La función de activación {activation_function} no es una función válida. Utilizar: ", *activation_names)
        
    def predict(self, inputs):
        """Toma los pesos y bias y genera una salida con los inputs dados"""
        return self._activation_function(np.dot(inputs, self._weights)+self.bias)
    
    def train(self, error):
        """Actualiza pesos y bias usando backpropagation"""

    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        if weights is None:
            self._weights = np.random.random(10)
        else:
            self._weights = weights

    @property
    def bias(self):
        return self._bias
    
    @bias.setter
    def bias(self, bias):
        if bias is None:
            self._bias = np.random.random()
        else:
            self._bias = bias

    @property
    def activation_function_name(self):
        return self._activation_function_name
    
    @property
    def learning_rate(self):
        return self._learning_rate
    
    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    def __str__(self):
        return f"{self.name}({self._weights}, {self._bias}, {self._activation_function_name}, {self.learning_rate})"
    
    def __repr__(self) -> str:
        return f"{self.name}({self._weights}, {self._bias}, {self._activation_function_name}, {self.learning_rate})"
