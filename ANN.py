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

    def __init__(self, weights:np.ndarray, bias:float, learning_rate:float = 0.1, activation_function:str = "sigmoid"):
        """Inicializa un perceptrón con los valores dados."""
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        activation_names = ["sigmoid", "tanh", "relu", "leaky_relu", "softmax", "linear", "step", "identity", "binary_step"]
        activation_functions_list = [sigmoid, tanh, relu, leaky_relu, softmax, linear, step, identity, binary_step]
        if activation_function in activation_names:
            self.activation_function = activation_functions_list[activation_names.index(activation_function)]
        else:
            raise ValueError(f"La función de activación {activation_function} no es una función válida. Utilizar: ", *activation_names)
        
    def predict(self, inputs):
        """Toma los pesos y bias y genera una salida con los inputs dados"""
        return self.activation_function(np.dot(inputs, self.weights)+self.bias)
    
    def train(self, error):
        """Actualiza pesos y bias"""
        pass


if __name__ == "__main__":
    np.random.seed(4)
    w1 = np.random.random(10)
    b1 = np.random.random()
    inputs = np.array([-1,-1,-0.5,0.1,-1,0,-1,0.5,-0.1,0.0])

    p1 = Perceptron(w1,b1, activation_function="sigmoid")
    p2 = Perceptron(w1,b1, activation_function="sigmoid")

    matrix = np.vstack((p1.weights,p2.weights))
    np.dot(matrix, inputs)

    p1.predict(inputs)

    