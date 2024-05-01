"""Módulo para crear redes neuronales artificiales.
"""

import numpy as np
import Layers as layers
import warnings

class NeuralNetwork:
    """Clase para representar una red neuronal.
    La clase recibe las capas de la red y las entrena.
    """
    def __init__(self, layers:list, name = "NeuralNetwork"):
        """Inicializa una red neuronal con las capas que recibe.
        Args:
        - layers (list): lista con objetos Layers.
        - name: nombre de la red.
        """
        self._layers = layers
        self.name = name

    def add(self, layer):
        self._layers.append(layer)
    
    def predict(self, inputs):
        """Predice el output de la red neuronal. Se toman los datos de entrada y a partir de ahí se computa forward propagation para
        cada capa de la red."""
        outputs = inputs.copy()
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs
    
    def train(self, error):
        """Actualiza pesos y bias usando backpropagation"""
        pass
        
    @property
    def layers(self):
        if len(self._layers) == 0:
            raise ValueError("La red no tiene capas")
        return self._layers
    
    # def __str__(self):
    #     return f"{self.name}({self.layers})"
    
    # def __repr__(self) -> str:
    #     return f"{self.name}({self.layers})"

if __name__ == "__main__":
    from Layers import Input, Dense
    nn = NeuralNetwork([layers.Input((2,)),
                        layers.Dense(ninputs=2,noutputs=3,name="Hidden1",activation="tanh"),
                        layers.Dense(ninputs=3,noutputs=1,name="Output",activation="tanh")])

    x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    y1 = nn.predict(x_train[0])
    y1

