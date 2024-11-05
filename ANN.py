"""Módulo para crear redes neuronales artificiales.
"""

import numpy as np
import Layers as layers

class NeuralNetwork:
    """Clase para representar una red neuronal."""
    def __init__(self, layers=None, name="NeuralNetwork"):
        self._layers = layers if layers is not None else []
        self.name = name

    def add(self, layer):
        """Añade una capa a la red neuronal."""
        if not isinstance(layer, (Input, Dense)):
            raise TypeError("La capa debe ser una instancia de Input o Dense.")
        self._layers.append(layer)
    
    def predict(self, inputs):
        """Forward propagation a través de todas las capas."""
        outputs = inputs
        for layer in self._layers:
            outputs = layer.predict(outputs)
        return outputs
    
    def train(self, X, y, learning_rate=0.01, epochs=1000, tolerancia=1e-5, imprimir_cada=100):
        """Entrena la red neuronal usando backpropagation."""
        for epoch in range(epochs):
            ##Aplicamos el forward pass
            predictions = self.predict(X)
            ## Calculamos el error
            loss = predictions - y
            
            # Condición de parada si la pérdida es suficientemente baja
            mse = np.mean(loss**2)
            if mse < tolerancia:
                print(f"Entrenamiento detenido en la época {epoch}, MSE: {mse}")
                break
            
            ##Aplicamos el algoritmo de backpropagation
            error = loss ##error en la última capa
            for layer in reversed(self._layers):
                ##sólo las capas densas tienen el método train
                if isinstance(layer, layers.Dense):
                    error = layer.train(error, learning_rate)
            
            
            if epoch % imprimir_cada == 0:
                print(f"Época {epoch}, MSE: {mse}")

    @property
    def layers(self):
        """Devuelve las capas de la red o lanza un error si está vacía."""
        if len(self._layers) == 0:
            raise ValueError("La red no contiene capas. Use el método 'add' para añadir capas.")
        return self._layers

    def __str__(self):
        return f"{self.name} con {len(self._layers)} capas"

    def __repr__(self):
        return f"NeuralNetwork(nombre={self.name}, capas={len(self._layers)})"

if __name__ == "__main__":
    from Layers import Input, Dense
    nn = NeuralNetwork([layers.Input((2,)),
                        layers.Dense(ninputs=2,noutputs=3,name="Hidden1",activation="relu",random_seed=0),
                        layers.Dense(ninputs=3,noutputs=1,name="Output",activation="sigmoid",random_seed=0)])

    x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]]).reshape(4,1)

    #predicciones antes del entrenamiento
    y1 = nn.predict(x_train)
    y1.round(1)

    nn.train(X=x_train, y=y_train, learning_rate=0.1, epochs=10000, tolerancia=1e-5)

    #predicciones después del entrenamiento
    y2 = nn.predict(x_train)
    y2.round(1)

