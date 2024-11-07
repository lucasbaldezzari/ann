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
    
    def classify(self, inputs, proba_threshold=0.5):
        """Forward propagation a través de todas las capas y clasifica los resultados."""
        outputs = self.predict(inputs)
        return (outputs > proba_threshold).astype(int)

    def use(self, loss, loss_prima):
        """Añade la función de pérdida y su derivada a la red."""
        self.loss = loss
        self.loss_prima = loss_prima
    
    def train(self, X, y, learning_rate=0.01, epochs=1000, tolerancia=1e-5, batch_size = 32, imprimir_cada=100):
        """Entrena la red neuronal usando backpropagation."""

        num_samples = X.shape[0]
        self.epoch_loss_hist = [0]

        for epoch in range(epochs):

            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            ##iteramos sobre los batches
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                ##Aplicamos el forward pass
                predictions = self.predict(X_batch)
                batch_loss = self.loss(y_batch, predictions)
                epoch_loss += batch_loss
                # ## Calculamos el error
                mse = self.loss(y_batch, predictions)
            
                ##Aplicamos el algoritmo de backpropagation
                error = self.loss_prima(y_batch, predictions) ##error en la última capa
                for layer in reversed(self._layers):
                    ##sólo las capas densas tienen el método train
                    if isinstance(layer, layers.Dense):
                        error = layer.train(error, learning_rate)

            self.epoch_loss_hist.append(epoch_loss)

            epoch_loss /= (num_samples/batch_size)
            if epoch_loss < tolerancia:
                print(f"Entrenamiento detenido en la época {epoch}, MSE: {mse}")
                break
            
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
    from loss_functions import mse, mse_prima
    from activationsFunctions import *

    nn = NeuralNetwork([layers.Input((2,)),
                        layers.Dense(ninputs=2,noutputs=3,name="Hidden1",activation=tanh,activation_prima=tanh_prima,random_seed=1),
                        layers.Dense(ninputs=3,noutputs=1,name="Output",activation=sigmoid,activation_prima=sigmoid,random_seed=1)])
    
    nn.use(loss=mse, loss_prima=mse_prima)

    x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]]).reshape(4,1)

    #predicciones antes del entrenamiento
    y1 = nn.predict(x_train)
    y1.round(1)

    nn.train(X=x_train, y=y_train, learning_rate=0.01, epochs=int(1e5), tolerancia=1e-5, batch_size=4, imprimir_cada=int(1e3))

    #predicciones después del entrenamiento
    y2 = nn.predict(x_train)
    y2.round(1)

    nn.classify(x_train)

