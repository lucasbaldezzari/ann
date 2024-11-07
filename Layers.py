import numpy as np

##https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65

class Input:
    """Clase para representar una capa de entrada de una red neuronal.
    La clase sólo recibe los inputs y los pasa a la siguiente capa.
    """
    def __init__(self, n_neuronas: int, name = "InputLayer"):
        """Inicializa una capa de entrada con la forma de los inputs y la forma esperada de los inputs.
        
        Args:
        - n_neuronas (entero): Cantidad de neuronas de la capa.
        - name: nombre de la capa.
        """

        self.n_neuronas = n_neuronas
        self.name = name

    def predict(self, X):
        """
        Valida y pasa los inputs a la siguiente capa.

        Args:
        - X (np.ndarray): array con los inputs (de una o dos dimensiones).

        Returns:
        - np.ndarray: array con los inputs si son compatibles con n_neuronas.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X debe ser un array de numpy")
        ##chqueo que X no tenga más de dos dimensiones
        if X.ndim > 2:
            raise ValueError("X debe tener una o dos dimensiones")

        return X
    
    def __str__(self):
        return f"InputLayer({self.n_neuronas})"
    
    def __repr__(self) -> str:
        return f"InputLayer({self.n_neuronas})"

class Dense:
    """Clase para representar una capa densa en una red neuronal."""
    def __init__(self, ninputs: int, noutputs: int, activation, activation_prima,
                 weights_init="uniform", bias_init="zeros", name="DenseLayer",
                 random_seed = None):
        self._inputs = ninputs
        self._outputs = noutputs
        self._activation = activation
        self.random_seed = random_seed

        self.activation = activation
        self.activation_prima = activation_prima
        
        self._weights_init = weights_init
        self._weights = self.__set_weights()
        self._bias_init = bias_init
        self._bias = self.__set_bias()
        self.name = name

    def __set_weights(self):
        """Inicializa los pesos de acuerdo al método especificado."""
        np.random.seed(self.random_seed)
        if self._weights_init == "uniform":
            weights = np.random.uniform(-0.5, 0.5, (self._inputs, self._outputs))
        elif self._weights_init == "zeros":
            weights = np.zeros((self._inputs, self._outputs))
        else:
            raise ValueError("Método de inicialización de pesos no reconocido. Usar 'uniform' o 'zeros'.")
        return weights

    def __set_bias(self):
        """Inicializa el bias de acuerdo al método especificado."""
        np.random.seed(self.random_seed)
        if self._bias_init == "zeros":
            bias = np.zeros(self._outputs)
        elif self._bias_init == "random":
            bias = np.random.random(-0.5, 0.5,self._outputs)
        elif self._bias_init == "ones":
            bias = np.ones(self._outputs)
        else:
            raise ValueError("Método de inicialización de bias no definido. Usar 'zeros', 'random' o 'ones'.")
        return bias
    
    def predict(self, X):
        """Forward propagation: Calcula la salida de la capa."""
        self.inputs = X
        self.z = np.dot(self.inputs, self._weights) #+ self._bias
        self.a = self.activation(self.z)
        return self.a
    
    def train(self, error, learning_rate=0.01):
        """Backpropagation: Actualiza los pesos y el bias."""

        delta = error * self.activation_prima(self.z)
        input_error = np.dot(delta, self._weights.T)
        weights_update = np.dot(self.inputs.T, delta)

        self._weights -= learning_rate * weights_update
        self._bias -= learning_rate * delta.sum(axis=0)

        return input_error

    @property
    def weights(self):
        return self._weights
    
    @property
    def bias(self):
        return self._bias
    
    def __str__(self):
        return f"{self.name}({self._inputs},{self._outputs})"
    
    def __repr__(self):
        return f"{self.name}({self._inputs},{self._outputs})"