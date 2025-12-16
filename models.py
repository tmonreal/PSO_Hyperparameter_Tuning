from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

class ModelWrapper(ABC):

    def __init__(self, X, y, alpha_complexity=0.0, cv=5, random_state=0):
        self.X = X
        self.y = y
        self.alpha_complexity = alpha_complexity
        self.cv = cv
        self.random_state = random_state

    @property
    @abstractmethod
    def param_bounds(self):
        """
        Devuelve lista de (low, high) para cada hiperparámetro.
        """
        pass

    @abstractmethod
    def decode(self, x):
        """
        Convierte un vector continuo x en un dict de hiperparámetros para el modelo de sklearn.
        """
        pass

    @abstractmethod
    def build_model(self, params):
        """
        A partir de un dict de hiperparámetros, construye y devuelve el modelo sklearn.
        """
        pass

    @abstractmethod
    def complexity(self, params):
        """
        Devuelve una medida escalar de complejidad del modelo (cuanto más grande, más complejo).
        Se usa para penalizar modelos demasiado complejos en la fitness.
        """
        pass

    def evaluate(self, x):
        """
        Función objetivo para PSO: dado un vector de hiperparámetros (continuo),
        entrena y evalúa el modelo con CV, y devuelve un fitness a MINIMIZAR.

        Por defecto:
        fitness = 1 - accuracy_mean + alpha_complexity * complexity
        """
        params = self.decode(x)
        model = self.build_model(params)

        scores = cross_val_score(
            model,
            self.X,
            self.y,
            cv=self.cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        acc = scores.mean()
        comp = self.complexity(params)

        fitness = 1.0 - acc + self.alpha_complexity * comp
        return fitness

class KNNWrapper(ModelWrapper):
    """
    Hiperparámetros:
      x[0] -> n_neighbors (int, 1-50)
      x[1] -> p (int, 1-3)  distancia Minkowski
      x[2] -> weights_code (float, 0-1)  <0.5 = 'uniform', >=0.5 = 'distance'
    """

    @property
    def param_bounds(self):
        return [
            (1, 50),   # n_neighbors
            (1, 3),    # p
            (0.0, 1.0) # weights code
        ]

    def decode(self, x):
        x = np.array(x, dtype=float)

        # clip a los límites
        bounds = np.array(self.param_bounds, dtype=float)
        x = np.clip(x, bounds[:, 0], bounds[:, 1])

        n_neighbors = int(round(x[0]))
        p = int(round(x[1]))
        w_code = x[2]
        weights = "uniform" if w_code < 0.5 else "distance"

        # corregir mínimos
        n_neighbors = max(n_neighbors, 1)
        p = max(min(p, 3), 1)

        return {
            "n_neighbors": n_neighbors,
            "p": p,
            "weights": weights,
        }

    def build_model(self, params):
        return KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            p=params["p"],
            weights=params["weights"],
        )

    def complexity(self, params):
        # cuanto mayor k, más costoso + más "suavizado"
        k = params["n_neighbors"]
        k_max = self.param_bounds[0][1]
        return k / k_max

class DecisionTreeWrapper(ModelWrapper):
    """
    Hiperparámetros:
      x[0] -> max_depth (int, 1-20)
      x[1] -> min_samples_split (int, 2-50)
      x[2] -> min_samples_leaf (int, 1-20)
      x[3] -> max_features_frac (float, 0.2-1.0)
    """

    @property
    def param_bounds(self):
        return [
            (1, 20),     # max_depth
            (2, 50),     # min_samples_split
            (1, 20),     # min_samples_leaf
            (0.2, 1.0),  # max_features_frac
        ]

    def decode(self, x):
        x = np.array(x, dtype=float)
        bounds = np.array(self.param_bounds, dtype=float)
        x = np.clip(x, bounds[:, 0], bounds[:, 1])

        max_depth = int(round(x[0]))
        min_samples_split = int(round(x[1]))
        min_samples_leaf = int(round(x[2]))
        max_features = float(x[3])

        min_samples_split = max(min_samples_split, 2)
        min_samples_leaf = max(min_samples_leaf, 1)

        return {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }

    def build_model(self, params):
        return DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=self.random_state,
        )

    def complexity(self, params):
        # penalizamos árboles muy profundos
        depth = params["max_depth"]
        depth_max = self.param_bounds[0][1]
        return depth / depth_max

class MLPWrapper(ModelWrapper):
    """
    Hiperparámetros:

      x[0] -> n_hidden (int, 5-100)
      x[1] -> log10(alpha)          (float, -5 a 1)
      x[2] -> log10(learning_rate)  (float, -4 a -1)

    Trabajo en log10 para recorrer varias órdenes de magnitud.
    """

    @property
    def param_bounds(self):
        return [
            (5, 100),    # n_hidden
            (-5, 1),     # log10(alpha)
            (-4, -1),    # log10(learning_rate_init)
        ]

    def decode(self, x):
        x = np.array(x, dtype=float)
        bounds = np.array(self.param_bounds, dtype=float)
        x = np.clip(x, bounds[:, 0], bounds[:, 1])

        n_hidden = int(round(x[0]))
        log_alpha = x[1]
        log_lr = x[2]

        alpha = 10 ** log_alpha
        learning_rate_init = 10 ** log_lr

        n_hidden = max(n_hidden, 5)

        return {
            "hidden_layer_sizes": (n_hidden,),
            "alpha": alpha,
            "learning_rate_init": learning_rate_init,
        }

    def build_model(self, params):
        return MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            max_iter=80,            # mantener bajo para que PSO sea rápido
            early_stopping=True,
            random_state=self.random_state,
        )

    def complexity(self, params):
        # complejidad ~ número de neuronas en la capa oculta
        n_hidden = params["hidden_layer_sizes"][0]
        n_max = self.param_bounds[0][1]
        return n_hidden / n_max
