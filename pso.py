import numpy as np

class PSOOptimizer:
    
    def __init__(
        self,
        num_particles=20,
        max_iter=30,
        w=0.7,
        c1=1.5,
        c2=1.5,
        random_state=None,
    ):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.random_state = random_state

        self.history_best = []
        self.best_position_ = None
        self.best_fitness_ = None

    def _init_swarm(self, bounds):
        rng = np.random.default_rng(self.random_state)

        self.dim = len(bounds)
        self.bounds = np.array(bounds, dtype=float)  # shape: (dim, 2)

        low = self.bounds[:, 0]
        high = self.bounds[:, 1]

        positions = rng.uniform(low, high, size=(self.num_particles, self.dim))
        velocities = rng.uniform(
            -(high - low) * 0.1,
            (high - low) * 0.1,
            size=(self.num_particles, self.dim),
        )

        # Limite de velocidad (20% del rango)
        v_max = (high - low) * 0.2

        return positions, velocities, v_max

    def optimize(self, objective_fn, bounds):
        """
        Ejecuta PSO.

        Parameters
        ----------
        objective_fn : callable
            Recibe un vector numpy (posición de la partícula) y devuelve un escalar (fitness).
        bounds : list of (low, high)
            Límites para cada dimensión.

        Returns
        -------
        best_position, best_fitness, history_best
        """
        positions, velocities, v_max = self._init_swarm(bounds)

        # Evaluación inicial
        fitness_values = np.array([objective_fn(p) for p in positions])

        pbest_positions = positions.copy()
        pbest_fitness = fitness_values.copy()

        gbest_index = np.argmin(fitness_values)
        gbest_position = positions[gbest_index].copy()
        gbest_fitness = fitness_values[gbest_index]

        self.history_best = [gbest_fitness]

        rng = np.random.default_rng(self.random_state)

        for it in range(self.max_iter):
            for i in range(self.num_particles):
                r1 = rng.random(self.dim)
                r2 = rng.random(self.dim)

                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (gbest_position - positions[i])

                velocities[i] = self.w * velocities[i] + cognitive + social

                # Limitar velocidad
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                # Actualizar posición
                positions[i] = positions[i] + velocities[i]

                # Respetar límites
                positions[i] = np.clip(positions[i], self.bounds[:, 0], self.bounds[:, 1])

            # Re-evaluar
            fitness_values = np.array([objective_fn(p) for p in positions])

            # Actualizar pbest
            for i in range(self.num_particles):
                if fitness_values[i] < pbest_fitness[i]:
                    pbest_fitness[i] = fitness_values[i]
                    pbest_positions[i] = positions[i].copy()

            # Actualizar gbest
            current_best_index = np.argmin(fitness_values)
            current_best_fit = fitness_values[current_best_index]

            if current_best_fit < gbest_fitness:
                gbest_fitness = current_best_fit
                gbest_position = positions[current_best_index].copy()

            self.history_best.append(gbest_fitness)
            print(f"[PSO] Iter {it + 1}/{self.max_iter} - Best fitness: {gbest_fitness:.4f} - Best global position: {gbest_position}")

        self.best_position_ = gbest_position
        self.best_fitness_ = gbest_fitness
        return gbest_position, gbest_fitness, self.history_best
