import matplotlib.pyplot as plt
from pso import PSOOptimizer
from sklearn.model_selection import cross_val_score

def run_experiment(model_wrapper_cls, X, y, model_name, dataset_name,
                   alpha_complexity=0.0, pso_config=None, random_state=0):
    """
    Corre PSO para un modelo y dataset dados.
    Devuelve best_fitness, best_accuracy y la curva de convergencia.
    """
    if pso_config is None:
        pso_config = dict(
            num_particles=20,
            max_iter=30,
            w=0.7,
            c1=1.5,
            c2=1.5,
            random_state=random_state,
        )

    # instanciar wrapper
    wrapper = model_wrapper_cls(
        X=X,
        y=y,
        alpha_complexity=alpha_complexity,
        cv=5,
        random_state=random_state,
    )

    bounds = wrapper.param_bounds

    # función objetivo para PSO
    def objective_fn(x):
        return wrapper.evaluate(x)

    optimizer = PSOOptimizer(**pso_config)
    best_pos, best_fit, history = optimizer.optimize(objective_fn, bounds)

    # Para reportar accuracy: recomputamos sin penalización
    # (fitness = 1 - acc + alpha * comp -> acc = 1 - (fitness - alpha * comp))
    params = wrapper.decode(best_pos)
    model = wrapper.build_model(params)
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    best_acc = scores.mean()

    print(f"\n=== Resultados {model_name} en {dataset_name} ===")
    print("Mejores hiperparámetros:", params)
    print(f"Mejor fitness (con penalización): {best_fit:.4f}")
    print(f"Mejor accuracy (cv): {best_acc:.4f}\n")

    # Gráfico de convergencia
    plt.figure(figsize=(7, 4))
    plt.plot(history, marker="o")
    plt.xlabel("Iteración")
    plt.ylabel("Mejor fitness global")
    plt.title(f"Convergencia PSO - {model_name} en {dataset_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/convergencia_{model_name}_{dataset_name}.png")
    plt.close()

    return best_fit, best_acc, history
