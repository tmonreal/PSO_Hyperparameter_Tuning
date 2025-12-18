import matplotlib.pyplot as plt
from pso import PSOOptimizer
from sklearn.model_selection import cross_val_score
import time
import pandas as pd
from baselines import random_search, grid_search, bayes_opt, SKOPT_AVAILABLE

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


def benchmark(model_wrapper_cls,
              X,
              y,
              model_name,
              dataset_name,
              pso_config=None,
              baselines_to_run=None,
              baseline_budget=30,
              random_state=0):
    if pso_config is None:
        pso_config = dict(
            num_particles=20,
            max_iter=30,
            w=0.7,
            c1=1.5,
            c2=1.5,
            random_state=random_state,
        )

    if baselines_to_run is None:
        baselines_to_run = ["pso", "random", "bayes"] if SKOPT_AVAILABLE else ["pso", "random"]

    results = []

    # PSO
    if "pso" in baselines_to_run:
        t0 = time.time()
        best_fit, best_acc, history = run_experiment(
            model_wrapper_cls,
            X,
            y,
            model_name,
            dataset_name,
            alpha_complexity=0.0,
            pso_config=pso_config,
            random_state=random_state,
        )
        runtime = time.time() - t0
        results.append({
            "method": "pso",
            "best_fitness": best_fit,
            "best_accuracy": best_acc,
            "history": history,
            "runtime": runtime,
        })

    bounds = model_wrapper_cls(X=X, y=y).param_bounds

    # Random search
    if "random" in baselines_to_run:
        t0 = time.time()
        best_pos, best_fit, history = random_search(model_wrapper_cls, X, y, bounds, n_iter=baseline_budget, random_state=random_state)
        runtime = time.time() - t0
        # recompute accuracy for reporting
        params = model_wrapper_cls(X=X, y=y).decode(best_pos)
        model = model_wrapper_cls(X=X, y=y).build_model(params)
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=1)
        best_acc = scores.mean()
        results.append({
            "method": "random",
            "best_fitness": best_fit,
            "best_accuracy": best_acc,
            "history": history,
            "runtime": runtime,
        })

    # Grid search (chico)
    if "grid" in baselines_to_run:
        t0 = time.time()
        best_pos, best_fit, history = grid_search(model_wrapper_cls, X, y, bounds, n_per_dim=4)
        runtime = time.time() - t0
        params = model_wrapper_cls(X=X, y=y).decode(best_pos)
        model = model_wrapper_cls(X=X, y=y).build_model(params)
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=1)
        best_acc = scores.mean()
        results.append({
            "method": "grid",
            "best_fitness": best_fit,
            "best_accuracy": best_acc,
            "history": history,
            "runtime": runtime,
        })

    # Optimizacion Bayesiana
    if "bayes" in baselines_to_run:
        if not SKOPT_AVAILABLE:
            raise RuntimeError("scikit-optimize not available; install it to run 'bayes' baseline")
        t0 = time.time()
        best_pos, best_fit, history = bayes_opt(None, model_wrapper_cls, X, y, bounds, n_calls=baseline_budget, random_state=random_state)
        runtime = time.time() - t0
        params = model_wrapper_cls(X=X, y=y).decode(best_pos)
        model = model_wrapper_cls(X=X, y=y).build_model(params)
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=1)
        best_acc = scores.mean()
        results.append({
            "method": "bayes",
            "best_fitness": best_fit,
            "best_accuracy": best_acc,
            "history": history,
            "runtime": runtime,
        })

    df = pd.DataFrame([{
        "dataset": dataset_name,
        "model": model_name,
        "method": r["method"],
        "best_accuracy": r["best_accuracy"],
        "best_fitness": r["best_fitness"],
        "runtime": r["runtime"],
    } for r in results])

    return df, results
