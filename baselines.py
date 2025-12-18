from typing import Callable, Dict, List, Tuple
import time
import numpy as np

try:
    from skopt import Optimizer as SkoptOptimizer
    from skopt.space import Real, Integer
    SKOPT_AVAILABLE = True
except Exception:
    SKOPT_AVAILABLE = False

from sklearn.model_selection import cross_val_score


def random_search(wrapper_cls, X, y, bounds, n_iter=30, random_state=None, scoring="accuracy"):
    rng = np.random.default_rng(random_state)
    dim = len(bounds)
    best = None
    history = []
    for it in range(n_iter):
        pos = np.array([rng.uniform(low, high) for (low, high) in bounds])
        fitness = _eval_pos(wrapper_cls, X, y, pos, scoring)
        history.append(fitness)
        if best is None or fitness < best[0]:
            best = (fitness, pos.copy())
    return best[1], best[0], history


def grid_search(wrapper_cls, X, y, bounds, n_per_dim=5, scoring="accuracy"):
    # Small grid: create linspace per dimension
    grids = [np.linspace(low, high, n_per_dim) for (low, high) in bounds]
    mesh = np.array(np.meshgrid(*grids)).T.reshape(-1, len(bounds))
    best = None
    history = []
    for pos in mesh:
        fitness = _eval_pos(wrapper_cls, X, y, pos, scoring)
        history.append(fitness)
        if best is None or fitness < best[0]:
            best = (fitness, pos.copy())
    return best[1], best[0], history


def bayes_opt(skopt_space, wrapper_cls, X, y, bounds, n_calls=25, random_state=None, scoring="accuracy"):
    if not SKOPT_AVAILABLE:
        raise RuntimeError("scikit-optimize (skopt) is required for bayes_opt; install it in requirements.txt")

    # Build skopt search space from bounds (simple mapping: Integer if both bounds are integer-like)
    space = []
    for (low, high) in bounds:
        if float(low).is_integer() and float(high).is_integer():
            space.append(Integer(int(low), int(high)))
        else:
            space.append(Real(float(low), float(high)))

    opt = SkoptOptimizer(space, random_state=random_state)
    best = None
    history = []

    for it in range(n_calls):
        x = opt.ask()
        fitness = _eval_pos(wrapper_cls, X, y, np.array(x), scoring)
        opt.tell(x, fitness)
        history.append(fitness)
        if best is None or fitness < best[0]:
            best = (fitness, np.array(x).copy())

    return best[1], best[0], history


def _eval_pos(wrapper_cls, X, y, pos, scoring="accuracy"):
    wrapper = wrapper_cls(X=X, y=y, alpha_complexity=0.0, cv=5)
    params = wrapper.decode(pos)
    model = wrapper.build_model(params)
    scores = cross_val_score(model, X, y, cv=5, scoring=scoring, n_jobs=1)
    acc = scores.mean()
    # Convert to same minimization fitness as PSO uses: 1 - acc (no complexity penalty here)
    return 1.0 - acc
