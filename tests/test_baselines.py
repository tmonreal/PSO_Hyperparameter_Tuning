import pytest
import numpy as np

from baselines import random_search, grid_search, SKOPT_AVAILABLE
from datasets import get_dataset_wine
from models import KNNWrapper


def test_random_search_smoke():
    X, y, name = get_dataset_wine()
    bounds = KNNWrapper(X=X, y=y).param_bounds
    best_pos, best_fit, history = random_search(KNNWrapper, X, y, bounds, n_iter=3, random_state=0)
    assert isinstance(best_pos, np.ndarray)
    assert isinstance(best_fit, float)
    assert len(history) == 3


def test_grid_search_smoke():
    X, y, name = get_dataset_wine()
    bounds = KNNWrapper(X=X, y=y).param_bounds
    best_pos, best_fit, history = grid_search(KNNWrapper, X, y, bounds, n_per_dim=2)
    assert isinstance(best_pos, np.ndarray)
    assert isinstance(best_fit, float)
    assert len(history) >= 1


def test_bayes_smoke_or_skip():
    if not SKOPT_AVAILABLE:
        pytest.skip("skopt not installed")
    from baselines import bayes_opt
    X, y, name = get_dataset_wine()
    bounds = KNNWrapper(X=X, y=y).param_bounds
    best_pos, best_fit, history = bayes_opt(None, KNNWrapper, X, y, bounds, n_calls=3, random_state=0)
    assert isinstance(best_pos, np.ndarray)
    assert isinstance(best_fit, float)
    assert len(history) == 3
