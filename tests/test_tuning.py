import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from skopt.space import Real, Integer, Categorical
from CAgsalML.tuning import GPOptimizer

# Fixture for a simple model and kwargs
@pytest.fixture
def simple_model_kwargs():
    return {
        'C': Real(1e-6, 1e+6, prior='log-uniform', name='C'),
        'solver': Categorical(['liblinear', 'lbfgs'], name='solver'),
        'max_iter': Integer(100, 1000, name='max_iter')
    }

@pytest.fixture
def simple_data():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    groups = np.repeat(np.arange(10), 10)
    return X, y, groups

def test_gpoptimizer_initialization(simple_model_kwargs):
    model = LogisticRegression
    optimizer = GPOptimizer(model, simple_model_kwargs)
    assert optimizer.model == model
    assert len(optimizer._dimensions) == 3
    assert 'C' in optimizer._optimize_kwargs
    assert 'solver' in optimizer._optimize_kwargs
    assert 'max_iter' in optimizer._optimize_kwargs
    assert not optimizer._model_kwargs # No fixed parameters in this case

def test_gpoptimizer_initialization_with_fixed_params():
    model = LogisticRegression
    model_kwargs = {
        'C': Real(1e-6, 1e+6, prior='log-uniform', name='C'),
        'solver': Categorical(['liblinear', 'lbfgs'], name='solver'),
        'random_state': 42 # Fixed parameter
    }
    optimizer = GPOptimizer(model, model_kwargs)

    assert len(optimizer._dimensions) == 2
    assert 'C' in optimizer._optimize_kwargs
    assert 'solver' in optimizer._optimize_kwargs
    assert 'random_state' in optimizer._model_kwargs
    
def test_gpoptimizer_cv_optimize(simple_model_kwargs, simple_data):
    X, y, groups = simple_data
    model = LogisticRegression
    optimizer = GPOptimizer(model, simple_model_kwargs)

    # Test with LeaveOneGroupOut
    result, _  = optimizer.cv_optimize(X, y, cv=LeaveOneGroupOut(), groups=groups, gp_kwargs={'n_calls': 5, 'n_initial_points': 2})
    assert result is not None
    assert isinstance(result, LogisticRegression)
    assert len(optimizer.gpopt_res.x) == len(optimizer._dimensions)

    # Test with GroupKFold
    result, _ = optimizer.cv_optimize(X, y, cv=GroupKFold(n_splits=2), groups=groups, gp_kwargs={'n_calls': 5, 'n_initial_points': 2})
    assert result is not None
    assert isinstance(result, LogisticRegression)

    # Test without groups for a non-group-based CV (e.g., KFold, though not explicitly tested here,
    # the assertion for groups=None should not trigger if cv is not GroupKFold/LeaveOneGroupOut)
    # For simplicity, we'll just ensure it runs without error for a standard CV
    # (though our fixture uses groups, the internal logic should handle it)
    from sklearn.model_selection import KFold
    optimizer_no_groups = GPOptimizer(model, {'C': Real(1e-6, 1e+6, prior='log-uniform', name='C')})
    result_no_groups = optimizer_no_groups.cv_optimize(X, y, cv=KFold(n_splits=2), gp_kwargs={'n_calls': 3, 'n_initial_points': 1})
    assert result_no_groups is not None

    # Test assertion for missing groups with LeaveOneGroupOut
    with pytest.raises(AssertionError, match="groups cannot be None when using LeaveOneGroupOut or GroupKFold"):
        optimizer.cv_optimize(X, y, cv=LeaveOneGroupOut(), gp_kwargs={'n_calls': 1, 'n_initial_points': 1})