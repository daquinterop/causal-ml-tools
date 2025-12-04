from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, GroupKFold
from sklearn.base import BaseEstimator
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Integer, Real, Categorical

class GPOptimizer(dict):
    """
    A class for Bayesian Optimization of model hyperparameters using scikit-optimize.
    """
    def __init__(self, model:BaseEstimator, model_kwargs:dict):        
        """
        Initializes the GPOptimizer with a model and its keyword arguments.

        Parameters
        ----------
        model : BaseEstimator
            The scikit-learn compatible model to be optimized.
        model_kwargs : dict
            A dictionary of keyword arguments for the model.
            Hyperparameters to be optimized should be defined using skopt's `Integer`, `Real`, or `Categorical` dimensions.
            Other fixed parameters should be passed as regular values.
        """
        self.model = model
        self._optimize_kwargs = {}
        self._model_kwargs = {}
        super().__init__({'model': self.model, 'model_kwargs': self._model_kwargs, 'optimize_kwargs': self._optimize_kwargs})
        self._dimensions = []
        self.gpopt_res = None
        for key, value in model_kwargs.items():
            if isinstance(value, (Integer, Real, Categorical)):
                self._optimize_kwargs[key] = value
                self._dimensions.append(value)
            else:
                self._model_kwargs[key] = value

    def _cv_scores(self, model, X, y, cv, groups, **kwargs):
        scores = cross_val_score(
            model,
            cv=cv,
            X=X, y=y,
            groups=groups,
            **kwargs
        )
        return scores

    def cv_optimize(self, X, y, cv=LeaveOneGroupOut(), groups=None, 
                    cv_kwargs=dict(n_jobs=-1, scoring='r2'),
                    gp_kwargs=dict(n_calls=20, n_initial_points=5)):
        """
        Performs Bayesian Optimization with cross-validation to find the best hyperparameters for the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.
        cv : cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            By default, LeaveOneGroupOut().
        groups : array-like of shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used when `cv` is a GroupKFold or LeaveOneGroupOut instance.
            By default, None.
        cv_kwargs : dict, optional
            Additional keyword arguments to be passed to `sklearn.model_selection.cross_val_score`.
            By default, dict(n_jobs=-1, scoring='r2').
        gp_kwargs : dict, optional
            Additional keyword arguments to be passed to `skopt.gp_minimize`.
            By default, dict(n_calls=20, n_initial_points=5).

        Returns
        -------
        OptimizeResult
            The best model found by Bayesian Optimization.
        """

        if isinstance(cv, (LeaveOneGroupOut, GroupKFold)):
            assert groups is not None, "groups cannot be None when using LeaveOneGroupOut or GroupKFold"
            assert len(groups) == len(y), "groups must have the same length as y"

        @use_named_args(dimensions=self._dimensions)
        def _loss(**kwargs):
            model_kwargs = {**self._model_kwargs, **kwargs}
            model = self.model(**model_kwargs)
            scores = self._cv_scores(model, cv=cv, X=X, y=y, groups=groups, **cv_kwargs)
            return -scores.mean()

        self.gpopt_res = gp_minimize(
            _loss,
            dimensions=self._dimensions, 
            **gp_kwargs
        )
        best_pars = dict(zip(self._optimize_kwargs.keys(), self.gpopt_res.x))
        best_model = self.model(**{**self._model_kwargs, **best_pars})
        cv_scores = self._cv_scores(best_model, cv=cv, X=X, y=y, groups=groups, **cv_kwargs)
        return best_model, cv_scores
    
