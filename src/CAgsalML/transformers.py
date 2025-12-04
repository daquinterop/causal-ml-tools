import skgstat as skg
import numpy as np

class SpatialTransform:
    def __init__(self, geometry, maxlag, estimator='cressie', model='matern',
                 n_lags=50, fit_method='trf', bin_func='even', **kwargs):
        """
        Initializes the SpatialTransform object for spatial data processing.

        Parameters
        ----------
            geometry : object
                An object with 'x' and 'y' attributes representing the coordinates.
            maxlag : float
                The maximum distance up to which the variogram is calculated.
            estimator : str, optional
                The variogram estimator to use (e.g., 'cressie', 'matheron'). Defaults to 'cressie'.
            model : str, optional
                The variogram model to fit (e.g., 'matern', 'spherical', 'exponential'). Defaults to 'matern'.
            n_lags : int, optional
                The number of bins for the variogram. Defaults to 50.
            fit_method : str, optional
                The method used to fit the variogram model (e.g., 'trf', 'lm'). Defaults to 'trf'.
            bin_func : str, optional
                The function used to define the variogram bins (e.g., 'even', 'uniform'). Defaults to 'even'.
            **kwargs
                Additional keyword arguments to be passed to the skgstat.Variogram constructor.
        """
        self.coordinates = np.array([geometry.x, geometry.y]).T
        self.estimator = estimator
        self.bin_func = bin_func
        self.model = model
        self.maxlag = maxlag
        self.n_lags = n_lags
        self.fit_method = fit_method
        self.kwargs = kwargs
        self.variogram = None
        self.kriging = None
        self._y = None


    def fit(self, y):
        """
        Fits the variogram model to the provided data.

        Parameters
        ----------
        y : array-like
            The 1-dimensional array of values to which the variogram will be fitted.
        """
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional array.")
        if len(y) != len(self.coordinates):
            raise ValueError("The length of y must match the number of coordinates.")
        self._y = y.copy()
        self.variogram = skg.Variogram(
            coordinates=self.coordinates,
            values=self._y,
            estimator=self.estimator,
            model=self.model,
            bin_func=self.bin_func,
            n_lags=self.n_lags,
            fit_method=self.fit_method,
            maxlag=self.maxlag,
            **self.kwargs
        )

        self.kriging = skg.OrdinaryKriging(self.variogram)

    def transform_demean(self):
        """
        Transforms the input data by subtracting the spatially-predicted mean.

        This method uses the fitted Kriging model to predict spatial values at the
        original coordinates and then subtracts these predicted values from the
        original `y` data, effectively performing a spatial de-meaning.

        Returns
        -------
        numpy.ndarray
            A 1-dimensional array of spatially de-meaned values.

        Raises
        ------
        RuntimeError
            If the variogram and Kriging model have not been fitted (i.e., `fit()` has not been called).
        """
        if self.kriging is None or self._y is None:
            raise RuntimeError("Variogram and Kriging model not fitted. Call fit() first.")
        return self._y - self.predict()
            
    def predict(self, coords=None):
        """
        Predicts spatial values using the fitted Kriging model.

        Parameters
        ----------
        coords : array-like, optional
            The coordinates at which to predict spatial values.
            If None, predictions are made at the original `self.coordinates`.

        Returns
        -------
        array-like
            The predicted spatial values.

        Raises
        ------
        RuntimeError
            If the Kriging model has not been fitted (i.e., `fit()` has not been called).
        """
        if self.variogram is None:
            raise RuntimeError("Variogram not fitted. Call fit() first.")
        if coords is None:        
            return self.kriging.transform(self.coordinates)
        return self.kriging.transform(coords)