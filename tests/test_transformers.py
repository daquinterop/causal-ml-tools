import pytest
import numpy as np
from CAgsalML.transformers import SpatialTransform
import skgstat as skg

# Mock geometry object for testing
class MockGeometry:
    def __init__(self, x, y):
        self.x = x
        self.y = y

@pytest.fixture
def sample_data():
    np.random.seed(42)
    x = np.random.rand(100) * 100
    y = np.random.rand(100) * 100
    values = np.random.rand(100) * 10 + 5 * (x / 100) # Introduce some spatial trend
    geometry = MockGeometry(x, y)
    return geometry, values

@pytest.fixture
def spatial_transformer(sample_data):
    geometry, _ = sample_data
    return SpatialTransform(geometry=geometry, maxlag=50, n_lags=20)

def test_spatial_transform_initialization(sample_data):
    geometry, _ = sample_data
    transformer = SpatialTransform(
        geometry=geometry, maxlag=50, estimator='matheron', model='spherical', 
        n_lags=30, fit_method='lm', bin_func='uniform', normalize=True
    )

    assert np.array_equal(transformer.coordinates, np.array([geometry.x, geometry.y]).T)
    assert transformer.estimator == 'matheron'
    assert transformer.model == 'spherical'
    assert transformer.maxlag == 50
    assert transformer.n_lags == 30
    assert transformer.fit_method == 'lm'
    assert transformer.bin_func == 'uniform'
    assert transformer.kwargs['normalize'] is True
    assert transformer.variogram is None
    assert transformer.kriging is None
    assert transformer._y is None

def test_spatial_transform_fit(spatial_transformer, sample_data):
    _, values = sample_data
    spatial_transformer.fit(values)

    assert spatial_transformer._y is not None
    assert np.array_equal(spatial_transformer._y, values)
    assert isinstance(spatial_transformer.variogram, skg.Variogram)