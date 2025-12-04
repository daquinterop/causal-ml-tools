import pytest
import pandas as pd
import numpy as np
from CAgsalML.causal import LogisticRegressionPSModel

@pytest.fixture
def ps_model():
    np.random.seed(42)
    data = {
        'treatment': np.random.randint(0, 2, 100),
        'outcome': np.random.randint(0, 2, 100),
        'covariate1': np.random.rand(100) * 10,
        'covariate2': np.random.rand(100) * 5,
        'covariate3': np.random.rand(100) * 20
    }
    df = pd.DataFrame(data)
    # Introduce some correlation for more realistic testing
    df['outcome'] = df['outcome'] + (df['treatment'] * 0.5).astype(int)
    df['covariate1'] = df['covariate1'] + (df['treatment'] * 2)
    return df

def test_psmodel_initialization(ps_model):
    treatment_col = 'treatment'
    outcome_col = 'outcome'
    covariates_list = ['covariate1', 'covariate2', 'covariate3']

    ps_model = LogisticRegressionPSModel(
        data=ps_model,
        treatment=treatment_col,
        outcome=outcome_col,
        covariates=covariates_list
    )

    assert isinstance(ps_model, LogisticRegressionPSModel)
    assert ps_model.treatment == treatment_col
    assert ps_model.outcome == outcome_col
    assert ps_model.covariates == covariates_list
    assert 'weight' in ps_model.data.columns
    assert 'propensity_score' in ps_model.data.columns
    assert ps_model.propensity_score is not None
    assert ps_model.sample is not None
    assert ps_model.target_sample is not None
    assert ps_model.adjusted_sample is not None
    assert ps_model.covariate_smd is not None
    assert 'pre' in ps_model.covariate_smd.columns
    assert 'post' in ps_model.covariate_smd.columns