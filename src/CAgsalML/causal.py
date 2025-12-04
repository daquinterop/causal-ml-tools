from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from balance import Sample
from pandas import DataFrame
import seaborn as sns
from matplotlib import pyplot as plt
from econml.validate.sensitivity_analysis import dml_sensitivity_values, sensitivity_summary


class LogisticRegressionPSModel(LogisticRegression):
    def __init__(self,
                 data: DataFrame,
                 treatment: str,
                 outcome: str, 
                 covariates: list[str],
                 random_state: int = 42,
                 weigths: list[float] = None,
                 **kwargs):
        """
        Fits a Propensity Score model using the Covariate Balance Propensity Score approach (CBPS).
        CBPS weights are used in the logistic regression model.This model extends `LogisticRegression` to incorporate propensity scores for causal inference.

        Parameters
        ----------
        data : DataFrame
            The input DataFrame containing treatment, outcome, and covariate columns.
        treatment : str
            The name of the column indicating the treatment assignment (binary: 0 or 1).
        outcome : str
            The name of the column indicating the outcome variable.
        covariates : list[str]
            A list of column names representing the covariates to be used in the model.
        random_state : int, optional
            Seed for the random number generator, by default 42.
        **kwargs
            Additional keyword arguments to be passed to the `LogisticRegression` constructor.
        """
        super().__init__(random_state=random_state, **kwargs)
        self.data = data.copy()
        self.data.loc[:, covariates] = self.data[covariates].apply(
            lambda row: (row - row.mean())/row.std(), axis=1
        )
        self.treatment = treatment
        self.outcome = outcome
        self.covariates = covariates
        self.sample = self.target_sample = self.adjusted_sample = \
            self.covariate_smd = self.weigths = None
        if weigths is not None:
            self.data['weight'] = weigths
            self.weigths = self.data['weight']
        else:
            self._calculate_weigths()
            self.weigths = self.data['weight']
        self._calculate_ps()        

    def _calculate_ps(self):
        self.fit(
            self.data[self.covariates], self.data[self.treatment], 
            sample_weight=self.data['weight']
        )
        self.data['propensity_score'] = self.predict_proba(self.data[self.covariates])[:, 1]
        self.propensity_score = self.data['propensity_score']

    def evaluate_ps_model(self):
        y_true = self.data[self.treatment]
        y_pred = self.predict(self.data[self.covariates])
        print(classification_report(y_true, y_pred))

    def _calculate_weigths(self):
        self.sample = Sample.from_frame(
            self.data.loc[~self.data[self.treatment].astype(bool), self.covariates].reset_index(), id_column='index', 
            use_deepcopy=True
        )
        self.target_sample = Sample.from_frame(
            self.data.loc[self.data[self.treatment].astype(bool), self.covariates].reset_index(), id_column='index', 
            use_deepcopy=True
        )
        self.sample = self.sample.set_target(self.target_sample)
        self.adjusted_sample = self.sample.adjust(method='cbps')
        self.data['weight'] = self.adjusted_sample._df.set_index('index').weight
        self.data['weight'] = self.data.weight.fillna(1)

        self.covariate_smd = self.sample.covars().mean().T.diff(axis=1)[['target']].rename(columns={'target': 'pre'})
        self.covariate_smd['post'] = self.adjusted_sample.covars().mean().T.diff(axis=1)['target']

    def plot_common_support(self, ps_trim:tuple[float, float]=(0.05, 0.95), **kwargs):
        """
        Plots the distribution of propensity scores for treated and control groups
        to visualize common support.

        Parameters
        ----------
        ps_trim : tuple(float, float), optional
            A tuple (lower_bound, upper_bound) to trim propensity scores for plotting.
            Scores outside this range will be excluded from the plot.
            By default (0.05, 0.95).
        **kwargs
            Additional keyword arguments to be passed to `seaborn.histplot`.
        """
        ax = sns.histplot(
            self.data.loc[self.data.propensity_score.map(lambda x: ps_trim[0] < x < ps_trim[1])],
            x='propensity_score', element='bars', stat='count', kde=True,
            hue=self.data[self.treatment].astype(bool).rename('Treated'), **kwargs
        )
        ax.set_xlabel("Propensity Score")

def dml_sensitivity_analysis(estimator, conditional_mask, null_hypothesis=0, 
                        alpha=0.05, c_y=0.05, c_t=0.05, rho=1., decimals=3): 
    '''
    Wrapper to some of the methods defined by Chernozhukov (2022) and defined in econml
    for the ATE. Here we add the posibility to do the sensitivity analysis for a specific
    CATE on the subpopulation defined in the conditional mask.

    Parameters
    -------------
    estimator: DML estimator
    conditional_mask: Array (N, ) 
        Array to mask the population of interest
    null_hypothesis: float=0
    alpha: float=0.05
        Expected statistical significance
    c_y: float=0.05
        Proportion of the outcome residual variance that is explained by the unobserved
        confounders
    c_t: float=0.05
        Proportion of the treatment residual variance explained by the unobserved
        confounders
    rho: float (0-1): 1.0
        The correlation between the regression error (impact of the confounder). 1 is the 
        worst-case scenario where the missing confounder is completely correlated to the outcome.

    ----------
    References:
        Chernozhukov, C. Cinelli, N. Kallus, W. Newey, A. Sharma, and V. Syrgkanis. 
        Long Story Short: Omitted Variable Bias in Causal Machine Learning. NBER Working Paper No. 
        30302, 2022. URL https://www.nber.org/papers/w30302.
    '''
    sensitivity_params = dml_sensitivity_values(
        t_res=estimator._cached_values.nuisances[1][conditional_mask], 
        y_res=estimator._cached_values.nuisances[0][conditional_mask]
    )
    return sensitivity_summary(
            **sensitivity_params._asdict(), 
            null_hypothesis=null_hypothesis, 
            alpha=alpha,
            c_y=c_y, 
            c_t=c_y, 
            rho=rho, 
            decimals=decimals
        )

        

        
