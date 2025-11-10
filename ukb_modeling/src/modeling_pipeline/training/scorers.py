## TODO: make these compatible with sksurv!

from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv
from lifelines.statistics import logrank_test
import numpy as np
from sklearn.metrics import make_scorer


def ibs_scorer(estimator, X, y):
    surv = Surv.from_arrays(event=y[:, 0], time=y[:, 1])
    time_grid = np.linspace(0, np.max(y[:, 0]), 100)
    surv_func = estimator.predict_survival_function(X)
    return -integrated_brier_score(surv, surv_func, time_grid)


def logrank_test_scorer(estimator, X, y, n_groups=2):
    """
    Custom scorer to evaluate survival models using the log-rank test.

    Parameters:
    - estimator: Fitted survival model (compatible with `sksurv`).
    - X: Features (n_samples, n_features).
    - y: Structured array with survival times and event indicators.
         Example: Surv.from_arrays(event, time).
    - n_groups: Number of risk groups to split predictions into.

    Returns:
    - logrank_stat: The log-rank test statistic (higher is better for separation).
    """

    # Ensure survival data is in the correct format
    if not isinstance(y, np.ndarray) or y.dtype.names is None:
        raise ValueError("y must be a structured array with 'event' and 'time' fields.")

    # Predict risk scores
    risk_scores = estimator.predict(X)

    # Split predictions into risk groups
    quantiles = np.linspace(0, 1, n_groups + 1)
    thresholds = np.quantile(risk_scores, quantiles)
    risk_groups = np.digitize(risk_scores, thresholds, right=True)

    # Perform the log-rank test
    logrank_stat = 0
    for i in range(1, n_groups):
        group_a_mask = risk_groups == i
        group_b_mask = risk_groups == i + 1

        if np.any(group_a_mask) and np.any(group_b_mask):
            logrank_result = logrank_test(
                y["time"][group_a_mask],
                y["time"][group_b_mask],
                event_observed_A=y["event"][group_a_mask],
                event_observed_B=y["event"][group_b_mask],
            )
            logrank_stat += logrank_result.test_statistic

    return logrank_stat
