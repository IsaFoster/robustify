from _plot import plotPermutationImportance, plotMeanAccuracyDecrease
import pandas as pd
from sklearn.inspection import permutation_importance
import numpy as np


def construct_permutation_df(importances, sorted_importances_idx, X_test):
    importances = pd.DataFrame(
        importances.importances[sorted_importances_idx].T,
        columns=np.array(X_test.columns.values)[sorted_importances_idx],
    )
    return importances

def permutationImportance(baseline_model, noisy_model, X_test, y_test, n_repeats, random_state, scoring):
    result_baseline = permutation_importance(baseline_model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1, scoring=scoring)
    result_noise = permutation_importance(noisy_model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1, scoring=scoring)
    sorted_importances_idx = result_baseline.importances_mean.argsort()
    baseline_importances = construct_permutation_df(result_baseline, sorted_importances_idx, X_test)
    noisy_importances = construct_permutation_df(result_noise, sorted_importances_idx, X_test)
    fig = plotPermutationImportance(baseline_importances, noisy_importances, str(n_repeats), str(baseline_model))
    fig.show()
    return None # df with importances

def meanAccuracyDecrease(baseline_model, noisy_model, X_test, y_test, n_repeats, random_state, scoring):
    result = permutation_importance(baseline_model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1, scoring=scoring)
    result_noise = permutation_importance(noisy_model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1, scoring=scoring)
    feature_names = np.array(X_test.columns.values)
    baseline_importances = pd.Series(result.importances_mean, index=feature_names.tolist())
    noisy_importances = pd.Series(result_noise.importances_mean, index=feature_names.tolist())
    fig = plotMeanAccuracyDecrease(baseline_importances, noisy_importances, result, str(n_repeats), str(baseline_model))
    return fig

# TODO: take in whole df for test