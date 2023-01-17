from _plot import plotPermutationImportance, plotMeanAccuracyDecrease
import pandas as pd
from sklearn.inspection import permutation_importance
import numpy as np

def permutationImportance(model, X_test, y_test, n_repeats, random_state, scoring):
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1, scoring=scoring)
    sorted_importances_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=np.array(X_test.columns.values)[sorted_importances_idx],
    )
    plotPermutationImportance(importances, str(n_repeats), str(model))

def meanAccuracyDecrease(model, X_test, y_test, n_repeats, random_state, scoring):
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1, scoring=scoring)
    feature_names = np.array(X_test.columns.values)
    forest_importances = pd.Series(result.importances_mean, index=feature_names.tolist())
    plotMeanAccuracyDecrease(forest_importances, result, str(n_repeats), str(model))

# TODO: take in whole df for test