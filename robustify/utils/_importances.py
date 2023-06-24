import numpy as np
from sklearn.inspection import permutation_importance
from eli5.sklearn import PermutationImportance
from eli5.permutation_importance import get_score_importances
from lime import lime_tabular
import shap
import pandas as pd
from ._filter import is_keras_model, is_tree_model
from ._transform import convert_to_numpy
import warnings

def fxn():
    warnings.warn("userWarn", UserWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def filter_on_importance_method(model, index, X, y, X_test, random_state, scoring, measure, custom_predict):
    if measure: measure = measure.lower()
    switcher = {
        None: lambda: check_for_deafult_properties(model, index, X, y, X_test, random_state, scoring),
        'eli5': lambda: calculate_eli5_importances(model, index, X, y, X_test, random_state, scoring),
        'shap': lambda: calculate_shap_importances(model, index, X, X_test, random_state, custom_predict),
        'lime': lambda: calculate_lime_importances(model, index, X, X_test, custom_predict)
    }
    return switcher.get(measure, lambda:
                        print("Invalid importance measure for {}".format(str(model))))()

def check_for_deafult_properties(model, index, X, y, X_test, random_state, scoring):
    if hasattr(model, 'feature_importances_'):
        measured_property = 'feature importance'
        if index is not None:
            return model.feature_importances_[index], measured_property
        else:
            return model.feature_importances_, measured_property 
    elif hasattr(model, 'coef_'):
        measured_property = 'coefficients'
        if isinstance(model.coef_[0], (np.ndarray, list)):
            if index is not None:
                return model.coef_[0][index], measured_property
            else:
                return model.coef_[0], measured_property
        if index is not None:
            return model.coef_[index], measured_property
        else:
            return model.coef_, measured_property
    else:
        if is_keras_model:
            return calculate_eli5_importances(model, index, X, y, X_test, random_state, scoring)
        return calculate_permuation_importances(model, index, X, y, X_test, random_state, scoring)

def calculate_permuation_importances(model, index, X, y, X_test, random_state, scoring):
    measured_property = 'permutation importance'
    importances = permutation_importance(model, convert_to_numpy(X), convert_to_numpy(y),
                                            n_repeats=1, random_state=random_state,
                                            n_jobs=-1, scoring=scoring)
    if index is not None:
        return importances.importances_mean[index], measured_property
    else:
        return importances.importances_mean, measured_property

def calculate_eli5_importances(model, index, X, y, X_test, random_state, scoring):
    importances = PermutationImportance(model, scoring=scoring, random_state=random_state,
                                        n_iter=1, cv="prefit", refit=False).fit(X, y)
    measured_property = "eli5 permutation importance"
    if index is not None:
        return importances.feature_importances_[index], measured_property
    else:
        return importances.feature_importances_, measured_property

def calculate_lime_importances(model, index, X, X_test, custom_predict):
    measured_property = "lime explainer"
    explainer_classification = lime_tabular.LimeTabularExplainer(
        training_data=X.to_numpy(),
        feature_names=X.columns.tolist(),
        class_names=['data_type'],
        mode='classification',
        verbose=False)
    explainer_regression = lime_tabular.LimeTabularExplainer(
        training_data=X.to_numpy(),
        feature_names=X.columns.tolist(),
        class_names=['data_type'],
        mode='regression',
        verbose=False)
    values = []
    values_df = pd.DataFrame(columns=list(range(len(X.columns))))
    for i in range(int((X.shape[0]/10))):
        if custom_predict:
            predict_fn_rf = lambda x: custom_predict(model, x).astype(float)
        elif is_keras_model(model):
            predict_fn_rf = lambda x: model.predict(x, verbose=0).astype(float)
        else:
            predict_fn_rf = lambda x: model.predict_proba(x).astype(float)
        try:
            exp = explainer_classification.explain_instance(X.iloc[i], predict_fn_rf, num_features=len(X.columns.tolist()))
        except: 
            exp = explainer_regression.explain_instance(X.iloc[i], predict_fn_rf, num_features=len(X.columns.tolist()))
        dic = dict(list(exp.as_map().values())[0])
        if index is not None:
            values.append(dic.get(index))
        else:
            limes = pd.DataFrame(columns=list(dic.keys()))
            limes.loc[len(limes.index)] = list(dic.values())
            values_df = pd.concat([values_df, limes])
    if index is not None:
        average_value = np.mean(values, axis=0)
        return average_value, measured_property
    else:
        average_value = values_df.mean(axis=0)
        return average_value, measured_property

def calculate_shap_importances(model, index, X, X_test, random_state, custom_predict):
    measured_property = "shap values"
    if custom_predict:
        model_pred = lambda x: custom_predict(model, x)
        explainer = shap.Explainer(model_pred, X, seed=random_state, silent=True)
        shap_values = explainer(X)
        average_values = [sum(sub_list) / len(sub_list) for sub_list in zip(*shap_values.values)]
    elif is_keras_model(model):
        average_values = shap_values_keras(model, X, X_test, random_state)
    elif is_tree_model(model):
        average_values =shap_values_tree(model, X, X_test, random_state)
    else:
        if hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model.predict, X, seed=random_state, silent=True)
            shap_values = explainer(X)
        else:
            f = lambda x: model.predict(x)
            med = X.median().values.reshape((1,X.shape[1]))
            explainer = shap.Explainer(f, med, seed=random_state, silent=True)
            shap_values = explainer(X) 
        average_values = [sum(sub_list) / len(sub_list) for sub_list in zip(*shap_values.values)]
    if index is not None:
        return average_values[index], measured_property   
    else:
        return average_values, measured_property    
    raise Exception("Could not compute shap importances")
    
def shap_values_keras(model, X, X_test, random_state):
    ind = X.shape[0]/10
    X_train_summary = shap.kmeans(X, 20)
    def f(X):
        return model.predict(X, verbose=0)
    #explainer = shap.DeepExplainer(model, X)
    #shap_values = explainer.shap_values(X)

    explainer = shap.KernelExplainer(f, X, seed=random_state, silent=True)
    shap_values = explainer.shap_values(X_test, nsamples=100)
    average_values = np.sum(np.abs(shap_values).mean(1), axis=0)
    return average_values

def shap_values_tree(model, X, X_test, random_state):
    explainer = shap.TreeExplainer(model, seed=random_state, silent=True)
    shap_values = explainer.shap_values(X)
    average_values = np.mean(shap_values, axis=0)
    average_values = np.mean(average_values, axis=0)
    return average_values
