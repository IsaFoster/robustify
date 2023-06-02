import numpy as np
from sklearn.inspection import permutation_importance
from eli5.sklearn import PermutationImportance
from eli5.permutation_importance import get_score_importances
from lime import lime_tabular
import shap
from ._filter import is_keras_touch_model, is_tree_model

def filter_on_importance_method(model, index, X, y, random_state, scoring, measure, custom_predict):
    if measure: measure = measure.lower()
    switcher = {
        None: lambda: check_for_deafult_properties(model, index, X, y, random_state, scoring),
        'eli5': lambda: calculate_eli5_importances(model, index, X, y, random_state, scoring),
        'shap': lambda: calculate_shap_importances(model, index, X, random_state, custom_predict),
        'lime': lambda: calculate_lime_importances(model, index, X, custom_predict)
    }
    return switcher.get(measure, lambda:
                        print("Invalid importance measure for {}".format(str(model))))()

def check_for_deafult_properties(model, index, X, y, random_state, scoring):
    if hasattr(model, 'feature_importances_'):
        measured_property = 'feature importance'
        return model.feature_importances_[index], measured_property
    elif hasattr(model, 'coef_'):
        measured_property = 'coefficients'
        if isinstance(model.coef_[0], (np.ndarray, list)):
            return model.coef_[0][index], measured_property
        return model.coef_[index], measured_property
    else:
        return calculate_permuation_importances(model, index, X, y, random_state, scoring)

def calculate_permuation_importances(model, index, X, y, random_state, scoring):
    try:
        measured_property = 'permutation importance'
        importances = permutation_importance(model, X, y, n_repeats=1,
                                             random_state=random_state,
                                             n_jobs=-1, scoring=scoring)
        return importances.importances_mean[index], measured_property
    except:
        raise Exception("cound not calculate coefficients or feature importance")

def calculate_eli5_importances(model, index, X, y, random_state, scoring):
    if not isinstance(scoring, str):
        try:
            measured_property = "eli5_custom_score_function"
            _, score_decreases = get_score_importances(scoring, np.array(X),
                                                       y, n_iter=1,
                                                       random_state=random_state)
            feature_importances = np.mean(score_decreases, axis=0)
            return feature_importances[index], measured_property
        except:
            raise Exception("Could not compute eli5 importances")
    else:
        try:
            importances = PermutationImportance(model, scoring=scoring, random_state=random_state,
                                                n_iter=1, cv="prefit", refit=False).fit(X, y)
            measured_property = "eli5 permutation importance"
            return importances.feature_importances_[index], measured_property
        except:
            raise Exception("Could not compute eli5 importances") 

def calculate_lime_importances(model, index, X, custom_predict):
    try:
        measured_property = "lime explainer"
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X.to_numpy(),
            feature_names=X.columns.tolist(),
            class_names=['data_type'],
            mode='classification',
            verbose=False)
        values = []
        for i in range(int((X.shape[0]/10))):
            if custom_predict:
                predict_fn_rf = lambda x: custom_predict(model, x).astype(float)
            else:
                predict_fn_rf = lambda x: model.predict_proba(x).astype(float)
            exp = explainer.explain_instance(X.to_numpy()[i], predict_fn_rf, 
                                             num_features=len(X.columns.tolist()))
            dic = dict(list(exp.as_map().values())[0])
            values.append(dic.get(index))
        average_value = np.mean(values, axis=0)
        return average_value, measured_property
    except:
        raise Exception("Could not compute lime importances")


# model agnostic shap: explainer = shap.KernelExplainer(model.predict_proba, X_train, link="logit")
# shap_values = explainer.shap_values(X_test, nsamples=100)
def calculate_shap_importances(model, index, X, random_state, custom_predict):
    measured_property = "shap"
    if is_keras_touch_model(model):
        assert (False)
        #reg 
        # rather than use the whole training set to estimate expected values, we summarize with
        # a set of weighted kmeans, each weighted by the number of points they represent.
        X_train_summary = shap.kmeans(X, 10)
        explainer = shap.KernelExplainer(model.predict, X_train_summary)
        shap_values = explainer.shap_values(X)
        # clas
        explainer = shap.KernelExplainer(model.predict_proba, X)
        shap_values = explainer.shap_values(X)
    elif is_tree_model(model):
        if custom_predict:
            assert (False)
            model_pred = lambda x: custom_predict(model, x)
            explainer = shap.Explainer(model_pred, X)
        elif hasattr(model, "predict_proba"):
            explainer = shap.KernelExplainer(model.predict_proba, X, seed=random_state)
            shap_values = explainer.shap_values(X)
            average_values = np.sum(np.abs(shap_values).mean(1), axis=0)
        else:
            explainer = shap.TreeExplainer(model, seed=random_state)
            shap_values = explainer.shap_values(X)
            average_values = np.abs(shap_values).mean(0)
            
        return average_values[index], measured_property            
    else:
        if custom_predict:
            assert (False)
            model_pred = lambda x: custom_predict(model, x)
            explainer = shap.Explainer(model_pred, X)
        elif hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model.predict, X, seed=random_state)
            shap_values = explainer(X)
        else:
            f = lambda x: model.predict(x)
            med = X.median().values.reshape((1,X.shape[1]))
            explainer = shap.Explainer(f, med, seed=random_state)
            shap_values = explainer(X) ###OBS NOT HTERE shap_values = explainer.shap_values(X_test)
        average_values = [sum(sub_list) / len(sub_list) for sub_list in zip(*shap_values.values)]
        return average_values[index], measured_property   
    raise Exception("Could not compute shap importances")
    return average_values[index], measured_property
    
