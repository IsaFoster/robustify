from sklearn import metrics
from inspect import signature
from Robustness.utils._transform import convert_to_numpy
from Robustness.utils._predict import get_prediction

def get_scorer(metric, model, X_test, y_test, custom_predict):
    if isinstance(metric, str):
        return get_scorer_sckit_learn(metric, model, X_test, y_test, custom_predict)
    elif hasattr(metric, '__call__'):
        return get_custom_scorer(metric, model, X_test, y_test, custom_predict)
    else:
        raise ValueError("No scorer object found. Type {} is not callable".format(type(metric)))
    
def get_scorer_sckit_learn(metric, model, X_test, y_test, custom_predict):
    try: 
        scorer = metrics.get_scorer(metric)
        return scorer._score_func(y_test, get_prediction(model, X_test, custom_predict)) 
    except Exception:
        raise
    
def get_custom_scorer(metric, model, X_test, y_test, custom_predict):
    try:
        X_test = convert_to_numpy(X_test)
        y_test = convert_to_numpy(y_test)
        y_pred = get_prediction(model, X_test, custom_predict)
        sig = signature(metric)
        if len(sig.parameters) ==3:
            score = metric(model, y_pred, y_test)
        else:
            score = metric(y_pred, y_test)
        validate_score(score)
        return score 
    except Exception:
        raise

def validate_score(score):
    if score == None or isinstance(score, (str, bool)):
        raise TypeError("Score must be a quantitative value, not {}".format(type(score)))  
    if len(score) != 1 or isinstance(score, (list, dict, tuple)):
        raise ValueError("Score must be a single value, not {}".format(type(score)))

    