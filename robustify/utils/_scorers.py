from sklearn import metrics
from inspect import signature
from ._transform import convert_to_numpy
from ._predict import get_prediction

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
        y_pred = get_prediction(model, X_test, custom_predict)
        return scorer._score_func(convert_to_numpy(y_test), convert_to_numpy(y_pred)) 
    except Exception:
        raise
    
def get_custom_scorer(metric, model, X_test, y_test, custom_predict):
    try:
        X_test = convert_to_numpy(X_test)
        y_test = convert_to_numpy(y_test)
        y_pred = get_prediction(model, X_test, custom_predict)
        sig = signature(metric)
        if len(sig.parameters) ==3:
            score = metric(model, X_test, y_test)
        else:
            score = metric(y_pred, y_test)
        validate_score(score)
        return score 
    except Exception:
        raise

def validate_score(score):
    if score is None or isinstance(score, (str, bool)):
        raise TypeError("Score must be a quantitative value, not {}".format(type(score)))  
    if isinstance(score, (list, dict, tuple)):
        raise ValueError("Score must be a single value, not {}".format(type(score)))

    