from sklearn import metrics
import pandas as pd
import tensorflow as tf
import numpy as np

def get_scorer(metric, model, X_test, y_test, custom_predict):
    if isinstance(metric, str):
        return get_scorer_sckit_learn(metric, model, X_test, y_test, custom_predict)
    elif hasattr(metric, '__call__'):
        return get_custom_scorer(metric, model, X_test, y_test, custom_predict)
    else:
        raise Exception("No usabel scorer object found")
    
def get_scorer_sckit_learn(metric, model, X_test, y_test, custom_predict):
    # from https://scikit-learn.org/stable/modules/model_evaluation.html
    try: 
        scorer = metrics.get_scorer(metric)
        return scorer._score_func(y_test, model.predict(X_test))
    except:
        print("not recognized")

def get_custom_scorer(metric, model, X_test, y_test, custom_predict):
    try:
        X_test = convert_to_numpy(X_test)
        y_test = convert_to_numpy(y_test)
        y_pred = custom_predict(model, X_test)
        y_pred = convert_to_numpy(y_pred)
        print("y?pred before", y_pred[0:10])
        return metric(y_pred, y_test)
    except: raise Exception ("soemthing went wrong")  

def convert_to_numpy(col):
    if isinstance(col, pd.DataFrame):
        return col.to_numpy()
    elif tf.is_tensor(col):
        return col.numpy()
    elif isinstance(col, np.ndarray):
        return col
    else:
        raise Exception("could not convert {} to numpy").__format__(type(col))