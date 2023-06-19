from ._importances import filter_on_importance_method
from ._scorers import get_scorer
from ._transform import convert_to_numpy, normalize_max_min, make_scaler
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import tensorflow as tf
import keras

def is_keras_model(model):
    return isinstance(model, (tf.keras.Model, keras.Model, tf.estimator.Estimator))

def custom_train_model(model, X, y, custom_train):
    if not hasattr(custom_train, '__call__'):
        raise Exception("Custom training must be a callable function")
    try:
        X = convert_to_numpy(X)
        y = convert_to_numpy(y)
        return custom_train(model, X, y)
    except:
        raise Exception("Could not train the model")

def train_model(model, X, y, custom_train):
    if custom_train != None:
        return custom_train_model(model, X, y, custom_train)  
    if is_keras_model(model):
        model.fit(X.values, y.values.ravel(), verbose=0)
    else:
        model.fit(X.values, y.values.ravel())
    return model

def reset_model(model):
    if isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_uniform(model.weight.data)
    return model

def train_baseline(df_train, X_test, y_test, model, scorer, measure, label_name, random_state, custom_train, custom_predict, normalize):
    """ Train a baseline model on the data without any corruptions. 
    """
    baseline_results = pd.DataFrame(columns=['feature_name', 'value', 'variance', 'score'])
    if (label_name is None):
        label_name = str(list(df_train)[-1])
    y = df_train[label_name]
    X = df_train.drop([label_name], axis=1)
    if normalize:
        scaler = make_scaler(X)
        X = normalize_max_min(X, scaler)
        X_test = normalize_max_min(X_test, scaler)
    else:
        scaler = None
    model = train_model(model, X, y, custom_train)
    score = get_scorer(scorer, model, X_test, y_test, custom_predict)
    for feature_name in X.columns:
        index = df_train.columns.get_loc(feature_name)
        value, _ = filter_on_importance_method(model, index, X, y, random_state=random_state, scoring=scorer, measure=measure, custom_predict=custom_predict)
        variance = np.var(X[feature_name])
        baseline_results.loc[len(baseline_results.index)] = [feature_name, value, variance, score]
        model = reset_model(model)
    return baseline_results, label_name, scaler
