from Robustness.utils._sampling import sampleData
from Robustness.utils._plot import plotData
from Robustness.utils._importances import filter_on_importance_method
from Robustness.utils._scorers import get_scorer_sckit_learn, get_scorer
from Robustness.utils._train import reset_model, train_model
from Noise._filter import filter_on_method, getLevels
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import tensorflow as tf
import types
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

def set_random_seed(random_state):
    np.random.seed(random_state)
    random.seed(random_state)
    tf.random.set_seed(random_state) 

def initialize_progress_bar(corruption_list, corruptions, df):
    total = 1 
    for item in list(corruption_list):
        feature_names, levels = getLevels(item, df)
        total += ((len(feature_names) * len(levels)) * corruptions)
    return tqdm(total=total, desc="Total progress: ", position=0)

def baseline(df_train, X_test, y_test, model, metric, feature_importance_measure, label_name, random_state, custom_train, custom_predict):
    baseline_results = pd.DataFrame(columns=['feature_name', 'value', 'variance', 'score'])
    if (label_name is None):
        label_name = str(list(df_train)[-1])
    y = df_train[label_name]
    X = df_train.drop([label_name], axis=1)
    model = train_model(model, X, y, custom_train)
    score = get_scorer(metric, model, X_test, y_test, custom_predict)
    for feature_name in X.columns:
        index = df_train.columns.get_loc(feature_name)
        value, _ = filter_on_importance_method(model, index, X, y, random_state=random_state, scoring=metric, feature_importance_measure=feature_importance_measure, custom_predict=custom_predict)
        variance = np.var(X[feature_name])
        baseline_results.loc[len(baseline_results.index)] = [feature_name, value, variance, score]
        model = reset_model(model)
    return baseline_results, label_name

def fill_in_missing_columns(corrupted_df, df_train):
    for column_name in corrupted_df:
        if corrupted_df[column_name].isnull().all(): 
            corrupted_df[column_name] = df_train[column_name].values
    return corrupted_df

def return_df_from_array_with_indexes_as_columns(X, column_names, y=None, label_name=None):
    if (isinstance(X, (np.ndarray, np.generic, list))):
        df = pd.DataFrame(X, columns = column_names)
        if (label_name == None):
            label_name = str(len(df.columns))
        if (y is not None) and (label_name not in df):
            df[label_name] = y
        df.columns = df.columns.astype(str)
        return df
    return X

def fill_in_column_names_for_indexes(df, corruption_list):
    for method in corruption_list:
        feature_names_string, levels = getLevels(method, df)
        for key, value in method.items():
            value = [feature_names_string, levels]
            method[key] = value
    return corruption_list

def corruptData(df_train, X_test, y_test, model, metric, corruption_list, corruptions, column_names=None, y_train=None, label_name=None, feature_importance_measure=None, random_state=None, plot=True, custom_train=None, custom_predict=None):
    set_random_seed(random_state)
    df_train = return_df_from_array_with_indexes_as_columns(df_train, column_names, y_train, label_name)
    X_test = return_df_from_array_with_indexes_as_columns(X_test, column_names)
    corruption_list = fill_in_column_names_for_indexes(df_train, corruption_list)
    progress_bar = initialize_progress_bar(corruption_list, corruptions, df_train)
    corrupted_df = pd.DataFrame(columns=list(df_train))
    baseline_results, label_name = baseline(df_train, X_test, y_test, model, metric, feature_importance_measure, label_name, random_state, custom_train, custom_predict)
    progress_bar.update(1)
    randomlist = random.sample(range(1, 1000), corruptions)
    corruption_result_list = []
    for method in list(corruption_list):
        method_name = list(method.keys())[0]
        method_corrupt_df, corruption_result, measured_property = corruptDataMethod(df_train, X_test, y_test, model, metric, feature_importance_measure, method, randomlist, label_name, random_state, progress_bar, custom_train, custom_predict)
        corruption_result_list.append(corruption_result)
        for column_name in list(method_corrupt_df):
            corrupted_df[column_name] = method_corrupt_df[column_name].values  
    if (plot):
        plotData(baseline_results, corruption_result_list, str(model), corruptions, measured_property, method_name, corruption_list)
    corrupted_df = fill_in_missing_columns(corrupted_df, df_train)
    progress_bar.close()
    return corrupted_df, corruption_result

def corruptDataMethod(df_train, X_test, y_test, model, metric, feature_importance_measure, method, randomlist, label_name, random_state, progress_bar, custom_train, custom_predict):
    corruption_result = pd.DataFrame(columns=['feature_name', 'level', 'value', 'variance', 'score'])
    feature_names, levels = getLevels(method, df_train)
    method_corrupt_df = pd.DataFrame(columns=feature_names)
    for level in levels: 
        for feature_name in feature_names:
            average_value = []
            average_score = []
            average_variance = []
            for random in randomlist:
                if (random == randomlist[-1]): 
                    X, y = sampleData(df_train, label_name, 1, random_state=random)
                else: 
                    X, y = sampleData(df_train, label_name, 0.4, random_state=random)
                X = filter_on_method(X, list(method.keys())[0], feature_name, level, random_state)
                average_variance.append(np.var(X[feature_name]))
                model = train_model(model, X, y, custom_train)
                index = df_train.columns.get_loc(feature_name)
                measured_value, measured_property = filter_on_importance_method(model, index, X, y, random_state=random, scoring=metric, feature_importance_measure=feature_importance_measure, custom_predict=custom_predict)
                average_value.append(measured_value)
                score = get_scorer(metric, model, X_test, y_test, custom_predict)
                average_score.append(score)
                model = reset_model(model)
                progress_bar.update(1)
            method_corrupt_df[feature_name] = X[feature_name].values
            average_variance = np.average(average_variance)
            average_value = np.average(average_value)
            average_score = np.average(average_score)
            corruption_result.loc[len(corruption_result.index)] = [feature_name, level, average_value, average_variance, average_score]
    return method_corrupt_df, corruption_result, measured_property