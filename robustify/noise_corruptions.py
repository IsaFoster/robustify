
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import Bunch
from .utils._sampling import sample_data
from .utils._plot import plot_data
from .utils._importances import filter_on_importance_method
from .utils._scorers import get_scorer
from .utils._train import reset_model, train_model, train_baseline
from .utils._filter import filter_on_method, get_levels
from .utils._progress import initialize_progress_bar
from .utils._transform import df_from_array, check_corruptions, fill_missing_columns

def set_random_seed(random_state):
    """ Set random seed for different packages
    Parameters
    ----------
    random_state: int, RandomState instance or None, deafult=None Controls
    randomness external packages
    """
    np.random.seed(random_state)
    random.seed(random_state)
    tf.random.set_seed(random_state)

def corrupt_data(model, corruption_list, X_train, X_test, scorer, y_train=None,
                 y_test=None, column_names=None, label_name=None, measure=None,
                 n_corruptions=10, random_state=None, custom_train=None, 
                 custom_predict=None, show_plots=True):
    """ Perform noise corruption on tabular data.

    Parameters
    ----------
    model: object
        An object which manages the estimation and decoding of a model. Must
        either provide a fit and predict method, or these will have to be
        specified in custom_train and custom_predict.
	
    corruption_list: dict or list of dicts
        Dict of noise corruptions to be performed. The key is a string (or list
        of strings)
          corresponding to feature name or index. Values are list of noise
          method and levels (where applicable). Pass a list of dicts to perform
          several noise corruptions. See <insert link to examples> for examples.
	
    X_train: ndarray, dataframe or sparse matrix of shape (n_samples,
    n_features)
        The training input samples. Will be transformed to dataframe object (if
        it is not already) with column names from column_names property, or
        index values as names. Can contain the target values if label_name is
        specified.
	
    X_test: ndarray, dataframe or sparse matrix of shape (n_samples, n_features)
        The testing input samples. n_features must be equal for X_train and
        X_test, and in the same order. Will use column_names as specified in the
        column_names property, or indexes. Can contain the target values if
        label_name is specified.
	
    score_func: str or callable.
        Scoring method as string, use sklearn.metrics.get_scorer_names to find
        available scorers. If callable score function (or loss function)
        required signature is either score_func(y_pred, y_true) or (model,
        y_pred, y_true).
	
    y_train: ndarray, dataframe or sparse of shape (n_samples,) or (n_samples,
    n_outputs), deafult=None
        The training target values, labels in classification, real numbers in
        regression. Required if label_name is None and if it is not included in
        X_train as a feature.
	
    y_test: ndarray, dataframe or sparse of shape (n_samples,) or (n_samples,
    n_outputs), default=None**
        The testing target values. Required if label_name is None and if it is
        not included in X_test as a feature.
	
    column_names: list, default=None
        List of column names for input samples. Values defaults to index values
        if None. Not required if X_train is a dataframe with column names.
	
    label_name: str or int, deafult=None
        Column name or column index corresponding to the target values. Required 
        if y_train or y_test are None and X_train is a DataFrame containg the 
        target values.
	
    measure: {None, "eli5", "lime", "shap"}, default=None
        If None default is either coef_ or feature_importances_ attributes where
        applicable, or permutation importance. Not available for models that use
        custom fit or predict methods. See further explanation of the other
        methods <link>here.
	
    n_corruptions: int, default=10
        Number of corruptions to average over for each feature level. Decreasing
        the number of corruptions will decrease running time, but can make the
        results of Gaussian noise more vulnerable to outliers drawn from the
        distribution.
	
    random_state: int, RandomState instance or None, deafult=None
        Controls randomness in sampling and noise addition. Note: models with
        aspects of randomness in training or predicting are not affected by this
        property, and must be set on model initialization.
	
    custom_train: callable, default=None
        Callable with signature train_func(model, X, y). Must return trained
        model object. Not required if model
          object has fit method with signature fit(X, y).
	
    custom_predict: callable, default=None
        Callable with signature predict_func(model, X). Must return list,
        ndarray, tensor, or dataframe of same length as X. Not required if model
        object has predict method with signature predict(X).
	
    show_plots: bool, default=True
        Determines whether plots of feature importances, variance and score are
        shown. 
          
    Returns
    -------
	result: Bunch or dict of such instances.
        Dictionary-like object, with the following attributes: corrupted_df:
        DataFrame
            The noisy data as a result of all corruptions specified in
            corruption_list. If levels are specified as a range or list of
            values, corrupted_df will be a result of the final level.
        corruption_result: DataFrame
            The value, variance and score (as defined by the scorer) for each
            feature and each level specified in corruption_list.
        value_plot: Figure
            A plot showing the average value of the specified features for each noise corruption. 
        variance_plot: Figure
            A plot showing the average variance of the value of the specified features for each noise corruption. 
        score_plot: Figure
            A plot showing the average score of the specified features for each noise corruption. 
    """
    set_random_seed(random_state)
    df_train, label_name = df_from_array(X_train, column_names, y_train, label_name)
    X_test, _ = df_from_array(X_test, column_names)
    corruption_list = check_corruptions(df_train, corruption_list)
    progress_bar = initialize_progress_bar(corruption_list, n_corruptions, df_train)
    corrupted_df = pd.DataFrame(columns=list(df_train))
    baseline_results, label_name = train_baseline(df_train, X_test, y_test, model,
                                                  scorer, measure, label_name, random_state,
                                                  custom_train, custom_predict)
    progress_bar.update(1)
    randomlist = random.sample(range(1, 1000), n_corruptions)
    corruption_results = pd.DataFrame(columns=['feature_name', 'level',
                                               'value', 'variance', 'score'])

    for method in list(corruption_list):
        method_name = list(method.keys())[0]
        method_corrupt_df, corruption_result, measured_property = perform_corruption(
                                                                df_train, X_test, y_test, model, scorer,
                                                                measure, method, randomlist, label_name,
                                                                random_state, progress_bar, custom_train,
                                                                custom_predict)
                                                                
        corruption_results = pd.concat([corruption_results, corruption_result])

        for column_name in list(method_corrupt_df):
            corrupted_df[column_name] = method_corrupt_df[column_name].values

    
    value_plot, variance_plot, score_plot = plot_data(baseline_results, corruption_results, str(model), n_corruptions,
                  measured_property, corruption_list)
    if show_plots:
        value_plot.show()
        variance_plot.show()
        score_plot.show()
        
    corrupted_df = fill_missing_columns(corrupted_df, df_train)
    progress_bar.close()
    corruption_results = corruption_results.sort_values(by=['feature_name', 'level'])
    result = Bunch(corrupted_df=corrupted_df, corruption_result=corruption_results, value_plot=value_plot, variance_plot=variance_plot, score_plot=score_plot)
    return result

def perform_corruption(df_train, X_test, y_test, model, scorer, measure, method,
                       randomlist, label_name, random_state, progress_bar,
                       custom_train, custom_predict):
    """ Perfroms a specific noise corruption on features. 

    Will perfrom corruptions n_corruptions times (determined by randomlist) and
    average the value, variance and score for each level and for for each
    feature. 
    """
    corruption_result = pd.DataFrame(columns=['feature_name', 'level', 'value', 'variance', 'score'])
    feature_names, levels = get_levels(method, df_train)
    method_corrupt_df = pd.DataFrame(columns=feature_names)
    for level in levels:
        for feature_name in feature_names:
            average_value = []
            average_score = []
            average_variance = []
            for random_int in randomlist:
                if random_int == randomlist[-1]:
                    X, y = sample_data(df_train, label_name, 1, random_state=random_int)
                else:
                    X, y = sample_data(df_train, label_name, 0.4, random_state=random_int)
                X = filter_on_method(X, list(method.keys())[0], feature_name, level, random_state)
                average_variance.append(np.var(X[feature_name]))
                model = train_model(model, X, y, custom_train)
                index = df_train.columns.get_loc(feature_name)
                measured_value, measured_property = filter_on_importance_method(model, index, X, y,
                                                                        random_state=random_int,
                                                                        scoring=scorer,
                                                                        measure=measure,
                                                                        custom_predict=custom_predict)
                average_value.append(measured_value)
                score = get_scorer(scorer, model, X_test, y_test, custom_predict)
                average_score.append(score)
                model = reset_model(model)
                progress_bar.update(1)
            method_corrupt_df[feature_name] = X[feature_name].values
            average_variance = np.average(average_variance)
            average_value = np.average(average_value)
            average_score = np.average(average_score)
            corruption_result.loc[len(corruption_result.index)] = [feature_name,
                                                                   level, average_value,
                                                                   average_variance,
                                                                   average_score]
    return method_corrupt_df, corruption_result, measured_property
