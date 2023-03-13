from Robustness._sampling import sampleData
from Robustness._plot import plotNoiseCorruptionValues, plotNoiseCorruptionValuesHistogram, plotNoiseCorruptionBarScore
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
import random
from Noise.continuous import Gaussian_Noise
from Noise.discrete import Poisson_noise, Binomial_noise

def other(df, feature_name):
    return df[feature_name] * (-1) #TODO: makes no sense to plot this 

'********************************************************************************************************************************'

def filter_on_method(df, method, feature_name, level=None, random_state=None):
    switcher = {
        'other': lambda: other(df, feature_name, level),
        'Binomial': lambda: Binomial_noise(df, level, feature_name, random_state),
        'Gaussian': lambda: Gaussian_Noise(df, level, feature_name, random_state),
        'Poisson': lambda: Poisson_noise(df, feature_name, random_state)
    }
    return switcher.get(method, lambda: print("Invalid corruption method for feature {}".format(feature_name)))()

def train_model(model, X, y):
    model.fit(X, y.values.ravel())
    return model

def get_results(model, index):
    if hasattr(model, 'feature_importances_'):
        measured_property = 'feature importance'
        return model.feature_importances_[index], measured_property
    elif hasattr(model, 'coef_'):
        measured_property = 'coefficients'
        return model.coef_[0][index], measured_property
    elif hasattr(model, 'coefs_'): 
        measured_property = 'coefficients MLP'  # TODO: fix
        return model.coefs_[index], measured_property
    else:
        print("cound not calculate coefficients or feature importance")
        return None

def getLevels(methodSpecification, df):
    method = list(methodSpecification.keys())[0]
    if (method == "Gaussian" or method == "Binomial"):
        feature_names, levels = list(methodSpecification.values())[0][0], list(methodSpecification.values())[0][1]
        if all([isinstance(item, int) for item in feature_names]):
            feature_names = get_feature_name_from_index(feature_names, df)
        return feature_names, levels
    elif (method == "Poisson"):
        feature_names, levels = list(methodSpecification.values())[0][0], [-1]
        if all([isinstance(item, int) for item in feature_names]):
            feature_names = get_feature_name_from_index(feature_names, df)
        return feature_names, levels
    else:
        print('Error getting values')
        print(type(methodSpecification))

def initialize_progress_bar(corruption_list, corruptions, df):
    total = 0 
    for item in list(corruption_list):
        feature_names, levels = getLevels(item, df)
        total += ((len(feature_names) * len(levels)) * corruptions)
    return tqdm(total=total, desc="Total progress: ", position=0)

def set_random_seed(random_state):
    np.random.seed(random_state)
    random.seed(random_state)
    #import tensorflow as tf
    #tf.set_random_seed(seed_value)    

def baseline(df_train, X_test, y_test, model, label_name=None, random_state=None):
    baseline_results = pd.DataFrame(columns=['feature_name', 'value', 'variance', 'accuracy'])
    if (label_name is None):
        label_name = str(list(df_train)[-1])
    y = df_train[label_name]
    X = df_train.drop([label_name], axis=1)
    model = train_model(model, X, y)
    for feature_name in X.columns:
        index = df_train.columns.get_loc(feature_name)
        value, _ = get_results(model, index)
        variance = np.var(X[feature_name])
        accuracy = accuracy_score(y_test, model.predict(X_test))
        baseline_results.loc[len(baseline_results.index)] = [feature_name, value, variance, accuracy]
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

def get_feature_name_from_index(feature_names, df):
    return [list(df)[i] for i in feature_names]

def fill_in_column_names_for_indexes(df, corruption_list):
    for method in corruption_list:
        feature_names_string, _ = getLevels(method, df)
        for key, value in method.items():
            value[0] = feature_names_string
            method[key] = value
    return corruption_list

def corruptData(df_train, X_test, y_test, model, corruption_list, corruptions, column_names=None,  y_train=None, label_name=None, random_state=None, plot=True):
    set_random_seed(random_state)
    df_train = return_df_from_array_with_indexes_as_columns(df_train, column_names, y_train, label_name)
    X_test = return_df_from_array_with_indexes_as_columns(X_test, column_names)
    corruption_list = fill_in_column_names_for_indexes(df_train, corruption_list)
    corrupted_df = pd.DataFrame(columns=list(df_train))
    baseline_results, label_name = baseline(df_train, X_test, y_test, model, label_name)
    randomlist = random.sample(range(1, 1000), corruptions)
    progress_bar = initialize_progress_bar(corruption_list, corruptions, df_train)
    corruption_result_list = []
    for method in list(corruption_list):
        method_name = list(method.keys())[0]
        method_corrupt_df, corruption_result, measured_property = corruptDataMethod(df_train, X_test, y_test, model, method, randomlist, label_name, random_state, progress_bar)
        corruption_result_list.append(corruption_result)
        for column_name in list(method_corrupt_df):
            corrupted_df[column_name] = method_corrupt_df[column_name].values  
    if (plot):
        fig_1, fig_2 = plotData(baseline_results, corruption_result_list, str(model), corruptions, measured_property, method_name, corruption_list)
        fig_1.show()
        fig_2.show()
    corrupted_df = fill_in_missing_columns(corrupted_df, df_train)
    progress_bar.close()
    return corrupted_df, corruption_result

def corruptDataMethod(df_train, X_test, y_test, model, method, randomlist, label_name, random_state, progress_bar):
    corruption_result = pd.DataFrame(columns=['feature_name', 'level', 'value', 'variance', 'accuracy'])
    feature_names, levels = getLevels(method, df_train)
    method_corrupt_df = pd.DataFrame(columns=feature_names)
    for level in levels: 
        for feature_name in feature_names:
            average_value = []
            average_accuracy = []
            average_variance = []
            for random in randomlist:
                if (random == randomlist[-1]): 
                    X, y = sampleData(df_train, label_name, 1, random_state=random)
                else: 
                    X, y = sampleData(df_train, label_name, 0.4, random_state=random)
                X = filter_on_method(X, list(method.keys())[0], feature_name, level, random_state)  # TODO: no point in passing the whole DF to change one column
                average_variance.append(np.var(X[feature_name]))
                model = train_model(model, X, y)
                index = df_train.columns.get_loc(feature_name)
                measured_value, measured_property = get_results(model, index)
                average_value.append(measured_value)
                average_accuracy.append(accuracy_score(y_test, model.predict(X_test)))
                progress_bar.update(1)
            method_corrupt_df[feature_name] = X[feature_name].values
            average_variance = np.average(average_variance)
            average_value = np.average(average_value)
            average_accuracy = np.average(average_accuracy)
            corruption_result.loc[len(corruption_result.index)] = [feature_name, level, average_value, average_variance, average_accuracy]
    return method_corrupt_df, corruption_result, measured_property

def plotData(baseline_results, corruption_result_list, model_name, corruptions, measured_property, method_name, corruption_list):
    histogram_plot = []
    histogram_list = []
    line_plot = []
    line_list = []
    for corruption_result, corruption_type in zip(corruption_result_list, corruption_list):
        if (len(np.unique(corruption_result['level'].values)) < 3):
            histogram_plot.append(corruption_result)
            histogram_list.append(corruption_type)
        else:
            line_plot.append(corruption_result)
            line_list.append(corruption_type)
    if (len(histogram_plot) > 0):
        fig_1_1 = plotNoiseCorruptionValuesHistogram(baseline_results, histogram_plot, model_name, corruptions, measured_property, method_name, 'value', histogram_list)
        fig_2_1 = plotNoiseCorruptionValuesHistogram(baseline_results, histogram_plot, model_name, corruptions, measured_property, method_name, 'variance', histogram_list)
        #fig_3_1 = plotNoiseCorruptionBarScore(baseline_results, histogram_plot, model_name, corruptions, measured_property, method_name, 'accuracy', histogram_list)
        #fig_3_1 = plotNoiseCorruptionValuesHistogram(baseline_results, histogram_plot, model_name, corruptions, measured_property, method_name, 'accuracy', histogram_list)
        return fig_1_1, fig_2_1
    if (len(line_plot) > 0):
        fig_1_2 = plotNoiseCorruptionValues(baseline_results, line_plot, model_name, corruptions, measured_property, method_name, 'value', line_list)
        fig_2_2 = plotNoiseCorruptionValues(baseline_results, line_plot, model_name, corruptions, measured_property, method_name, 'variance', line_list)
        #fig_3_2 = plotNoiseCorruptionValues(baseline_results, line_plot, model_name, corruptions, measured_property, method_name,'accuracy', line_list)      
        return fig_1_2, fig_2_2



# TODO: check if coefs_ can be used 
# TODO: hardcoded y value. Need this as input when usinf DataFrames (or deafult last col)
# TODO: models that dont have fit?
# TODO: write test for randomness for sample + noisecorruption
