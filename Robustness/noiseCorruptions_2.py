from _sampling import sampleData
from _plot import plotNoiseCorruptionsAverageFeatureValue, plotNoiseCorruptionsVariance
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm

def addNoiseDf(X, factor, random_state):
    df_temp = X.copy()
    for (name, feature) in X.items():
        new_feature = addNoiseColumn(feature, factor, random_state)
        df_temp[name] = new_feature
    return df_temp

def addNoiseColumn(feature, factor, random_state):
    np.random.seed(random_state)
    sd = np.std(feature)
    q = factor * sd 
    noise = np.random.normal(0, q, len(feature))
    a = feature + noise
    return a

def sort_df(df):
    df_temp = df.copy()
    df_temp = df_temp.sort_values(by=['feature_name'])
    return df_temp



def percentage_shift(df, feature_name, percentage):
    add = (1+(percentage*0.1))
    df[feature_name] = df[feature_name] * add
    print("doing percentage shifts as specified by percentages for {} with level: {}".format(feature_name, add))
    return df

def flip_sign(df, feature_name):
    df[feature_name] = df[feature_name] * (-1)
    print("doing flips signs for {}".format(feature_name))
    return df

def gaussian_noise(df, feature_name, level):
    print("adding gaussian noise for {} with levels: {}".format(feature_name, level))

def add_or_subtract(df, feature_name, level):
    # TODO: should be based of probability and value of other feature
    df[feature_name] = df[feature_name] + level
    print("adding fixed noise for {} with value: {}".format(feature_name, level))
    return df
    

def filter_on_method(df, method, feature_name, level=None):
    switcher = {
        'percentageShift': lambda: percentage_shift(df, feature_name, level),
        'flipSign': lambda: flip_sign(df, feature_name),
        'gaussianNoise': lambda: gaussian_noise(df, feature_name, level),
        'addOrSubtract': lambda: add_or_subtract(df, feature_name, level)
    }
    return switcher.get(method, lambda: print("Invalid corruption method for feature {}".format(feature_name)))()


'***********************************************'

def noiseCorruptions_2(df, X_test, y_test, model, corruption_dict, random_state=None, plot=True):
    pbar_outer = tqdm(list(corruption_dict.items()), desc="Total progress: ", position=0)
    df_plot_average_value = pd.DataFrame(columns=['feature_name', 'feature_value', 'level'])
    df_plot_feature_variance = pd.DataFrame(columns=['feature_name', 'feature_variance', 'level'])

    average_accuracy_all = []

    corruptions = 10

    for level in pbar_outer:
        parameter_values = []
        accuracy_values = []
        feature_variance = []

        feature_name = level[0]
        if (isinstance(level[1], dict)):
            method = list(level[1].keys())[0]
            level = list(level[1].values())[0]
        elif (isinstance(level[1], set) and len(level[1]) == 1):
            method = list(level[1])[0]
            level=None
        else:
            print('Error getting values')
            print(type(level[1]))
        for tic in level:
            for _ in tqdm(range (corruptions), desc="Level: {}".format(level), position=1, leave=False):
                corrupted_noise = filter_on_method(df, method, feature_name, tic)
                X, y = sampleData(corrupted_noise, 'data_type', 0.4, random_state)
                model.fit(X, y.values.ravel())
                if hasattr(model, 'feature_importances_'):
                    measured = 'feature importance'
                    parameter_values.append(model.feature_importances_)
                elif hasattr(model, 'coef_'):
                    measured = 'coefficients'
                    parameter_values.append(model.coef_)
                elif hasattr(model, 'coefs_'):  # TODO: see if this can be used 
                    measured = 'coefficients MLP'
                    parameter_values.append(model.coefs_)
                else:
                    print("cound not calculate coefficients or feature importance")
                    return 
                accuracy_values.append(accuracy_score(model.predict(X_test), y_test))
                feature_variance.append(corrupted_noise.var().tolist())
        average_accuracy_all.append(np.average(accuracy_values))
        

        parameter_values_np = np.array(parameter_values)
        average_level_value = np.average(parameter_values_np, axis=0)
        average_feature_variance = np.average(feature_variance, axis=0)
        feature_names = X.columns
        df_temp_average = pd.DataFrame({'feature_name': feature_names, 'feature_value': average_level_value.flatten(), 'level': np.array([level]*len(feature_names))})
        df_temp_variance = pd.DataFrame({'feature_name': feature_names, 'feature_variance': average_feature_variance.flatten(), 'level': np.array([level]*len(feature_names))})
        df_plot_average_value = pd.concat([df_plot_average_value, df_temp_average], axis=0)
        df_plot_feature_variance = pd.concat([df_plot_feature_variance, df_temp_variance], axis=0)
    if plot:
        plotNoiseCorruptionsAverageFeatureValue(df_plot_average_value, str(model), measured, corruptions, 'feature_value')
        plotNoiseCorruptionsAverageFeatureValue(df_plot_feature_variance, str(model), measured, corruptions, 'feature_variance')
    return average_accuracy_all

'************************************************'

# TODO: sort df before plotting NOOPE