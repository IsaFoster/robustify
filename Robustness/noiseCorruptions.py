from _sampling import sampleData
from _plot import plotNoiseCorruptionValues, plotNoiseCorruptionValuesHistogram
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
import random
from Noise.continuous import Gaussian_Noise
from Noise.discrete import Poisson_noise

def other(df, feature_name):
    return df[feature_name] * (-1) #TODO: makes no sense to plot this 

def another(df, feature_name, level):
    # TODO: should be based of probability and value of other feature
    return df[feature_name] + level

'********************************************************************************************************************************'
 

def filter_on_method(df, method, feature_name, level=None, random_state=None):
    switcher = {
        'other': lambda: other(df, feature_name, level),
        'another': lambda: another(df, feature_name),
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
        return model.coef_, measured_property
    elif hasattr(model, 'coefs_'): 
        measured_property = 'coefficients MLP'
        return model.coefs_[index], measured_property
    else:
        print("cound not calculate coefficients or feature importance")
        return None


def getLevels(methodSpecification):
    if (isinstance(methodSpecification, dict)):
        return list(methodSpecification.keys())[0], list(methodSpecification.values())[0]
    elif (isinstance(methodSpecification, list) and len(methodSpecification) == 1):
        return [methodSpecification[0]], [-1]
    elif (isinstance(methodSpecification, list)):
        return methodSpecification[0], methodSpecification[1]
    else:
        print('Error getting values')
        print(type(methodSpecification))


def initialize_progress_bar(corruption_dict, corruptions):
    total = 0 
    for item in list(corruption_dict.items()):
        feature_names, levels = getLevels(item[1])
        total += ((len(feature_names) * len(levels)) * corruptions)
    return tqdm(total=total, desc="Total progress: ", position=0)

def set_random_seed(random_state):
    np.random.seed(random_state)
    random.seed(random_state)
    #import tensorflow as tf
    #tf.set_random_seed(seed_value)    

def all(df_train, X_test, y_test, model, corruption_dict, corruptions, random_state=None, plot=True):
    if (random_state):
        set_random_seed(random_state)
    randomlist = random.sample(range(1, 1000), corruptions)
    progress_bar = initialize_progress_bar(corruption_dict, corruptions)
    for method in list(corruption_dict.items()):
        method_name = method[0]
        corruption_result, measured_property = corruptData(df_train, X_test, y_test, model, method, randomlist, random_state, progress_bar)
        if (plot):
            plotData(corruption_result, str(model), corruptions, measured_property, method_name)
    progress_bar.close()

def corruptData(df_train, X_test, y_test, model, method, randomlist, random_state, progress_bar):
    corruption_result = pd.DataFrame(columns=['feature_name', 'level', 'value', 'variance', 'accuracy'])
    feature_names, levels = getLevels(method[1])
    for level in levels: 
        for feature_name in feature_names:
            average_value = []
            average_accuracy = []
            average_variance = []
            for random in randomlist:
                X, y = sampleData(df_train, 'data_type', 0.4, random_state=random)
                X = filter_on_method(X, method[0], feature_name, level, random_state)  # TODO: no point in passing the whole DF to change one column
                average_variance.append(np.var(X[feature_name]))
                model = train_model(model, X, y)
                index = df_train.columns.get_loc(feature_name)
                measured_value, measured_property = get_results(model, index)
                average_value.append(measured_value)
                average_accuracy.append(accuracy_score(y_test, model.predict(X_test)))
                progress_bar.update(1)
            average_variance = np.average(average_variance)
            average_value = np.average(average_value)
            average_accuracy = np.average(average_accuracy)
            corruption_result.loc[len(corruption_result.index)] = [feature_name, level, average_value, average_variance, average_accuracy]
    return corruption_result, measured_property

def plotData(corruption_result, model_name, corruptions, measured_property, method_name):
    if (len(np.unique(corruption_result['level'].values)) < 3):
        plotNoiseCorruptionValuesHistogram(corruption_result, model_name, corruptions, measured_property, method_name, 'value')
        print(corruption_result)
    else:
        plotNoiseCorruptionValues(corruption_result, model_name, corruptions, measured_property, method_name, 'value')
        plotNoiseCorruptionValues(corruption_result, model_name, corruptions, measured_property, method_name, 'variance')
        plotNoiseCorruptionValues(corruption_result, model_name, corruptions, measured_property, method_name,'accuracy')



# TODO: use another type of plot when theres only one value? 
# TODO: check if coefs_ can be used 
# TODO: finn ut av random state (burde være fixed men ikke den samme for hver iterasjon) 
# TODO: hardcoded y value. Need this as input when usinf DataFrames (or deafult last col)
# TODO: models that dont have fit?
# TODO: write test for randomness for sample + noisecorruption
# TODO: 