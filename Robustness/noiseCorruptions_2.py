from _sampling import sampleData
from _plot import plotNoiseCorruptionsAverageFeatureValue, plotNoiseCorruptionsVariance
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
from time import sleep

def addNoiseDf(X, factor, random_state):
    df_temp = X.copy()
    for (name, feature) in X.items():
        new_feature = addNoiseColumn(feature, factor, random_state)
        df_temp[name] = new_feature
    return df_temp

def addNoiseColumn(feature, factor, random_state=None):
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
    return df[feature_name] * add

def flip_sign(df, feature_name):
    return df[feature_name] * (-1)

def gaussian_noise(df, feature_name, level):
    return addNoiseColumn(df[feature_name], level)

def add_or_subtract(df, feature_name, level):
    # TODO: should be based of probability and value of other feature
    return df[feature_name] + level
    

def filter_on_method(df, method, feature_name, level=None):
    switcher = {
        'percentageShift': lambda: percentage_shift(df, feature_name, level),
        'flipSign': lambda: flip_sign(df, feature_name),
        'gaussianNoise': lambda: gaussian_noise(df, feature_name, level),
        'addOrSubtract': lambda: add_or_subtract(df, feature_name, level)
    }
    return switcher.get(method, lambda: print("Invalid corruption method for feature {}".format(feature_name)))()


def train_model(model, X, y):
    model.fit(X, y.values.ravel())
    return model

def get_results(model):
    if hasattr(model, 'feature_importances_'):
        measured = 'feature importance'
        return model.feature_importances_, measured
    elif hasattr(model, 'coef_'):
        measured = 'coefficients'
        return model.coef_, measured
    elif hasattr(model, 'coefs_'):  # TODO: see if this can be used 
        measured = 'coefficients MLP'
        return model.coefs_, measured
    else:
        print("cound not calculate coefficients or feature importance")
        return None




def getLevels(methodSpecification):
    if (isinstance(methodSpecification, dict)):
        return list(methodSpecification.keys())[0], list(methodSpecification.values())[0]
    elif (isinstance(methodSpecification, list) and len(methodSpecification) == 1):
        return methodSpecification[0], [-1]
    elif (isinstance(methodSpecification, list)):
        return methodSpecification[0], methodSpecification[1]
    else:
        print('Error getting values')
        print(type(methodSpecification))
# TODO: check for other usages/iputs

def all(df_train, X_test, y_test, model, corruption_dict, corruptions, random_state):
    for method in tqdm(list(corruption_dict.items()), desc="Total progress: ", position=0):
        method_name = method[0]
        corruption_result = corruptData(df_train, X_test, y_test, model, method, corruptions, random_state)
        plotData()

def corruptData(df_train, X_test, y_test, model, method, corruptions, random_state):
    corruption_result = pd.DataFrame(columns=['feature_name', 'values', 'level', 'variance'])
    feature_names, levels = getLevels(method[1])
    for level in levels: 
        for _ in tqdm(range (corruptions), desc="Level: {}".format(level), position=1, leave=False):
            X, y = sampleData(df_train, 'data_type', 0.4, random_state) #TODO: finn ut av random state (burde være fixed men ikke den samme for hver iterasjon)
            for feature_name in feature_names:
                X[feature_name] = filter_on_method(X, method[0], feature_name, level)
                model = train_model(model, X, y)
                measured_value, measured = get_results(model)
                print(measured_value)
                print(measured)
                # hent ut bare det featurene jeg er interessert i???
                ## lagre verdier i df
    # snitt av verdier 
    # returner df med snittverdier for alle features. 
    return None
def plotData():
    return
'************************************************'