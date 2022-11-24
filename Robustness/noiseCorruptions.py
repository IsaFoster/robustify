from _sampling import sampleData
from _plot import plotNoiseCorruptionsAverageFeatureValue, plotNoiseCorruptionsVariance
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm

def addNoiseDf(df, factor):
    df_temp = df.copy()
    for (name, feature) in df.items():
        new_feature = addNoiseColumn(feature, factor)
        df_temp[name] = new_feature
    return df_temp

def addNoiseColumn(feature, factor):
    sd = np.std(feature)
    q = factor * sd 
    noise = np.random.normal(0, q, len(feature))
    a = feature + noise
    return a

def sort_df(df):
    df_temp = df.copy()
    df_temp = df_temp.sort_values(by=['average_value'])
    return df_temp

'***********************************************'

def noiseCorruptions(df, X_test, y_test, model, corruptions=10, levels=np.linspace(0, 1, 11)):
    pbar_outer = tqdm(levels, desc="Total progress: ", position=0)
    df_plot = pd.DataFrame(columns=['feature_name', 'average_value', 'level'])

    average_accuracy_all = []

    for level in pbar_outer:
        parameter_values = []
        accuracy_values = []
        pbar = tqdm(range (corruptions), desc="Level: {}".format(level), position=1, leave=False)
        for _ in pbar:
            X, y = sampleData(df, 0.2)

            # corrupt
            corrupted_noise = addNoiseDf(X, level)
            model.fit(corrupted_noise, y.values.ravel())
            if hasattr(model, 'feature_importances_'):
                measured = 'feature importance'
                parameter_values.append(model.feature_importances_)
            elif hasattr(model, 'coef_'):
                measured = 'coefficients'
                parameter_values.append(model.coef_)
            else:
                print("cound not calculate coefficients or feature importance")
                return 
            accuracy_values.append(accuracy_score(model.predict(X_test), y_test))
        average_accuracy_all.append(np.average(accuracy_values))

        parameter_values_np = np.array(parameter_values)
        average_level_value = np.average(parameter_values_np, axis=0)
        feature_names = X.columns
        df_temp = pd.DataFrame({'feature_name': feature_names, 'average_value': average_level_value.flatten(), 'level': np.array([level]*len(feature_names))})
        df_plot = pd.concat([df_plot, df_temp], axis=0)
    df_plot = sort_df(df_plot)

    plotNoiseCorruptionsAverageFeatureValue(df_plot, str(model), measured, corruptions)
    plotNoiseCorruptionsVariance()
    return df_plot

'************************************************'

# TODO: sort df before plotting
# TODO: make reproducible