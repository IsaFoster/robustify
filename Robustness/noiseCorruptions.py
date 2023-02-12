from _sampling import sampleData
from _plot import plotNoiseCorruptionsAverageFeatureValue, plotNoiseCorruptionsVariance
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
from keras.utils import to_categorical 
from tensorflow import keras

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

'***********************************************'

def noiseCorruptions(df, X_test, y_test, model, random_state=None, corruptions=10, levels=np.linspace(0, 1, 11), plot=True):
    pbar_outer = tqdm(levels, desc="Total progress: ", position=0)
    df_plot_average_value = pd.DataFrame(columns=['feature_name', 'feature_value', 'level'])
    df_plot_feature_variance = pd.DataFrame(columns=['feature_name', 'feature_variance', 'level'])

    average_accuracy_all = []

    for level in pbar_outer:
        parameter_values = []
        accuracy_values = []
        feature_variance = []
        pbar = tqdm(range (corruptions), desc="Level: {}".format(level), position=1, leave=False)
        for _ in pbar:
            X, y = sampleData(df, 'data_type', 0.2, random_state)

            # corrupt
            corrupted_noise = addNoiseDf(X, level, random_state)
            if hasattr(model, 'compile'):
                model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                loss="binary_crossentropy", 
                metrics=['accuracy'])
                model.fit(corrupted_noise, to_categorical(y.values.ravel()), 
                epochs=500, 
                batch_size=1000,
                verbose=0)
            else:
                model.fit(corrupted_noise, y.values.ravel())
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