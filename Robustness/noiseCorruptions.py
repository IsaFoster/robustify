from _readData import getData, getDataFromFile
from _sampling import sampleData
from _plot import plotNoiseCorruptions
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm

'*********** Load and Split Data ***********'
df_train, df_val, df_test = getDataFromFile()

signal = df_train.loc[df_train['data_type'] == 1]
backgrond = df_train.loc[df_train['data_type'] == 0]

df_train_short = pd.concat([signal.iloc[:1000, :], backgrond.iloc[:9000, :]])
y_train_short = df_train_short[['data_type']]
X_train_short = df_train_short.drop(['data_type'], axis=1)


signal_test = df_test.loc[df_test['data_type'] == 1]
backgrond_test = df_test.loc[df_test['data_type'] == 0]

y_test = df_test[['data_type']]
X_test = df_test.drop(['data_type'], axis=1)
df_test_short = pd.concat([signal_test.iloc[:100, :], backgrond_test.iloc[:900, :]])
y_test_short = df_test_short[['data_type']]
X_test_short = df_test_short.drop(['data_type'], axis=1)

'*********************************************'

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
    df_temp.sort_values(by=['feature_name'])
    return df_temp

'***********************************************'

def doAll(df, X_test, y_test, model, corruptions=10, levels=np.linspace(0, 1, 11)):
    pbar_outer = tqdm(levels, desc="Total progress: ", position=0)
    df_plot = pd.DataFrame(columns=['feature_name', 'average_value', 'level'])

    average_accuracy_all = []

    for level in pbar_outer:
        parameter_values = []
        accuracy_values = []
        pbar = tqdm(range (corruptions), desc="Level: {}".format(level), position=1, leave=False)
        for _ in pbar:

            # sample
            df_ttemp = df.copy()
            X, y = sampleData(df_ttemp, 0.2)

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
    plotNoiseCorruptions(df_plot, str(model), measured, corruptions)
    return df_plot

model_1 = RandomForestClassifier(random_state=42)
model_2 = SVC(kernel='linear')

df_plot = doAll(df_train_short, X_test, y_test, model_1, corruptions=10)
df_plot = doAll(df_train_short, X_test, y_test_short, model_2, corruptions=10)

'************************************************'

# TODO: sort df before plotting
# TODO: make reproducible