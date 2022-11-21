from _readData import getData
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import plotly.express as px

'*********** Load and Split Data ***********'
df_train, df_val, df_test = getData()
#y_train = df_train[['data_type']]
#X_train = df_train.drop(['data_type'], axis=1)
#y_val = df_val[['data_type']]
#X_val =  df_val.drop(['data_type'], axis=1)
#y_test = df_test[['data_type']]
#X_test = df_test.drop(['data_type'], axis=1)
'******* Factor analysis procedure *********'

signal = df_train.loc[df_train['data_type'] == 1]
backgrond = df_train.loc[df_train['data_type'] == 0]

df_train_short = pd.concat([signal.iloc[:1000, :], backgrond.iloc[:9000, :]])
y_train_short = df_train_short[['data_type']]
df_train_short = df_train_short.drop(['data_type'], axis=1)


signal_test = df_test.loc[df_test['data_type'] == 1]
backgrond_test = df_test.loc[df_test['data_type'] == 0]

df_test_short = pd.concat([signal_test.iloc[:100, :], backgrond_test.iloc[:900, :]])
y_test_short = df_test_short[['data_type']]
df_test_short = df_test_short.drop(['data_type'], axis=1)

print(y_train_short.value_counts())
print(y_test_short.value_counts())

# skip section 2.2.1 factor analysis procedure 

'********* Noise Corruption *************'
# Gaussian noise 
dampenngFactors = np.linspace(0, 1, 11)

noise_list = []

def addNoiseList(features):
    for feature in features:
        feature_list = []
        for factor in dampenngFactors: 
            sd = np.std(feature)
            q = factor * sd 
            noise = np.random.normal(0, q, len(feature))
            a = feature + noise
            feature_list.append(a)
        noise_list.append(feature_list)


def addNoiseDf(df):
    for factor in dampenngFactors:
        df_temp = df.copy()
        for (name, feature) in df.items():
            new_feature = addNoiseColumn(feature, factor)
            df_temp[name] = new_feature
        noise_list.append(df_temp)

def addNoiseColumn(feature, factor):
    sd = np.std(feature)
    q = factor * sd 
    noise = np.random.normal(0, q, len(feature))
    a = feature + noise
    return a


addNoiseDf(df_train_short)
feature_names = df_train_short.columns

'********************************************'
models = []

models.append(RandomForestClassifier())
models.append(SVC())

def train(model):
    average_parameter_value = []
    feature_variance = []
    acc_list = []
    i = 0 
    for data in noise_list:
        av = []
        print(str(i))
        model.fit(data, y_train_short.values.ravel())
        pred = model.predict(df_test_short)
        acc_list.append(accuracy_score(pred, y_test_short))
        i += 1
        feature_variance.append(data.var().tolist())
        for feature in feature_names:
            av.append(data[feature].mean())
        average_parameter_value.append(av)
    return acc_list, average_parameter_value, feature_variance
    
acc_0, average_parameter_value_0, feature_variance_0 = train(models[0])
acc_1, average_parameter_value_1, feature_variance_1 = train(models[1])

plt.plot(dampenngFactors, acc_0, label='RandomForestClassifier')
plt.plot(dampenngFactors, acc_1, label='SVC')
plt.legend()
plt.show()

plt.plot(feature_names, average_parameter_value_0[0], label='RandomForestClassifier')
plt.plot(feature_names, average_parameter_value_1[0], label='SVC')
plt.legend()
plt.show()


df_test = pd.DataFrame(average_parameter_value_1, columns=feature_names)

df_1 = pd.DataFrame(dict(
    x = average_parameter_value_1[0],
    y = feature_names
))

df_2 = pd.DataFrame(dict(
    x = average_parameter_value_1[1],
    y = feature_names
))

fig = px.line(df_1, x="x", y="y", title="LALALA", color='y') 
#fig.add_trace(px.line(df_2, x="x", y="y", title="HIHIHI", color='y'))
fig.show()