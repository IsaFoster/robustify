from _readData import getDataFromFile
from noiseCorruptions import doAll
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import pandas as pd
import pickle

'*********** Load and Split Data ***********'
df_train, df_val, df_test = getDataFromFile()

signal = df_train.loc[df_train['data_type'] == 1]
backgrond = df_train.loc[df_train['data_type'] == 0]

signal_test = df_test.loc[df_test['data_type'] == 1]
backgrond_test = df_test.loc[df_test['data_type'] == 0]
'********** Reduced set ******************'
df_train_short = pd.concat([signal.iloc[:1000, :], backgrond.iloc[:9000, :]])
y_train_short = df_train_short[['data_type']]
X_train_short = df_train_short.drop(['data_type'], axis=1)

df_test_short = pd.concat([signal_test.iloc[:100, :], backgrond_test.iloc[:900, :]])
y_test_short = df_test_short[['data_type']]
X_test_short = df_test_short.drop(['data_type'], axis=1)

'********** Full set ***********************'
y_train = df_train[['data_type']]
X_train = df_train.drop(['data_type'], axis=1)

y_test = df_test[['data_type']]
X_test = df_test.drop(['data_type'], axis=1)
'*******************************************'

'********** Make and save models ***********'
modelName = "SVC_full_set"
model = SVC(kernel='linear')
model.fit(X_train, y_train.values.ravel())
pickle.dump(model, open('../Models/' + modelName, 'wb'))

modelName = "SVC_reduced_set"
model = SVC(kernel='linear')
model.fit(X_train_short, y_train_short.values.ravel())
pickle.dump(model, open('../Models/' + modelName, 'wb'))