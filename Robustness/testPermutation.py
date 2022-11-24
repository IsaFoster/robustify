from _readData import getData, getDataFromFile
from _plot import plotPermutationImportance, plotMeanAccuracyDecrease
from permutationImportance import permutationImportance, meanAccuracyDecrease
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.inspection import permutation_importance
import random
from sklearn.ensemble import GradientBoostingClassifier

'*********** Load and Split Data ***********'
df_train, df_val, df_test = getDataFromFile()

'********** Reduced set ******************'
signal = df_train.loc[df_train['data_type'] == 1]
backgrond = df_train.loc[df_train['data_type'] == 0]

df_train_short = pd.concat([signal.iloc[:1000, :], backgrond.iloc[:9000, :]])
y_train_short = df_train_short[['data_type']]
X_train_short = df_train_short.drop(['data_type'], axis=1)

signal_test = df_test.loc[df_test['data_type'] == 1]
backgrond_test = df_test.loc[df_test['data_type'] == 0]

df_test_short = pd.concat([signal_test.iloc[:100, :], backgrond_test.iloc[:900, :]])
y_test_short = df_test_short[['data_type']]
X_test_short = df_test_short.drop(['data_type'], axis=1)

'********** Full set ***********************'
y_train = df_train[['data_type']]
X_train = df_train.drop(['data_type'], axis=1)

y_test = df_test[['data_type']]
X_test = df_test.drop(['data_type'], axis=1)
'*******************************************'

seed = 39
random.seed(seed)
np.random.seed(seed=seed)

print('Training')
model = GradientBoostingClassifier(n_estimators=10, verbose=1)
model.fit(X_train, y_train.values.ravel())
print('Training finished')
n_repeats = 10
random_state = 44

result = permutationImportance(model, X_test, y_test, n_repeats, random_state)
meanAccuracyDecrease(result, model, X_test, y_test, n_repeats, random_state)