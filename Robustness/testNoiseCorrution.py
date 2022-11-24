from _readData import getData, getDataFromFile
from _sampling import sampleData
from noiseCorruptions import doAll
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
import random


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
model = GradientBoostingClassifier(verbose=1)
model.fit(X_train, y_train.values.ravel())
print('Training finished')
n_repeats = 10
random_state = 44

doAll(df_train, X_test, y_test, model, corruptions=10)