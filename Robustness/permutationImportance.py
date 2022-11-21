from _readData import getData
from _plot import plotPermutationImportance
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.inspection import permutation_importance

'*********** Load and Split Data ***********'

df_train, df_val, df_test = getData()

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

'*******************************************'

model = RandomForestClassifier(random_state=42)
model.fit(X_train_short, y_train_short.values.ravel())

'*******************************************'

def permutationImportance(model, X_test, y_test, n_repeats):
    return permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42, n_jobs=-1)


result = permutationImportance(model, X_test_short, y_test_short, 10)

sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=X_train_short.columns[sorted_importances_idx],
)

'''ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances (test set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
ax.figure.show()
'''

#import plotly.express as px
#fig = px.box(importances, orientation='h')
#fig.show()


plotPermutationImportance(importances)

'''
fig = go.Figure()
# Use x instead of y argument for horizontal plot
for (columnName, columnData) in importances.iteritems():
    fig.add_trace(go.Box(x=columnData))

fig.show()'''