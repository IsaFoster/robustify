from readData import getData
from sampling import sampleData
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.express as px
from tqdm import tqdm

'*********** Load and Split Data ***********'
df_train, df_val, df_test = getData()

signal = df_train.loc[df_train['data_type'] == 1]
backgrond = df_train.loc[df_train['data_type'] == 0]

df_train_short = pd.concat([signal.iloc[:1000, :], backgrond.iloc[:9000, :]])
#y_train_short = df_train_short[['data_type']]
#df_train_short = df_train_short.drop(['data_type'], axis=1)


signal_test = df_test.loc[df_test['data_type'] == 1]
backgrond_test = df_test.loc[df_test['data_type'] == 0]

df_test_short = pd.concat([signal_test.iloc[:100, :], backgrond_test.iloc[:900, :]])
y_test_short = df_test_short[['data_type']]
df_test_short = df_test_short.drop(['data_type'], axis=1)

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

'***********************************************'

def sort_df(df):
    df_temp = df.copy()
    df_temp.sort_values(by=['average_value'])
    return df_temp

def top5features(df):
    #df_ordered = df.loc[df['level'] == 0.0].groupby('feature_name').nlargest(5)
    #print(df_ordered.head(5))

    df_grouped = df.groupby('level')
    df_group = df_grouped.get_group(0.0)
    sorted = df_group.sort_values('average_value', ascending=False)
    return sorted['feature_name'].tolist()[:5]

def mostDiff(df):
    df_temp = pd.DataFrame(columns=['feature_name', 'most_diff'])
    df_grouped = df.groupby('feature_name')
    for group in df_grouped:
        li = group[1]['average_value'].tolist()
        diff = max(li)- min(li)
        row = pd.Series([group[0], diff])
        row = pd.DataFrame([[group[0], diff]], columns=['feature_name', 'most_diff'])
        df_temp = pd.concat([df_temp, row], axis=0)
    sorted = df_temp.sort_values('most_diff', ascending=False)
    return sorted['feature_name'].tolist()[:5]

def plotAll(df):
    visible_features = top5features(df)
    different_features = mostDiff(df)
    title = "Average feature importance over 100 replacement noise corruptions at increasing noise levels"
    fig = px.line(df, x="level", y="average_value", title=title, color='feature_name').update_traces(visible="legendonly", selector=lambda t: not t.name in visible_features) 
    fig.update_layout(dict(updatemenus=[
                        dict(
                            type = "buttons",
                            direction = "left",
                            buttons=list([
                                dict(
                                    args=["visible", "legendonly"],
                                    label="Deselect All",
                                    method="restyle"
                                ),
                                dict(
                                    args=["visible", True],
                                    label="Select All",
                                    method="restyle"
                                ),
                                dict(
                                    args=[{"visible": [i in visible_features for i in df['feature_name'].unique()]}],
                                    label="Top 5",
                                    method="restyle"
                                ),
                                dict(
                                    args=[{"visible": [i in different_features for i in df['feature_name'].unique()]}],
                                    label="Most diff",
                                    method="restyle"
                                )
                            ]),
                            pad={"r": 10, "t": 10},
                            showactive=False,
                            x=1,
                            xanchor="right",
                            y=1.1,
                            yanchor="top"
                        ),
                    ]
              ))
    fig.show()

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
                parameter_values.append(model.feature_importances_)
            elif hasattr(model, 'coef_'):
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
    plotAll(df_plot)
    return df_plot

model_1 = RandomForestClassifier(random_state=42)
model_2 = SVC(kernel='linear')

df_plot = doAll(df_train_short, df_test_short, y_test_short, model_1, corruptions=100)
df_plot = doAll(df_train_short, df_test_short, y_test_short, model_2, corruptions=100)

'************************************************'
