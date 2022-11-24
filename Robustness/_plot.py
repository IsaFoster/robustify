import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def topFromDfGrouped(df, group, groupValue, value, feature, num):
    df_grouped = df.groupby(group)
    df_group = df_grouped.get_group(groupValue)
    sorted = df_group.sort_values(value, ascending=False)
    return sorted[feature].tolist()[:num]

def topFromDf(df, value, feature, num):
    sorted = df.sort_values(value, ascending=False)
    return sorted[feature].iloc[:num].tolist()

def topFromSeries(df, num):
    df_temp = df.mean(axis=0)
    sorted = df_temp.sort_values(ascending=False)
    print("topFromSeries:", sorted.index[:num].tolist())
    return sorted.index[:num].tolist()

def mostDiff(df, groupValue):
    df_temp = pd.DataFrame(columns=['feature_name', 'most_diff'])
    df_grouped = df.groupby('feature_name')
    for group in df_grouped:
        li = group[1][groupValue].tolist()
        diff = max(li)- min(li)
        row = pd.Series([group[0], diff])
        row = pd.DataFrame([[group[0], diff]], columns=['feature_name', 'most_diff'])
        df_temp = pd.concat([df_temp, row], axis=0)
    sorted = df_temp.sort_values('most_diff', ascending=False)
    return sorted['feature_name'].tolist()[:5]

def plotButtons(visibleFeaturesList, featureNames):
    return [dict(
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
                args=[{"visible": [i in visibleFeaturesList for i in featureNames]}],
                label="Top 5",
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

def plotNoiseCorruptionsAverageFeatureValue(df, model_name, measured, corruptions, lalal):
    visible_features = topFromDfGrouped(df, 'level', 0.0, lalal, 'feature_name', 5)
    different_features = mostDiff(df, lalal)
    title = "Average {} of {} over {} replacement noise corruptions at increasing noise levels for {}".format(lalal, measured, corruptions, model_name)
    fig = px.line(df, x="level", y=lalal, title=title, color='feature_name').update_traces(visible="legendonly", selector=lambda t: not t.name in visible_features) 
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

def plotNoiseCorruptionsVariance():
    pass

def plotPermutationImportance(df, n_repeats, modelName):
    visible_features = topFromSeries(df, 5)
    title = "Permutation Importances using n={} for {}".format(n_repeats, modelName)
    fig = go.Figure()
    for (columnName, columnData) in df.items():
        fig.add_trace(go.Box(x=columnData, boxmean=True, name=columnName))

    fig.update_layout(dict(updatemenus=plotButtons(visible_features, df.columns),
                    legend={'traceorder': 'reversed'}
              ),
              title=title,
              xaxis_title="Decrease in accuracy score")
    fig.show()

def plotMeanAccuracyDecrease(df, result, permutations, modelName):
    title = "Feature importances using n={} permutation on {}".format(permutations, modelName)
    df_temp = pd.DataFrame(columns=['feature_name', 'value', 'error'])
    df_temp['feature_name'] = df.index.values.tolist()
    df_temp['value'] = df.values.tolist()
    df_temp['error'] = result.importances_std
    df_temp = df_temp.sort_values('value', ascending=False)
    visible_features = topFromDf(df_temp, 'value', 'feature_name', 5)

    fig = go.Figure()
    for (_, rowData) in df_temp.iterrows():
        fig.add_trace(
            go.Bar(
                x=[rowData['feature_name']],
                y=[rowData['value']],
                name=rowData['feature_name'],
                error_y={'type':'data', 'array':[rowData['error']]}
            )
        )
    fig.update_layout(dict(updatemenus=plotButtons(visible_features, df.index.values.tolist()),
              ),     
            title=title,
            yaxis_title="Mean accuracy decrease")
    fig.show()
