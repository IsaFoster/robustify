import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def topFromDf(df, n):
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

def plotNoiseCorruptions(df, model_name, measured, corruptions):
    visible_features = topFromDf(df, 5)
    different_features = mostDiff(df)
    title = "Average {} over {} replacement noise corruptions at increasing noise levels for {}".format(measured, corruptions, model_name)
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


def topFromSeries(df, num):
    df2 = df.mean(axis=0)
    sorted = df2.sort_values(ascending=False)
    return sorted.index[:num]

def plotPermutationImportance(df):
    visible_features = topFromSeries(df, 5)

    fig = go.Figure()
    for (columnName, columnData) in df.items():
        fig.add_trace(go.Box(x=columnData, boxmean=True, name=columnName))

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
                                    args=[{"visible": [i in visible_features for i in df.columns]}],
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
                    ],
                    legend={'traceorder': 'reversed'}
              ))
    fig.show()

def topFromSeries2(df, num):
    sorted = df.sort_values(ascending=False)
    return sorted.index[:num]

def plotMeanAccuracyDecrease(df, result):
    visible_features = topFromSeries2(df, 5)
    title = "Feature importances using permutation on full model"
    df_temp = pd.DataFrame(columns=['feature_name', 'value', 'error'])
    df_temp['feature_name'] = df.index.values.tolist()
    df_temp['value'] = df.values.tolist()
    df_temp['error'] = result.importances_std
    df_temp = df_temp.sort_values('value', ascending=False)

    #fig = go.Figure(go.Bar(y=df, x=df.index.values.tolist(), error_y=dict(type='data', array=result.importances_std), marker_color=colors))
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
                                    args=[{"visible": [i in visible_features for i in df.index.values.tolist()]}],
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
                    ],
              ),     
            title=title,
            yaxis_title="Mean accuracy decrease")
    fig.show()
