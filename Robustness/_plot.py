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
    return sorted.index[:5]

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