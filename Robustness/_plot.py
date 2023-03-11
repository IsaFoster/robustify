import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

## TODO: move helper functions to wherever
def getLevels(methodSpecification):
    method = list(methodSpecification.keys())[0]
    if (method == "Gaussian" or method == "Binomial"):
        return list(methodSpecification.values())[0][0], list(methodSpecification.values())[0][1]
    elif (method == "Poisson"):
        return list(methodSpecification.values())[0], [-1]
    else:
        print('Error getting values')
        print(type(methodSpecification))

def topFromDfGrouped(df, group, groupValue, value, feature, num):
    df_grouped = df.groupby(group)
    df_group = df_grouped.get_group(int(groupValue))
    sorted = df_group.sort_values(value, ascending=False)
    return sorted[feature].tolist()[:num]

def topFromDf(df, value, feature, num):
    sorted = df.sort_values(value, ascending=False)
    return sorted[feature].iloc[:num].tolist()

def topFromSeries(df, num):
    df_temp = df.mean(axis=0)
    sorted = df_temp.sort_values(ascending=False)
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

def plotButtons(corruption_list, featureNames):
    button_layout = [dict(
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
    if corruption_list != None:
        for noise_method in corruption_list:
            visible_features, levels = getLevels(noise_method)
            print("visible_features:", visible_features)
            print("feature names:", featureNames)
            print("show values:", [{"visible": [i in visible_features for i in featureNames]}])
            button_values = button_layout[0]['buttons']
            button_values.append(dict(
                    args=[{"visible": [i in visible_features for i in featureNames]}],
                    label=str(list(noise_method.keys())[0]) + ' ' + str(levels).replace('[', ''),
                    method="restyle"))
            button_layout[0].update({'buttons': button_values})


    return button_layout

def plotNoiseCorruptionsAverageFeatureValue(df, model_name, measured, corruptions, lalal, level_start):
    visible_features = topFromDfGrouped(df, 'level', level_start, lalal, 'feature_name', 5)
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

def plotPermutationImportance(df_baseline, df_noisy, n_repeats, modelName):
    visible_features = topFromSeries(df_baseline, 5)
    title = "Permutation Importances using n={} for {}".format(n_repeats, modelName)
    fig = go.Figure(layout={"xaxis_title":"Feature", "yaxis_title":"Permutation importance", "font":dict(size=18)})
    for (columnName, columnData) in df_baseline.items():
        fig.add_trace(go.Box(x=columnData, boxmean=True, name=columnName))
    for (columnName, columnData) in df_noisy.items():
        fig.add_trace(go.Box(x=columnData, boxmean=True, name=columnName))
    fig.update_layout(dict(updatemenus=plotButtons(visible_features, df_baseline.columns),
                    legend={'traceorder': 'reversed'}
              ),
              title=title,
              xaxis_title="Decrease in accuracy score")
    return fig

def get_colors_from_fig(fig):
    colors = []
    f = fig.full_figure_for_development(warn=False)
    f.for_each_trace(lambda t: colors.append(t.marker.color))
    return colors

def hex_to_rgba(hex, alpha, factor=1):
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex[i:i+2], 16)
        decimal = int(decimal * factor)
        if (decimal > 255): decimal = 255
        if (decimal < 0): decimal = 0
        rgb.append(decimal)
    rgb.append(alpha)
    return tuple(rgb)


def get_plotly_xcoordinates(index, width=0.8):
    x_location = index
    return x_location - width/2, x_location + width/2


def plotMeanAccuracyDecrease(df, noisy_df, result, permutations, modelName, corruption_list):
    title = "Feature importances using n={} permutation on {}".format(permutations, modelName)
    df_temp = pd.DataFrame(columns=['feature_name', 'value', 'value_noisy', 'error'])
    df_temp['feature_name'] = df.index.values.tolist()
    df_temp['value'] = df.values.tolist()
    df_temp['value_noisy'] = noisy_df.values.tolist()
    df_temp['error'] = result.importances_std
    df_temp = df_temp.sort_values('value', ascending=False)
    df_temp = df_temp.reset_index()
    visible_features = corruption_list
    print("visible features", visible_features)
    fig = go.Figure()
    for (index, rowData) in df_temp.iterrows():
        fig.add_trace(
            go.Bar(
                x=[rowData['feature_name']],
                y=[rowData['value']],
                name=rowData['feature_name'],
                error_y={'type':'data', 'array':[rowData['error']]},
                legendgroup=index)
        )
    colors = get_colors_from_fig(fig)
    for (index, rowData) in df_temp.iterrows():
        if (rowData['value'] > rowData['value_noisy']):
            color = 'rgba' + str(hex_to_rgba(colors[index][1:], 0.5, 1.5))
        else:
            color = 'rgba' + str(hex_to_rgba(colors[index][1:], 0.5))
        fig.add_trace(
            go.Bar(
                x=[rowData['feature_name']],
                y=[rowData['value_noisy']],
                name=rowData['feature_name'],
                marker_color= color,
                opacity=0.8,
                legendgroup=index,
                showlegend=False)
            )
    fig.update_layout(dict(updatemenus=plotButtons(visible_features, df_temp['feature_name'].values.tolist()),
              ),     
            title=title,
            yaxis_title="Mean accuracy decrease",
            barmode = 'overlay')
    return fig


def plotNoiseCorruptionValues(baseline_results, corruption_result_list, model_name, corruptions, measured_property, method_name, measured_name, corruptions_list):
    all_features = []
    fig = go.Figure()
    title = "Average {} of {} over {} {} corruptions at increasing noise levels for {}".format(measured_name.replace("_", " "), measured_property, corruptions, method_name, model_name)
    for corruption_result in corruption_result_list:
        for feature in corruption_result['feature_name'].unique().tolist():
            all_features.append(feature)
            df = corruption_result.loc[corruption_result['feature_name'] == feature]
            fig.add_trace(go.Scatter(x=df["level"], y=df[measured_name], mode='lines', line=dict(width=4), name=feature, legendgroup=feature))
    fig.update_layout(dict(updatemenus=plotButtons(corruptions_list, all_features)), xaxis_title="Feature", yaxis_title=measured_property, title=title, font=dict(size=18))
    return fig


def plotNoiseCorruptionValuesHistogram(baseline_results, corruption_result_list, model_name, corruptions, measured_property, method_name, measured_name, corruptions_list):
    title = "Average {} of {} for features for {} over {} {} noise corruptions".format(measured_name.replace("_", " "), measured_property, model_name, corruptions, method_name)
    results = pd.DataFrame(columns=['feature_name', measured_name, measured_name +'_noisy'])
    baseline_results = baseline_results.sort_values("feature_name")
    for corruption_result in corruption_result_list:
        results_temp = pd.DataFrame(columns=['feature_name', measured_name, measured_name+'_noisy'])
        corruption_result = corruption_result.sort_values("feature_name")
        results_temp['feature_name'] = corruption_result['feature_name'].unique().tolist()
        results_temp['feature_name'] = results_temp['feature_name'].astype(str)
        results_temp[measured_name] = baseline_results[baseline_results['feature_name'].isin(results_temp['feature_name'].values.tolist())][measured_name].values.tolist()
        results_temp[measured_name+'_noisy'] = corruption_result[measured_name].values.tolist()
        results_temp = results_temp.sort_values(measured_name, ascending=False)
        results = pd.concat([results, results_temp], axis=0)
    results = results.reset_index()
    print("result:", results)
    #fig = go.Figure(layout={"title": title, "xaxis_title":"Feature", "yaxis_title":measured_property, "font":dict(size=18)})
    fig = go.Figure()
    for (index, rowData) in results.iterrows():
        fig.add_trace(
            go.Bar(x=[rowData["feature_name"]], 
                   y=[rowData[measured_name]], 
                   name=rowData['feature_name'], 
                   legendgroup=index)
                )
    colors = get_colors_from_fig(fig)
    for (index, rowData) in results.iterrows():
        fig.add_trace(
            go.Bar(x=[rowData["feature_name"]], 
                   y=[rowData[measured_name+'_noisy']], 
                   name=rowData['feature_name'],  
                   marker_color = 'rgba' + str(hex_to_rgba(colors[index][1:], 0.5, 1.2)), 
                   showlegend=False,
                   legendgroup=index)
                )
    fig.update_traces(width=0.4)
    fig.update_layout(dict(updatemenus=plotButtons(corruptions_list, results['feature_name'].values.tolist()),
              ),    
              title=title, 
              xaxis_title="Feature", 
              yaxis_title=measured_property,
              font=dict(size=18),
              bargroupgap=0,  
              barmode='group')
    return fig

def plotNoiseCorruptionBarScore(baseline_results, corruption_result, model_name, corruptions, measured_property, method_name, measured_name, corruption_list):
    features = np.unique(baseline_results['feature_name'].values.ravel())
    fig = go.Figure()
    fig.add_trace(go.Bar(y=np.unique(baseline_results['accuracy'].values.ravel()), name='baseline', marker_color='seagreen'))
    fig.add_trace(go.Bar(x=features, y=corruption_result['accuracy'], name='noisy', marker_color='maroon'))
    return fig
