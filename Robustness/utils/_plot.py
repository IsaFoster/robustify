import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from Noise._filter import getLevels

def plotData(baseline_results, corruption_result_list, model_name, corruptions, measured_property, method_name, corruption_list):
    histogram_plot = []
    histogram_list = []
    line_plot = []
    line_list = []
    for corruption_result, corruption_type in zip(corruption_result_list, corruption_list):
        if (len(np.unique(corruption_result['level'].values)) < 3):
            histogram_plot.append(corruption_result)
            histogram_list.append(corruption_type)
        else:
            line_plot.append(corruption_result)
            line_list.append(corruption_type)
    if (len(histogram_plot) > 0):
        plotNoiseCorruptionValuesHistogram(baseline_results, histogram_plot, model_name, corruptions, measured_property, 'value', histogram_list)
        plotNoiseCorruptionValuesHistogram(baseline_results, histogram_plot, model_name, corruptions, measured_property, 'variance', histogram_list)
        plotNoiseCorruptionScoresHistogram(baseline_results, histogram_plot, model_name, corruptions, measured_property, 'score', histogram_list)
    if (len(line_plot) > 0):
        plotNoiseCorruptionValues(baseline_results, line_plot, model_name, corruptions, measured_property, 'value', line_list)
        plotNoiseCorruptionValues(baseline_results, line_plot, model_name, corruptions, measured_property, 'variance', line_list)
        plotNoiseCorruptionValues(baseline_results, line_plot, model_name, corruptions, measured_property,'score', line_list)      
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
        active=len(corruption_list)+1,
        pad={"r": 10, "t": 10},
        showactive=True,
        x=1,
        xanchor="right",
        y=1.1,
        yanchor="top"
        ),
    ]
    if corruption_list != None:
        for noise_method in corruption_list:
            visible_features, levels = getLevels(noise_method)
            button_values = button_layout[0]['buttons']
            button_values.append(dict(
                    args=[{"visible": [i in visible_features for i in featureNames]}],
                    label=str(list(noise_method.keys())[0]) + ' ' + str(levels).replace('[', ''),
                    method="restyle"))
            button_layout[0].update({'buttons': button_values})
    return button_layout, visible_features


## ???????????
def plotPermutationImportance(df_baseline, df_noisy, n_repeats, modelName):
    visible_features = [] # TODO: see if used
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
              xaxis_title="Decrease in score")
    return fig

## ????? 
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
    buttons, _ = plotButtons(corruption_list, df_temp['feature_name'].values.tolist())
    fig.update_layout(dict(updatemenus=buttons),     
            title=title,
            yaxis_title="Mean score decrease",
            font=dict(size=30),
            barmode = 'overlay')
    return fig

def plotNoiseCorruptionValues(baseline_results, corruption_result_list, model_name, corruptions, measured_property, measured_name, corruptions_list):
    all_features = []
    fig = go.Figure()
    title = "Average {} of {} over {} {} corruptions at increasing noise levels".format(measured_name.replace("_", " "), measured_property, corruptions, model_name)
    for corruption_result in corruption_result_list:
        for feature in corruption_result['feature_name'].unique().tolist():
            all_features.append(feature)
            df = corruption_result.loc[corruption_result['feature_name'] == feature]
            fig.add_trace(go.Scatter(x=df["level"], y=df[measured_name], mode='lines', line=dict(width=4), name=feature, legendgroup=feature))
    buttons, visible_features = plotButtons(corruptions_list, all_features)
    fig.update_layout(dict(updatemenus=buttons), xaxis_title="Feature", yaxis_title=measured_property, title=title, font=dict(size=18))
    if (measured_name == "score"):
        fig.add_hline(y=baseline_results[measured_name].iloc[0], line_dash="dash", line_width=4)
    else:
        fig.update_traces(visible=False, selector=lambda t: not t.name in visible_features)
    fig.show()

def plotNoiseCorruptionValuesHistogram(baseline_results, corruption_result_list, model_name, corruptions, measured_property, measured_name, corruptions_list):
    title = "Average {} of {} for features for {} over {} noise corruptions".format(measured_name.replace("_", " "), measured_property, model_name, corruptions)
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
    fig = go.Figure()
    for (index, rowData) in results.iterrows():
        fig.add_trace(
            go.Bar(x=[rowData["feature_name"]], 
                   y=[rowData[measured_name]], 
                   name=rowData['feature_name'], 
                   legendgroup=index,
                   width=0.4)
                )
    colors = get_colors_from_fig(fig)
    for (index, rowData) in results.iterrows():
        fig.add_trace(
            go.Bar(x=[rowData["feature_name"]], 
                   y=[rowData[measured_name+'_noisy']], 
                   name=rowData['feature_name'],  
                   marker_color = 'rgba' + str(hex_to_rgba(colors[index][1:], 0.5, 1.2)), 
                   showlegend=False,
                   legendgroup=index,
                   width=0.4)
                )
    buttons, visible_features = plotButtons(corruptions_list, results['feature_name'].values.tolist())
    fig.update_layout(dict(updatemenus=buttons,
              ),   
              title=title, 
              xaxis_title="Feature", 
              yaxis_title=measured_property,
              font=dict(size=18),
              bargroupgap=0,  
              barmode='group')
    fig.update_traces(visible=False, selector=lambda t: not t.name in visible_features)
    fig.show()

def plotNoiseCorruptionScoresHistogram(baseline_results, corruption_result_list, model_name, corruptions, measured_property, measured_name, corruptions_list):
    title = "Average {} for {} over {} noise corruptions".format(measured_name.replace("_", " "), model_name, corruptions)
    results = pd.DataFrame(columns=['feature_name', measured_name])
    baseline_results = baseline_results.sort_values("feature_name")
    for corruption_result in corruption_result_list:
        results_temp = pd.DataFrame(columns=['feature_name', measured_name])
        corruption_result = corruption_result.sort_values("feature_name")
        results_temp['feature_name'] = corruption_result['feature_name'].unique().tolist()
        results_temp['feature_name'] = results_temp['feature_name'].astype(str)
        results_temp[measured_name] = corruption_result[measured_name].values.tolist()
        results_temp = results_temp.sort_values(measured_name, ascending=False)
        results = pd.concat([results, results_temp], axis=0)
    results = results.reset_index(drop=True)
    data = []
    data.insert(0, {"feature_name": "baseline", measured_name: baseline_results[measured_name].iloc[0]})
    results = pd.concat([pd.DataFrame(data), results], ignore_index=True)
    fig = go.Figure()
    for (index, rowData) in results.iterrows():
        fig.add_trace(
            go.Bar(x=[rowData["feature_name"]], 
                   y=[rowData[measured_name]], 
                   name=rowData['feature_name'], 
                   legendgroup=index)
                )
    fig.add_hline(y=baseline_results[measured_name].iloc[0], line_dash="dash", line_width=4)
    score_diff = results[measured_name].max() - results[measured_name].min()
    fig.update_layout(dict(updatemenus=[dict(
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
            showactive=True,
            x=1,
            xanchor="right",
            y=1.1,
            yanchor="top"),
            ],
            ),   
            title=title, 
            xaxis_title="Feature", 
            yaxis_title=measured_name,
            font=dict(size=18),
            yaxis_range=[results[measured_name].min() - score_diff, results[measured_name].max() + score_diff])
    fig.show()