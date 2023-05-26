import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ._filter import get_levels

def plot_data(baseline_results, corruption_results, model_name, corruptions,
              measured_property, corruption_list):
    df_plot_line = df_plot_bar =pd.DataFrame(columns=['feature_name', 'level',
                                                      'value', 'variance', 'score'])
    bar_list = []
    line_list = []
    for corruption in corruption_list:
        features, levels = get_levels(corruption)
        if len(levels) > 1:
            line_list.append(corruption)
            df_plot_line = pd.concat([df_plot_line,
                                      corruption_results[
                                          corruption_results['feature_name'].isin(features)]])
        else:
            bar_list.append(corruption)
            df_plot_bar = pd.concat([df_plot_bar,
                                     corruption_results[
                                         corruption_results['feature_name'].isin(features)]])
    if len(line_list) > 0:
        fig_1 = plot_corruption_values(baseline_results, df_plot_line, model_name, corruptions,
                               measured_property, 'value', line_list)
        fig_2 = plot_corruption_values(baseline_results, df_plot_line, model_name, corruptions,
                               measured_property, 'variance', line_list)
        fig_3 = plot_corruption_values(baseline_results, df_plot_line, model_name, corruptions,
                               measured_property,'score', line_list)
    if len(bar_list) > 0:
        fig_1 = plot_corruption_values_hist(baseline_results, df_plot_bar, model_name, corruptions,
                                    measured_property, 'value', bar_list)
        fig_2 = plot_corruption_values_hist(baseline_results, df_plot_bar, model_name, corruptions,
                                    measured_property, 'variance', bar_list)
        fig_3 = plot_corruption_scores_hist(baseline_results, df_plot_bar, model_name, corruptions, 'score')
    return fig_1, fig_2, fig_3

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
        decimal = min(decimal, 255)
        decimal = max(decimal, 0)
        rgb.append(decimal)
    rgb.append(alpha)
    return tuple(rgb)

def plot_buttons(corruption_list, featureNames):
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
            visible_features, levels = get_levels(noise_method)
            button_values = button_layout[0]['buttons']
            button_values.append(dict(
                    args=[{"visible": [i in visible_features for i in featureNames]}],
                    label=str(list(noise_method.keys())[0]) + ' ' + str(levels).replace('[', ''),
                    method="restyle"))
            button_layout[0].update({'buttons': button_values})
    return button_layout, visible_features

def sort_df_by_list(df, feature, order):
    df[feature] = df[feature].astype("category")
    df[feature] = df[feature].cat.set_categories(order)
    df = df.sort_values([feature])
    return df

def plot_corruption_values(baseline_results, corruption_results, model_name, corruptions,
                           measured_property, measured_name, corruptions_list):
    title = "Average {} of {} over {} {} corruptions at increasing noise levels".format(
        measured_name.replace("_", " "), measured_property, corruptions, model_name)
    fig = px.line(corruption_results, x="level", y=measured_name, color='feature_name')
    buttons, visible_features = plot_buttons(corruptions_list,
                                             corruption_results['feature_name'].unique().tolist())
    fig.update_layout(dict(updatemenus=buttons), xaxis_title="Feature",
                      yaxis_title=measured_property, title=title, font=dict(size=18))
    if measured_name == "score":
        fig.add_hline(y=baseline_results[measured_name].iloc[0], line_dash="dash", line_width=4)
    else:
        fig.update_traces(visible=False, selector=lambda t: not t.name in visible_features)
    return fig

def plot_corruption_values_hist(baseline_results, corruption_results, model_name,
                                corruptions, measured_property, measured_name, corruptions_list):
    title = "Average {} of {} for features for {} over {} noise corruptions".format(
        measured_name.replace("_", " "), measured_property, model_name, corruptions)
    results = pd.DataFrame(columns=['feature_name', measured_name, measured_name +'_noisy'])
    order = corruption_results['feature_name'].unique().tolist()
    results['feature_name'] = corruption_results.sort_values(
        "feature_name")['feature_name'].unique().tolist()
    results[measured_name] = baseline_results.sort_values(
        "feature_name")[baseline_results['feature_name'].isin(
        results['feature_name'].values.tolist())][measured_name].values.tolist()
    results[measured_name+'_noisy'] = corruption_results.sort_values(
        "feature_name")[measured_name].values.tolist() 
    results = sort_df_by_list(results, "feature_name", order)
    fig = go.Figure()
    for (index, row_data) in results.iterrows():
        fig.add_trace(
            go.Bar(x=[row_data["feature_name"]],
                   y=[row_data[measured_name]],
                   name=row_data['feature_name'],
                   legendgroup=index,
                   width=0.4)
                )
    colors = get_colors_from_fig(fig)
    for (index, row_data) in results.iterrows():
        fig.add_trace(
            go.Bar(x=[row_data["feature_name"]],
                   y=[row_data[measured_name+'_noisy']],
                   name=row_data['feature_name'],
                   marker_color = 'rgba' + str(hex_to_rgba(colors[order.index(
                    row_data["feature_name"])][1:], 0.5, 1.2)),
                   showlegend=False,
                   legendgroup=index,
                   width=0.4)
                )
    buttons, visible_features = plot_buttons(corruptions_list,
                                             results['feature_name'].values.tolist())
    fig.update_layout(dict(updatemenus=buttons,),
              title=title,
              xaxis_title="Feature",
              yaxis_title=measured_property,
              font={"size":18},
              bargroupgap=0,
              barmode='group')
    fig.update_traces(visible=False, selector=lambda t: not t.name in visible_features)
    return fig

def plot_corruption_scores_hist(baseline_results, corruption_results, model_name,
                                corruptions, measured_name):
    title = "Average {} for {} over {} noise corruptions".format(
        measured_name.replace("_", " "), model_name, corruptions)
    results = pd.DataFrame(columns=['feature_name', measured_name])
    baseline_results = baseline_results.sort_values("feature_name")
    results['feature_name'] = corruption_results['feature_name'].unique().tolist()
    results[measured_name] = corruption_results[measured_name].values.tolist()
    data = []
    data.insert(0, {"feature_name": "baseline", measured_name:
                    baseline_results[measured_name].iloc[0]})
    results = pd.concat([pd.DataFrame(data), results], ignore_index=True)
    fig = go.Figure()
    for (index, row_data) in results.iterrows():
        fig.add_trace(
            go.Bar(x=[row_data["feature_name"]],
                   y=[row_data[measured_name]],
                   name=row_data['feature_name'],
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
            yanchor="top"),],),
            title=title,
            xaxis_title="Feature",
            yaxis_title=measured_name,
            font={"size": 18},
            yaxis_range=[results[measured_name].min() - score_diff,
                         results[measured_name].max() + score_diff])
    return fig
