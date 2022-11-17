def sampleData(df, frac):
    df_sampled = df.sample(frac=frac, replace=True)
    y_sampled = df_sampled[['data_type']]
    X_sampled = df_sampled.drop(['data_type'], axis=1)
    return X_sampled, y_sampled