def sampleData(df, frac, random_state=None):
    df_sampled = df.sample(frac=frac, replace=True, random_state=random_state)
    y_sampled = df_sampled[['data_type']]
    X_sampled = df_sampled.drop(['data_type'], axis=1)
    return X_sampled, y_sampled