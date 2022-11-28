def sampleData(df, labelColumn, frac, random_state=None):
    df_sampled = df.sample(frac=frac, replace=True, random_state=random_state)
    y_sampled = df_sampled[[labelColumn]]
    X_sampled = df_sampled.drop([labelColumn], axis=1)
    return X_sampled, y_sampled