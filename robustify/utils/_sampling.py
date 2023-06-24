def sample_data(df, labelColumn, frac, random_state=None):
    if frac == 1:
        y = df[[labelColumn]].squeeze()
        X = df.drop([labelColumn], axis=1)
        return X, y
    df_sampled = df.sample(frac=frac, replace=True, random_state=random_state, ignore_index=True)
    y_sampled = df_sampled[[labelColumn]].squeeze()
    X_sampled = df_sampled.drop([labelColumn], axis=1)
    return X_sampled, y_sampled

def sample_X(X, frac, random_state=None):
    if frac == 1:
        return X
    X_sampled = X.sample(frac=frac, replace=True, random_state=random_state, ignore_index=True)
    return X_sampled