import numpy as np
import pandas as pd
import tensorflow as tf
from ._filter import get_levels

def convert_to_numpy(col):
    """ Convert tensors, DataFrames and Sereis objects to numpy ndarray. 
    """
    if isinstance(col, pd.DataFrame):
        return col.to_numpy()
    elif isinstance(col, pd.Series):
        return col.to_numpy()
    elif tf.is_tensor(col):
        return col.numpy()
    elif isinstance(col, np.ndarray):
        return col
    else:
        raise Exception("could not convert {} to numpy".format(type(col)))
    
def df_from_array(X, column_names, y=None, label_name=None):
    """ Transform X and y (if supplied) to a DataFrame with either provided column names or indexes 
    corresponding to the column position. 
    Make sure column names are strings to avoid errors with indexation. 
    """
    if (label_name is None):
        label_name = str(len(column_names))
    if y is None: column_names = column_names + [label_name]    
    df = pd.DataFrame(X, columns = column_names)
    if (isinstance(X, (np.ndarray, np.generic, list))):
        if (y is not None) and (label_name not in df):
            df[label_name] = y
        df.columns = df.columns.astype(str)
        return df
    if isinstance(X, pd.DataFrame) and y is not None:
        X[label_name] = y
    return X

def check_corruptions(df, corruption_list):
    """ Helper method to simplify the corruption instructions. Returns the feature names as strings for 
    the dataframe to avoid indexation errors. 
    """
    for method in corruption_list:
        feature_names_string, levels = get_levels(method, df)
        for key, value in method.items():
            value = [feature_names_string, levels]
            method[key] = value
    return corruption_list

def fill_missing_columns(corrupted_df, X_train):
    """corrupted_df will only contain the feature columns that have been modified. Add the original columns
    from X_train for features that are unchanged.
    """
    for column_name in corrupted_df:
        if corrupted_df[column_name].isnull().all(): 
            corrupted_df[column_name] = X_train[column_name].values
    return corrupted_df
