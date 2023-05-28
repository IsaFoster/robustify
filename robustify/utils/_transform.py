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
    
def df_from_array(X, column_names=None, y=None, label_name=None):
    """ Transform X and y (if supplied) to a DataFrame with either provided column names or indexes 
    corresponding to the column position. 
    Make sure column names are strings to avoid errors with indexation. 
    """
    if isinstance(X, (np.ndarray, np.generic, list)):
        df, label_name = df_from_ndarray(X, column_names, y, label_name)
    elif isinstance(X, pd.DataFrame):
        df, label_name = df_from_df(X, y, label_name)
    return df, label_name

def get_label_name(length, label_name):
    if label_name is None:
        label_name = str(length)
    return label_name

def df_from_ndarray(X, column_names, y, label_name):
    """ Transform X and y (if supplied) to a DataFrame from ndarray. Check allowed parametet 
    compatibility
    """
    if y is not None and label_name is not None:
        df = pd.DataFrame(X, columns = column_names)
        df[label_name] = y
    elif column_names is not None and y is not None:
        label_name = get_label_name(len(column_names), label_name)
        df = pd.DataFrame(X, columns = column_names)
        df[label_name] = y
    elif column_names is not None:
        if len(column_names) == X.shape[1]:
            df = pd.DataFrame(X, columns = column_names)
        elif len(column_names) == (X.shape[1]+1):
            df = pd.DataFrame(X, columns= column_names[:-1])
        else:
            label_name = get_label_name(len(column_names), label_name)
            df = pd.DataFrame(X, columns= column_names+[label_name])
    elif y is not None:
        df = pd.DataFrame(X)
        df[get_label_name(df.shape[1], label_name)] = y
    else:
        df = pd.DataFrame(X)
    df.columns = df.columns.astype(str)
    return df, label_name

def df_from_df(df, y, label_name):
    """ Transform X and y (if supplied) to a DataFrame from DataFrame. Check allowed parametet 
    compatibility
    """
    if  y is not None and label_name is not None:
        df[label_name] = y
    elif y is not None:
        label_name = str(df.shape[1])
        df[label_name] = y
    df.columns = df.columns.astype(str)
    return df, label_name

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

def normalize_max_min(column):
    return (column-column.min())/(column.max()-column.min())
