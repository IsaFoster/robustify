from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from robustify.utils._transform import df_from_array, normalize_max_min, make_scaler

def setup():
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    label_name = "species"

    iris = datasets.load_iris()
    X_ndarray = iris.data
    y_ndarray = iris.target
    Xy_ndarray = np.c_[ X_ndarray, y_ndarray ]
    return column_names, label_name, X_ndarray, y_ndarray, Xy_ndarray

def test_ndarray():
    column_names, label_name, X_ndarray, y_ndarray, Xy_ndarray = setup()
    X, _ = df_from_array(Xy_ndarray, column_names, y=None, label_name=None)
    assert (list(X) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', "4"])

def test_ndarray_y():
    column_names, label_name, X_ndarray, y_ndarray, Xy_ndarray = setup()
    X, _ = df_from_array(X_ndarray, column_names, y=y_ndarray, label_name=None)
    assert (list(X) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', "4"])

def test_ndarray_y_and_label():
    column_names, label_name, X_ndarray, y_ndarray, Xy_ndarray = setup()
    X, _ = df_from_array(X_ndarray, column_names, y=y_ndarray, label_name=label_name)
    assert (list(X) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', "species"])

def test_ndarray_no_names():
    column_names, label_name, X_ndarray, y_ndarray, Xy_ndarray = setup()
    X, _ = df_from_array(Xy_ndarray, column_names=None, y=None, label_name=None)
    assert (list(X) == ['0', '1', '2', '3', "4"])

def test_df_y():
    column_names, label_name, X_ndarray, y_ndarray, Xy_ndarray = setup()
    X, _ = df_from_array(pd.DataFrame(X_ndarray, columns=column_names), y=y_ndarray, label_name=None)
    assert (list(X) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', "4"])

def test_df_y_and_label():
    column_names, label_name, X_ndarray, y_ndarray, Xy_ndarray = setup()
    X, _ = df_from_array(pd.DataFrame(X_ndarray, columns=column_names), y=y_ndarray, label_name=label_name)
    assert (list(X) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', "species"])
 
def test_df_with_y():
    column_names, label_name, X_ndarray, y_ndarray, Xy_ndarray = setup()
    X, _ = df_from_array(pd.DataFrame(Xy_ndarray, columns=column_names+[label_name]), y=None, label_name=None)
    assert (list(X) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', "species"])

def test_df_with_y_and_label():
    column_names, label_name, X_ndarray, y_ndarray, Xy_ndarray = setup()
    X, _ = df_from_array(pd.DataFrame(Xy_ndarray, columns=column_names+[label_name]), y=None, label_name=label_name)
    assert (list(X) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', "species"])

def test_df_no_names():
    column_names, label_name, X_ndarray, y_ndarray, Xy_ndarray = setup()
    X, _ = df_from_array(pd.DataFrame(Xy_ndarray), y=None, label_name=None)
    assert (list(X) == ['0', '1', '2', '3', '4'])

def test_normalize():
    column_names, _, X_ndarray, _, _ = setup()
    df = pd.DataFrame(X_ndarray, columns=column_names)
    scaler = make_scaler(df[["sepal_length"]])
    normalized_col = normalize_max_min(df[["sepal_length"]], scaler)
    assert (len(normalized_col) == len(df["sepal_length"]))
    assert (normalized_col.values.max() == 1)
    assert (normalized_col.values.min() == 0)


