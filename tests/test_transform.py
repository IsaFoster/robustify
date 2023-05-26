from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from robustify.utils._transform import df_from_array 

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
label_name = "species"

iris = datasets.load_iris()
X_ndarray = iris.data
y_ndarray = iris.target
Xy_ndarray = np.c_[ X_ndarray, y_ndarray ]

X_df_without_label = df = pd.DataFrame(X_ndarray, columns=column_names)
Xy_df = X_df_without_label.copy()
Xy_df[label_name] = y_ndarray

def test_ndarray():
    X = df_from_array(Xy_ndarray, column_names, y=None, label_name=None)
    assert (list(X) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', "4"])

def test_ndarray_y():
    X = df_from_array(X_ndarray, column_names, y=y_ndarray, label_name=None)
    assert (list(X) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', "4"])

def test_ndarray_y_and_label():
    X = df_from_array(X_ndarray, column_names, y=y_ndarray, label_name=label_name)
    assert (list(X) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', "species"])

def test_df_y():
    X = df_from_array(X_df_without_label, column_names, y=y_ndarray, label_name=None)
    assert (list(X) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', "4"])

def test_df_y_and_label():
    X = df_from_array(X_df_without_label, column_names, y=y_ndarray, label_name=label_name)
    assert (list(X) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', "species"])

# column_names should have label name 
def test_df_with_y():
    X = df_from_array(Xy_df, column_names, y=None, label_name=None)
    assert (list(X) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', "species"])

def test_df_with_y_and_label():
    X = df_from_array(Xy_df, column_names, y=None, label_name=label_name)
    assert (list(X) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', "species"])
