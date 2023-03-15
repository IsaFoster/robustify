from Robustness.noiseCorruptions import corruptData
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# features ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
diabetes = datasets.load_diabetes()
X_regression = diabetes.data
y_regression = diabetes.target
X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X_regression, y_regression, test_size=0.3, random_state=0)
corruption_list_regression = [ 
    {'Poisson': [[0]]},
    {'Binomial': [[1], [0.1]]},
    {'Gaussian': [[2, 3, 4, 5, 6, 7, 8, 9], [0.2]]}]

# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
iris = datasets.load_iris()
X_classification = iris.data
y_classification = iris.target
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X_classification, y_classification, test_size=0.3, random_state=0)
corruption_list_classification = [ 
    {'Gaussian': [[0, 2], [0.2]]},
    {'Gaussian': [[1, 3], [0.3]]}]


def test_linear_model_regression():
    model = linear_model.LinearRegression()
    corrupted_df, corruption_result = corruptData(X_train_regression, 
                                          X_test_regression, y_test_regression, 
                                          model, 
                                          "r2",
                                          corruption_list_regression, 
                                          10, 
                                          y_train=y_train_regression,
                                          random_state=10, 
                                          plot=False)
    assert (corrupted_df is not None)
    assert (corruption_result is not None)
    assert (corruption_result.isnull().values.any() == False)

def test_linear_model_classification():
    model = linear_model.RidgeClassifier()
    corrupted_df, corruption_result = corruptData(X_train_classification, 
                                                  X_test_classification, 
                                                  y_test_classification, 
                                                  model, 
                                                  'accuracy',
                                                  corruption_list_classification, 
                                                  corruptions=10,
                                                  y_train=y_train_classification, 
                                                  label_name='species', 
                                                  column_names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], 
                                                  random_state=10, 
                                                  plot=False)
    assert (corrupted_df is not None)
    assert (corruption_result is not None)
    assert (corruption_result.isnull().values.any() == False)
