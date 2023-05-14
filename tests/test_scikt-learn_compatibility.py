from Robustness.noiseCorruptions import corruptData
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, svm, neighbors, gaussian_process, tree, neural_network

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

def run_corruption_classification(model):
    return corruptData(model,
                       corruption_list_classification,
                       X_train_classification, 
                       X_test_classification, 
                       'accuracy',
                       y_train_classification, 
                       y_test_classification,  
                       column_names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], 
                       label_name='species', 
                       corruptions=10,
                       random_state=10, 
                       plot=False)

def run_corruption_regression(model):
    return corruptData(model,
                       corruption_list_regression, 
                       X_train_regression, 
                       X_test_regression, 
                       "r2",
                       y_train_regression,
                       y_test_regression,
                       corruptions=10,
                       random_state=10,
                       plot=False)

def test_linear_model_regression():
    model = linear_model.LinearRegression()
    result = run_corruption_regression(model)
    assert (result.corrupted_df is not None)
    assert (result.corruption_result is not None)
    assert (result.corruption_result.isnull().values.any() == False)

def test_linear_model_classification():
    model = linear_model.RidgeClassifier()
    result = run_corruption_classification(model)
    assert (result.corrupted_df is not None)
    assert (result.corruption_result is not None)
    assert (result.corruption_result.isnull().values.any() == False)

def test_SVM_model_regression():
    model = svm.SVR()
    result = run_corruption_regression(model)
    assert (result.corrupted_df is not None)
    assert (result.corruption_result is not None)
    assert (result.corruption_result.isnull().values.any() == False)

def test_SVM_model_classification():
    model = svm.SVC(decision_function_shape='ovo')
    result = run_corruption_classification(model)
    assert (result.corrupted_df is not None)
    assert (result.corruption_result is not None)
    assert (result.corruption_result.isnull().values.any() == False)

def test_KNN_model_regression():
    model = neighbors.KNeighborsRegressor()
    result = run_corruption_regression(model)
    assert (result.corrupted_df is not None)
    assert (result.corruption_result is not None)
    assert (result.corruption_result.isnull().values.any() == False)

def test_KNN_model_classification():
    model = neighbors.KNeighborsClassifier()
    result = run_corruption_classification(model)
    assert (result.corrupted_df is not None)
    assert (result.corruption_result is not None)
    assert (result.corruption_result.isnull().values.any() == False)

def test_GP_model_regression():
    model = gaussian_process.GaussianProcessRegressor()
    result = run_corruption_regression(model)
    assert (result.corrupted_df is not None)
    assert (result.corruption_result is not None)
    assert (result.corruption_result.isnull().values.any() == False)

def test_GP_model_classification():
    model = gaussian_process.GaussianProcessClassifier()
    result = run_corruption_classification(model)
    assert (result.corrupted_df is not None)
    assert (result.corruption_result is not None)
    assert (result.corruption_result.isnull().values.any() == False)

def test_DT_model_regression():
    model = tree.DecisionTreeRegressor()
    result = run_corruption_regression(model)
    assert (result.corrupted_df is not None)
    assert (result.corruption_result is not None)
    assert (result.corruption_result.isnull().values.any() == False)

def test_DT_model_classification():
    model = tree.DecisionTreeClassifier()
    result = run_corruption_classification(model)
    assert (result.corrupted_df is not None)
    assert (result.corruption_result is not None)
    assert (result.corruption_result.isnull().values.any() == False)

def test_NN_model_regression():
    model = neural_network.MLPRegressor()
    result = run_corruption_regression(model)
    assert (result.corrupted_df is not None)
    assert (result.corruption_result is not None)
    assert (result.corruption_result.isnull().values.any() == False)

def test_NN_model_classification():
    model = neural_network.MLPClassifier()
    result = run_corruption_classification(model)
    assert (result.corrupted_df is not None)
    assert (result.corruption_result is not None)
    assert (result.corruption_result.isnull().values.any() == False)



