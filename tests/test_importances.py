from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from robustify.utils._importances import filter_on_importance_method 

iris = datasets.load_iris()
X_classification = iris.data
y_classification = iris.target
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X_classification, y_classification, test_size=0.3, random_state=0)

model = RandomForestClassifier()
model.fit(X_train_classification, y_train_classification)

corruption_list_classification = [ 
    {'Gaussian': [[0, 2], [0.2]]},
    {'Gaussian': [[1, 3], [0.3]]}]

def test_existing_measure():
    importance, measured_property = filter_on_importance_method(model, 0, X_train_classification, y_train_classification, X_test_classification, y_test_classification, 10, "accuracy", "shap", None)
    assert (measured_property == "shap values")
    assert (isinstance(importance, float))

def test_measure_can_be_capital():
    importance, measured_property = filter_on_importance_method(model, 1, X_train_classification, y_train_classification, X_test_classification, y_test_classification,10, "accuracy", "SHAP", None)
    assert (measured_property == "shap values")
    assert (isinstance(importance, float))

def test_non_existing_measure():
    importance = filter_on_importance_method(model, 1, X_train_classification, y_train_classification, X_test_classification, y_test_classification, 10, "accuracy", "IMAGINARY", None)
    assert (importance is None)