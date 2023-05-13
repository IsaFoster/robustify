[![codecov](https://codecov.io/gh/IsaFoster/MasterThesis/branch/main/graph/badge.svg?token=9CWBWHNZML)](https://app.codecov.io/gh/IsaFoster/MasterThesis/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CodeQL](https://github.com/IsaFoster/MasterThesis/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/IsaFoster/MasterThesis/actions/workflows/github-code-scanning/codeql)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/django)](https://test.pypi.org/project/Robustness/)
[![PyPI](https://badge.fury.io/py/robustness.svg)](https://badge.fury.io/py/adversarial-robustness-toolbox)
[![Build](https://github.com/IsaFoster/MasterThesis/actions/workflows/python-package.yml/badge.svg)](https://github.com/IsaFoster/MasterThesis/actions/workflows/python-package.yml)


NAME is a python library that can add and evaluate the effects of feature specific noise in machine learning models for tabular data. There are several different tyoes of noise for both continous and discrete features.
 
<p align="center">
  <img src="url" width="800" />
</p>

## Install
NAME can be installed from either PyPI or conda-forge:
<pre>
pip install NAME
<i>or</i>
conda install -c conda-forge NAME
</pre>

## Usages
Immitate uncertainty
Immitate actual nosie
Avouid overfirtting 
Data augmentation
Expand data grunnlag 
## Compatibility
NAME is compatible with most machine learning models trained on tabular data from scikit-learn, pytorch, tesorflow/keras, and FastAi. 

## Feature importance measures 
When adding deliberate noise to a particaule feature, it is important to look at how that effects the feature's importance to a particular model's predictive abilities. The featuer importance can be calcualted in different ways, and several external libraries are available to do this. 
### Scikit-learn featire importance, coefficients, and permuattion importance:
Scikit-learn (also known as sklearn) is a Python library that provides a wide range of machine learning algorithms for tasks such as classification, regression, clustering, and dimensionality reduction. It also includes tools for model selection, data preprocessing, and data visualization. Many scikit-learn models have built-in properties that estimate feature importance, for exaple a LinearRegression model will have coefficients, while a DecisionTreeClassifier will have the feature importances attribute. 
### Eli5
ELI5 (Explain Like I'm Five) is a Python library that provides a simple and intuitive way to explain machine learning models. In this application, permuation importance by ELI5 is used. 
### Lime
Lime (Local Interpretable Model-Agnostic Explanations) is a library that helps explain the predictions of machine learning models. It works by creating locally faithful linear models that approximate the predictions of a black-box model, and then providing explanations based on the coefficients of those models. Lime is useful for understanding how specific features contribute to a model's predictions, and for identifying potential biases or limitations in the model.
### Shap
SHAP (SHapley Additive exPlanations) is another Python library for explaining machine learning models. It provides a unified framework for interpreting a wide range of model types. SHAP uses game-theoretic concepts to compute feature importance values for each input feature, and provides global and local explanations of model behavior. It is particularly useful for understanding how different features interact with each other to affect model predictions.



## Examples
### Keras
### Pytorch
### Scikit-learn
### Comparing different feature importance measures

## stuff>

License is MIT.


