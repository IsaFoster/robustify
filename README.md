[![codecov](https://codecov.io/gh/IsaFoster/MasterThesis/branch/main/graph/badge.svg?token=9CWBWHNZML)](https://app.codecov.io/gh/IsaFoster/MasterThesis/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CodeQL](https://github.com/IsaFoster/robustify/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/IsaFoster/robustify/actions/workflows/codeql-analysis.yml)
[![Build](https://github.com/IsaFoster/MasterThesis/actions/workflows/python-package.yml/badge.svg)](https://github.com/IsaFoster/MasterThesis/actions/workflows/python-package.yml)


Welcome to Robustify, a GitHub repository focused on evaluating the effects of adding structurally conserving noise to data. The goal is to provide a comprehensive set of tools for researchers and practitioners interested in exploring the impact of noise on the score and robustness of their machine learning models. The repository includes a variety of noise generation and augmentation techniques, as well as methods for evaluating the effects of noise on model performance, robustness metrics and visualizations. 

<p align="center">
  <img src="https://github.com/IsaFoster/robustify/blob/ed05ecac1f5eb39c4858b292eccf63f077454e1a/docs/images/robustify.png" width="800" />
</p>

## Install
Robustify can be installed from either PyPI or conda-forge:
<pre>
pip install robustify
<i>or</i>
conda install -c conda-forge robustify
</pre>

## Usages
### Simulating uncertainty:
Adding noise to data is a widely used technique in machine learning with various benefits. One important use case is simulating uncertainty in data. By introducing random noise into the training data, machine learning models can learn to be more robust and effective in real-world scenarios where the input data may be noisy or the values uncertain.
### Robustness: 
Another key benefit of adding noise to data is improved robustness. Machine learning models trained on noisy data can learn to be more resilient to adversarial attacks or naturally occurring perturbations. Moreover, adding noise to data can help mitigate the effect of outliers in the training data and can improve the model's overall performance.
### Generalization: 
Adding noise to data can also help improve the generalization performance of machine learning models. Overfitting is a common problem in machine learning, where the model becomes too specialized to the training data and performs poorly on new, unseen data. By introducing noise to the training data, the model is forced to learn more robust and generalizable features, resulting in better performance on test data.
### Data augmentation:
Data augmentation is a technique used in machine learning to increase the size and diversity of a dataset by creating new examples from existing ones. Adding noise to data can be an effective form of data augmentation. By adding noise to the data, the model is exposed to more variations of the input data, which can improve its ability to recognize patterns and make accurate predictions on new, unseen data. Additionally, data augmentation can reduce the risk of overfitting and improve the model's generalization performance.

## Compatibility 
Robustify is compatible with most machine learning models trained on single-output tabular data from Scikit-learn, PyTorch, TensorFlow/Keras, and FastAI.

## Feature importance measures 
Feature importance is an important concept in machine learning that allows us to understand which features are most influential in making predictions. Adding deliberate noise to a particular feature can affect the feature's importance to a particular model's predictive abilities. Several external libraries are available, and different methods can be used to determine feature importance.
### Scikit-learn feature importance, coefficients, and permutation importance:
[Scikit-learn][2] is a popular Python library for machine learning that provides several methods for calculating feature importance. One of the simplest methods is to use the feature_importances_ attribute of decision tree-based models, such as Random Forest and Gradient Boosting. This attribute calculates the importance of each feature by measuring the reduction in impurity that results from splitting on that feature. Additionally, linear models can use the absolute value of the coefficients as a measure of feature importance. Scikit-learn provides the coef_ attribute for this purpose. Another approach is to use permutation-based feature importance, which involves randomly permuting the values of each feature and measuring the resulting decrease in model performance. Scikit-learn provides the [permutation importance][3] function to implement this method.
### Eli5
[ELI5 (Explain Like I'm Five)][4] is a Python library that provides a simple and intuitive way to explain machine learning models. In this application, permutation importance by ELI5 is used as an alternative to scikit-learn's permutation importance, that is not compatible with all types of models. 
### Lime
[Lime (Local Interpretable Model-Agnostic Explanations)][5] is a library that helps explain the predictions of machine learning models. It works by creating locally faithful linear models that approximate the predictions of a black-box model, and then providing explanations based on the coefficients of those models. Lime is useful for understanding how specific features contribute to a model's predictions, and for identifying potential biases or limitations in the model.
### Shap
[SHAP (SHapley Additive exPlanations)][6] is another Python library for explaining machine learning models. It provides a unified framework for interpreting a wide range of model types. SHAP uses game-theoretic concepts to compute feature importance values for each input feature and provides global and local explanations of model behaviour. It is particularly useful for understanding how different features interact with each other to affect model predictions.

## Examples
### Keras
### Pytorch
### Scikit-learn

## Documentation
See the [Wiki][1] for documentation of the availabel methods. 

License is MIT.

[1]: https://github.com/IsaFoster/robustify/wiki
[2]: https://scikit-learn.org/stable/index.html
[3]: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
[4]: https://github.com/TeamHG-Memex/eli5
[5]: https://github.com/marcotcr/lime
[6]: https://github.com/slundberg/shap


