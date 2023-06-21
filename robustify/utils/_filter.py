from robustify.noise.continuous import gaussian_noise, gaussian_noise_dep
from robustify.noise.discrete import poisson_noise, binomial_noise
import tensorflow as tf
import keras

def filter_on_method(df, method, feature_name, dep, level=None, random_state=None):
    switcher = {
        'Binomial': lambda: binomial_noise(df, level, feature_name, random_state),
        'Gaussian': lambda: gaussian_noise(df, level, feature_name, random_state),
        'Gaussian_': lambda: gaussian_noise_dep(df, level, feature_name, dep, random_state),
        'Poisson': lambda: poisson_noise(df, feature_name, random_state)
    }
    return switcher.get(method, lambda: ValueError("Invalid corruption method for feature {}".format(feature_name)))()

def get_feature_name_from_index(feature_names, df):
    return [list(df)[i] for i in feature_names]

def get_levels(methodSpecification, df=None):
    optional_param = None
    method = list(methodSpecification.keys())[0]
    if (method == "Gaussian" or method == "Binomial"):
        feature_names, levels = list(methodSpecification.values())[0][0], list(methodSpecification.values())[0][1]
    elif (method == "Poisson"):
        if isinstance(list(methodSpecification.values())[0][0], (str, int)):
            feature_names, levels = list(methodSpecification.values())[0], [-1]
        else:
            feature_names, levels = list(methodSpecification.values())[0][0], [-1]
    elif (method == "Gaussian_"):
        feature_names, levels, optional_param = list(methodSpecification.values())[0][0], list(methodSpecification.values())[0][1], list(methodSpecification.values())[0][2]
    else:
        raise ValueError("Method {} not recognized".format(method))
    if all([isinstance(item, int) for item in feature_names]):
        feature_names = get_feature_name_from_index(feature_names, df)
    return feature_names, levels, optional_param

def is_keras_model(model):
    return isinstance(model, (tf.keras.Model, keras.Model, tf.estimator.Estimator))
    