from Noise.continuous import Gaussian_Noise
from Noise.discrete import Poisson_noise, Binomial_noise

def filter_on_method(df, method, feature_name, level=None, random_state=None):
    switcher = {
        'Binomial': lambda: Binomial_noise(df, level, feature_name, random_state),
        'Gaussian': lambda: Gaussian_Noise(df, level, feature_name, random_state),
        'Poisson': lambda: Poisson_noise(df, feature_name, random_state)
    }
    return switcher.get(method, lambda: print("Invalid corruption method for feature {}".format(feature_name)))()

def get_feature_name_from_index(feature_names, df):
    return [list(df)[i] for i in feature_names]

def getLevels(methodSpecification, df):
    method = list(methodSpecification.keys())[0]
    if (method == "Gaussian" or method == "Binomial"):
        feature_names, levels = list(methodSpecification.values())[0][0], list(methodSpecification.values())[0][1]
    elif (method == "Poisson"):
        if isinstance(list(methodSpecification.values())[0][0], (str, int)):
            feature_names, levels = list(methodSpecification.values())[0], [-1]
        else:
            feature_names, levels = list(methodSpecification.values())[0][0], [-1]
    if all([isinstance(item, int) for item in feature_names]):
        feature_names = get_feature_name_from_index(feature_names, df)
    return feature_names, levels
    