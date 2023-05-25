
import numpy as np
import pandas as pd

def gaussian_noise(clean_data, percentage, feature_name=None, random_state=None):
    """
    Parameters:
    ----------
        clean_data: ndarray or dataframe of shape (n_samples, n_features)
            Array of the feature values that will be permuted. If dataframe,
            feature_name is required to determine which feature the noise should
            be applied to. .
        percentage: float
            Percentage of change in noise over signal that should be added (on
            average) when permuting the feature.
        feature_name: str, deafult=None
            If clean_data is a dataframe feature name is required to determine
            which feature will be permuted.
        random_state: int, RandomState instance or None, deafult=None
            Controls randomness in drawing from the distribution.
    Returns:  
    ----------    
        result: ndarray or dataframe of shape (n_samples, n_features)
            Array or dataframe (depends on input format) with the noisy feature
            permuted.
    """
    np.random.seed(random_state)
    if isinstance(clean_data, pd.DataFrame):
        data_col = clean_data[feature_name]
        noise = (data_col * percentage) * np.random.normal(0, 1, len(data_col))
        clean_data[feature_name] = data_col + noise
        return clean_data
    if isinstance(clean_data, (np.ndarray, np.generic, list)):
        noise = (clean_data * percentage) * np.random.normal(0, 1, len(clean_data))
        return clean_data + noise
    