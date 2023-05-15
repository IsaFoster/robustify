import numpy as np
import pandas as pd

def binomial_noise(clean_data, level, feature_name=None, random_state=None):
    """
    Parameters:
    ----------
    clean_data: ndarray or dataframe of shape (n_samples, n_features)
        Array of the feature values that will be permuted. If dataframe, feature_name is required to determine which feature the noise should be applied to. .
    level: float
        The probability of each value to be permuted. If 0, feature is returned without changes. If 1, every value is flipped.
    feature_name: str, deafult=None
        If clean_data is a dataframe feature name is required to determine which feature will be permuted.
    random_state: int, RandomState instance or None, deafult=None
        Controls randomness in drawing from the distribution.
    Returns:  
    ----------    
        result: ndarray or dataframe of shape (n_samples, n_features)
            Array or dataframe (depends on input format) with the noisy feature permuted.
    """
    np.random.seed(random_state)
    if (isinstance(clean_data, pd.DataFrame)):
        data_col = clean_data[feature_name]
        mask = np.random.binomial(1, level, data_col.shape[0]).astype(bool)
        clean_data[feature_name] = 1 - mask
        return clean_data
    if (isinstance(clean_data, (np.ndarray, np.generic))):
        mask = np.random.binomial(1, level, data_col.shape[0]).astype(bool)
        return 1 - mask 

def poisson_noise(clean_data, feature_name=None, random_state=None):
    """
    Parameters:
    ----------
        clean_data: ndarray or dataframe of shape (n_samples, n_features)
            Array of the feature values that will be permuted. If dataframe, feature_name is required to determine which feature the noise should be applied to. .
        level: float
            The probability of each value to be permuted. If 0, feature is returned without changes. If 1, every value is flipped.
        feature_name: str, deafult=None
            If clean_data is a dataframe feature name is required to determine which feature will be permuted.
        random_state: int, RandomState instance or None, deafult=None
            Controls randomness in drawing from the distribution.
    Returns:  
    ----------    
        result: ndarray or dataframe of shape (n_samples, n_features)
            Array or dataframe (depends on input format) with the noisy feature permuted.
    """
    np.random.seed(random_state)
    if (isinstance(clean_data, pd.DataFrame)):
        data_col = clean_data[feature_name]
        index_data = np.searchsorted(np.unique(data_col), data_col)
        noise = np.random.poisson(abs(np.mean(data_col)), size=data_col.shape)
        noisy_index = np.clip(index_data + noise, min(index_data), max(index_data))
        clean_data[feature_name] = np.take(np.unique(data_col), noisy_index)
        return clean_data
    if (isinstance(clean_data, (np.ndarray, np.generic))):
        index_data = np.searchsorted(np.unique(clean_data), clean_data)
        noise = np.random.poisson(abs(np.mean(clean_data)), size=clean_data.shape)
        noisy_index = np.clip(index_data + noise, min(index_data), max(index_data))
        return np.take(np.unique(clean_data), noisy_index)

