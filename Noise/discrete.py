import numpy as np
import pandas as pd

def Binomial_noise(clean_data, level, feature_name=None, random_state=None):
    np.random.seed(random_state)
    if (isinstance(clean_data, pd.DataFrame)):
        data_col = clean_data[feature_name]
        mask = np.random.binomial(1, level, data_col.shape[0]).astype(bool)
        clean_data[feature_name] = 1 - mask
        return clean_data
    if (isinstance(clean_data, (np.ndarray, np.generic))):
        mask = np.random.binomial(1, level, data_col.shape[0]).astype(bool)
        return 1 - mask 

def Poisson_noise(clean_data, feature_name=None, random_state=None):
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

