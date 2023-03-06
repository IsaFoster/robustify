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
        print(np.unique(clean_data[feature_name], return_counts=True))
        noise = np.random.poisson(lam=data_col, size=data_col.shape[0])
        print(np.unique(noise, return_counts=True))
        clean_data[feature_name] = data_col + noise
        print(np.unique(clean_data[feature_name], return_counts=True))
        return clean_data
    if (isinstance(clean_data, (np.ndarray, np.generic))):
        noise = np.random.poisson(np.mean(clean_data), len(clean_data))
        return clean_data + noise 
