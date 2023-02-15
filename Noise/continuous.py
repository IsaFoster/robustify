import numpy as np
import pandas as pd

def _type_check(clean_data, ):
    return 

def Gaussian_Noise(clean_data, percentage, feature_name=None, random_state=None):
    np.random.seed(random_state)
    if (isinstance(clean_data, pd.DataFrame)):
        data_col = clean_data[feature_name]
        noise = (data_col * percentage) * np.random.normal(0, 1, len(data_col))
        clean_data[feature_name] = data_col + noise
        return clean_data
    if (isinstance(clean_data, (np.ndarray, np.generic, list))):
        noise = (clean_data * percentage) * np.random.normal(0, 1, len(clean_data))
        return clean_data + noise
    