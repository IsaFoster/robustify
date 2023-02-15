import numpy as np
import pandas as pd
import plotly.graph_objects as go

def Binary_noise():
    pass

def Poisson_noise(clean_data, feature_name=None, random_state=None):
    np.random.seed(random_state)
    if (isinstance(clean_data, pd.DataFrame)):
        data_col = clean_data[feature_name]
        noise = np.random.poisson(np.mean(data_col), len(data_col))
        clean_data[feature_name] = data_col + noise
        return clean_data
    if (isinstance(clean_data, (np.ndarray, np.generic))):
        noise = np.random.poisson(np.mean(clean_data), len(clean_data))
        return clean_data + noise 
