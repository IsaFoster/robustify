import numpy as np
import pandas as pd
import plotly.graph_objects as go

def Binary_noise():
    pass

def Poisson_noise(clean_data, random_state=None):
    np.random.seed(random_state)
    noise = np.random.poisson(np.mean(clean_data), len(clean_data))
    #noise = np.random.poisson(clean_data, size=None)
    noisy_data = clean_data + noise 
    return noisy_data
