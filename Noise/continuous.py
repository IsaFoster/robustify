import numpy as np

def Gaussian_Noise(clean_data, percentage, random_state=None):
    np.random.seed(random_state)
    noiseSigma = clean_data * percentage
    noise = noiseSigma * np.random.normal(0, 1, len(clean_data))
    return clean_data + noise

