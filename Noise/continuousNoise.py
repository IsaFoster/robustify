import numpy as np

def Gaussian_Noise(clean_cata, percentage, random_state=None):
    np.random.seed(random_state)
    sd = np.std(clean_cata)
    noise = np.random.normal(0, sd, len(clean_cata))
    corrupted_data = clean_cata + (noise * (1 + percentage))
    return corrupted_data

def Binary():
    pass

def Poisson():
    pass
