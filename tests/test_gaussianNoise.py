from Noise import gaussianNoise
import numpy as np
import pandas as pd

test_data = np.array([100,60,130,122,99,87,103, 101,65,110,72,89,82,117,111,70,140,109,79,67,113,119,66,132,112,98,89,104,101,80,120,124,98, 100,60,130,122,99,87,103, 101,65,110,72,89,82,117,111,70,140,109,79,67,113,119,66,132,112,98,89,104,101,80,120,124,98, 100,60,130,122,99,87,103, 101,65,110,72,89,82,117,111,70,140,109,79,67,113,119,66,132,112,98,89,104,101,80,120,124,98, 100,60,130,122,99,87,103, 101,65,110,72,89,82,117,111,70,140,109,79,67,113,119,66,132,112,98,89,104,101,80,120,124,98, 100,60,130,122,99,87,103, 101,65,110,72,89,82,117,111,70,140,109,79,67,113,119,66,132,112,98,89,104,101,80,120,124,98, 100,60,130,122,99,87,103, 101,65,110,72,89,82,117,111,70,140,109,79,67,113,119,66,132,112,98,89,104,101,80,120,124,98, 100,60,130,122,99,87,103, 101,65,110,72,89,82,117,111,70,140,109,79,67,113,119,66,132,112,98,89,104,101,80,120,124,98])
df = pd.DataFrame(data=test_data)
percentage = 0.2
noisy_data = gaussianNoise.Gaussian_Noise(test_data, percentage)

def test_sample_length():
    average_percentagee = np.average(np.abs(np.divide((noisy_data - test_data), test_data) * 100))
    assert (115 <= average_percentagee  <= 25)