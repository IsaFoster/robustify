from Noise.continuous import Gaussian_Noise
import numpy as np
import pandas as pd

df = pd.read_csv('tests/test_column.csv', index_col=0)
test_data = df['jet_1_pt'].values

def test_gaussian_average_value_0():
    percentage = 0.0
    noisy_data = Gaussian_Noise(test_data, percentage)
    average_percentagee = np.average(np.abs(np.divide((noisy_data - test_data), test_data) * 100))
    assert (average_percentagee == 0)

def test_gaussian_average_value_10():
    percentage = 0.1
    noisy_data = Gaussian_Noise(test_data, percentage)
    average_percentagee = np.average(np.abs(np.divide((noisy_data - test_data), test_data) * 100))
    assert (5 <= average_percentagee  <= 15)

def test_gaussian_average_value_20():
    percentage = 0.2
    noisy_data = Gaussian_Noise(test_data, percentage)
    average_percentagee = np.average(np.abs(np.divide((noisy_data - test_data), test_data) * 100))
    assert (15 <= average_percentagee  <= 25)

def test_gaussian_average_value_25():
    percentage = 0.25
    noisy_data = Gaussian_Noise(test_data, percentage)
    average_percentagee = np.average(np.abs(np.divide((noisy_data - test_data), test_data) * 100))
    assert (15  <= average_percentagee  <= 30)

def test_gaussian_average_value_50():
    percentage = 0.5
    noisy_data = Gaussian_Noise(test_data, percentage)
    average_percentagee = np.average(np.abs(np.divide((noisy_data - test_data), test_data) * 100))
    assert ((percentage - (percentage * 0.20)) * 100  <= (percentage + (percentage * 0.20)) * 100 )

def test_gaussian_random_state_is_set():
    percentage = 0.2
    noisy_data_1 = Gaussian_Noise(test_data, percentage, random_state=10)
    noisy_data_2 = Gaussian_Noise(test_data, percentage, random_state=10)
    assert (noisy_data_1[0] == noisy_data_2[0])

def test_gaussian_random_state_not_set():
    percentage = 0.1
    noisy_data_1 = Gaussian_Noise(test_data, percentage, random_state=10)
    noisy_data_2 = Gaussian_Noise(test_data, percentage, random_state=11)
    assert (noisy_data_1[0] != noisy_data_2[0])