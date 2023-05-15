from robustify import gaussian_Noise
import numpy as np
import pandas as pd

df = pd.read_csv('tests/test_column_gaussian.csv', index_col=0)
test_data = df['jet_1_pt'].values

def test_gaussian_average_value_0():
    percentage = 0.0
    noisy_data = gaussian_Noise(test_data, percentage)
    average_percentagee = np.average(np.abs(np.divide((noisy_data - test_data), test_data) * 100))
    assert (average_percentagee == 0)

def test_gaussian_average_value_10():
    percentage = 0.1
    noisy_data = gaussian_Noise(test_data, percentage)
    average_percentagee = np.average(np.abs(np.divide((noisy_data - test_data), test_data) * 100))
    assert (5 <= average_percentagee  <= 15)

def test_gaussian_average_value_20():
    percentage = 0.2
    noisy_data = gaussian_Noise(test_data, percentage)
    average_percentagee = np.average(np.abs(np.divide((noisy_data - test_data), test_data) * 100))
    assert (15 <= average_percentagee  <= 25)

def test_gaussian_average_value_25():
    percentage = 0.25
    noisy_data = gaussian_Noise(test_data, percentage)
    average_percentagee = np.average(np.abs(np.divide((noisy_data - test_data), test_data) * 100))
    assert (15  <= average_percentagee  <= 30)

def test_gaussian_average_value_50():
    percentage = 0.5
    noisy_data = gaussian_Noise(test_data, percentage)
    average_percentagee = np.average(np.abs(np.divide((noisy_data - test_data), test_data) * 100))
    assert (35  <= average_percentagee <= 65)

def test_gaussian_random_state_is_set():
    percentage = 0.2
    noisy_data_1 = gaussian_Noise(test_data, percentage, random_state=10)
    noisy_data_2 = gaussian_Noise(test_data, percentage, random_state=10)
    assert (noisy_data_1[0] == noisy_data_2[0])

def test_gaussian_random_state_not_set():
    percentage = 0.1
    noisy_data_1 = gaussian_Noise(test_data, percentage, random_state=10)
    noisy_data_2 = gaussian_Noise(test_data, percentage, random_state=11)
    assert (noisy_data_1[0] != noisy_data_2[0])

def test_guassian_noise_DataFrame_input_returns_DataFrame():
    percentage = 0.1
    noisy_DataFrame = gaussian_Noise(df, percentage, 'jet_1_pt', random_state=10)
    assert (isinstance(noisy_DataFrame, pd.DataFrame))

def test_guassian_noise_DataFrame_average_value_20():
    percentage = 0.2
    noisy_DataFrame = gaussian_Noise(df, percentage, 'jet_1_pt', random_state=10)
    noisy_data = noisy_DataFrame['jet_1_pt'].values
    average_percentagee = np.average(np.abs(np.divide((noisy_data - test_data), test_data) * 100))
    assert (15 <= average_percentagee  <= 25)