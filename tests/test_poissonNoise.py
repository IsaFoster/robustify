from Noise.discrete import Poisson_noise
import numpy as np
import pandas as pd

df = pd.read_csv('tests/test_column_poission.csv', index_col=0)
test_data = df['jet_n'].values

def test_poisson_average_value():
    noisy_data = Poisson_noise(test_data)
    average_value = np.average(np.abs((noisy_data - test_data)))
    assert (0 <= average_value  <= 1)

def test_poisson_random_state_is_set():
    noisy_data_1 = Poisson_noise(test_data, random_state=10)
    noisy_data_2 = Poisson_noise(test_data, random_state=10)
    assert (sorted(noisy_data_1) == sorted(noisy_data_2))

def test_poisson_random_state_not_set():
    noisy_data_1 = Poisson_noise(test_data, random_state=10)
    noisy_data_2 = Poisson_noise(test_data, random_state=11)
    assert (sorted(noisy_data_1) != sorted(noisy_data_2))

def test_poisson_noise_DataFrame_input_returns_DataFrame():
    noisy_DataFrame = Poisson_noise(df, 'jet_n', random_state=10)
    assert (isinstance(noisy_DataFrame, pd.DataFrame))

def test_poisson_noise_DataFrame_average_value():
    noisy_DataFrame = Poisson_noise(df, 'jet_n', random_state=10)
    noisy_data = noisy_DataFrame['jet_n']
    average_value = np.average(np.abs((noisy_data - test_data)))
    assert (0 <= average_value  <= 1)