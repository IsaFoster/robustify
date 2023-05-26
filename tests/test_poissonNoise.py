from robustify import poisson_noise
import numpy as np
import pandas as pd

df = pd.read_csv('tests/resources/test_column_poission.csv', index_col=0)
test_data = df['jet_n'].values

def test_poisson_random_state_is_set():
    noisy_data_1 = poisson_noise(test_data, random_state=10)
    noisy_data_2 = poisson_noise(test_data, random_state=10)
    assert (sorted(noisy_data_1) == sorted(noisy_data_2))

def test_poisson_random_state_not_set():
    noisy_data_1 = poisson_noise(test_data, random_state=10)
    noisy_data_2 = poisson_noise(test_data, random_state=11)
    assert (sorted(noisy_data_1) != sorted(noisy_data_2))

def test_poisson_noise_DataFrame_input_returns_DataFrame():
    noisy_DataFrame = poisson_noise(df, 'jet_n', random_state=10)
    assert (isinstance(noisy_DataFrame, pd.DataFrame))

def test_poisson_noise_array_input_returns_array():
    noisy_array = poisson_noise(test_data, random_state=10)
    assert (isinstance(noisy_array, np.ndarray))

def test_poisson_noise_categories_remain():
    noisy_array = poisson_noise(test_data, random_state=10)
    assert (np.unique(test_data).all() == np.unique(noisy_array).all())