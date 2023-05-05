from Robustness.utils import _sampling
import pandas as pd
import numpy as np

test_data = {'numbers': [1, 2, 3, 4, 5, 6, 6, 8, 9, 10], 
             'letters': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], 
             'labels': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
df = pd.DataFrame(data=test_data)
sampled_data, sampled_labels = _sampling.sampleData(df, 'labels', 0.5, 10)

def test_sample_length():
    assert sampled_data.shape[0]  == 5
