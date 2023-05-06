import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import tensorflow as tf
import types
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

def convert_to_numpy(col):
    if isinstance(col, pd.DataFrame):
        return col.to_numpy()
    elif isinstance(col, pd.Series):
        return col.to_numpy()
    elif tf.is_tensor(col):
        return col.numpy()
    elif isinstance(col, np.ndarray):
        return col
    else:
        raise Exception("could not convert {} to numpy".format(type(col)))