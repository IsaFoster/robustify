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

'''custom train should tak in the model, x and y. SHould return trained model. Custom train likely also need custom scoring function'''
def custom_train_model(model, X, y, custom_train):
    if not hasattr(custom_train, '__call__'):
        raise Exception("Custom training must be a callable function")
    try:
        X = convert_to_numpy(X)
        y = convert_to_numpy(y)
        return custom_train(model, X, y)
    except:
        raise Exception("Could not train the model")

def train_model(model, X, y, custom_train):
    if custom_train != None:
        return custom_train_model(model, X, y, custom_train)  
    model.fit(X, y.values.ravel())
    return model

def reset_model(model):
    if isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_uniform(model.weight.data)
    return model

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