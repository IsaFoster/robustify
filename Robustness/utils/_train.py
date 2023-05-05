import torch
import torch.nn as nn
from Robustness.utils._transform import convert_to_numpy

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

