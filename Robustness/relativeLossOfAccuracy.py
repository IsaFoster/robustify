from sklearn.metrics import accuracy_score
from Setup import _readData
from noiseCorruptions import addNoiseDf, noiseCorruptions
from sklearn.svm import SVC
import random
import numpy as np
import plotly.express as px
import pandas as pd
import chart_studio.plotly as py
import chart_studio.tools as tls
import os
from dotenv import load_dotenv

load_dotenv()

username = os.getenv('USER_NAME')
api_key = os.getenv('API_KEY')

tls.set_credentials_file(username=username, api_key=api_key)

seed = 39
random.seed(seed)
np.random.seed(seed=seed)

model = SVC(kernel='linear')
X_train, y_train, X_test, y_test = _readData.getXandYFromFile()

def plotRLA(model, X_train, y_train, X_test, y_test, seed):
    X_train['data_type'] = y_train
    acc_list = noiseCorruptions(X_train, X_test, y_test, model, random_state=seed, plot=False)
    rla_list = []
    for accuracy in acc_list:
        rla_list.append(calculateRLA(acc_list[0], accuracy))
    df_temp = pd.DataFrame(columns=['Noise Level', 'Accuracy', 'Relative Loss of Accuracy'])
    df_temp['Noise Level'] = np.linspace(0, 1, 11)
    df_temp['Accuracy'] = acc_list
    df_temp['Relative Loss of Accuracy'] = rla_list

    fig = px.line(df_temp, x='Noise Level', y='Relative Loss of Accuracy', title="Relative loss off accuracy over increasing feature noise")
    fig_2 = px.line(df_temp, x='Noise Level', y='Accuracy', title="Accuracy over increasing feature noise")

    py.plot(fig, filename='RelativeLossPlot')
    py.plot(fig_2, filename='AccuracyPlot')
    print(df_temp)

def calculateRLA(accuracy_0, accuracy_x):
    return (accuracy_0 - accuracy_x) / accuracy_0


plotRLA(model, X_train, y_train, X_test, y_test, seed)