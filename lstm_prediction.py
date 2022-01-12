# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 09:54:34 2021

Predizioni con LSTM


@author: klaat
"""



import numpy as np
import matplotlib.pyplot as plt

from covid_lib import read_covid_data

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from aidam.math_utils import moving_function_rev1


#%% Lettura e preprocessing

trend,regioni,giorni,giorno_settimana,popolazione,day_counter=read_covid_data('stored_data.pkl')

# creazione del dataset

X=trend['nuovi_positivi']['Italia']
X=moving_function_rev1(X,np.mean,3,3)

scaler=MinMaxScaler()
X=scaler.fit_transform(X.reshape(-1, 1))

Y=X[1:]
X=X[:-1]

# divisione TR VD
num_samples=len(X)
num_tr=int(np.round(0.7*num_samples))
Xtr=X[:num_tr]
Xts=X[num_tr:]
Ytr=Y[:num_tr]
Yts=Y[num_tr:]


# reshape input to be [samples, time steps, features]
lstm_xtr_data = np.reshape(Xtr, (Xtr.shape[0], 1, 1))
lstm_xts_data = np.reshape(Xts, (Xts.shape[0], 1, 1))
lstm_ytr_data = np.reshape(Ytr, (Ytr.shape[0], 1, 1))
lstm_yts_data = np.reshape(Yts, (Yts.shape[0], 1, 1))


#%% Training

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(lstm_xtr_data, lstm_ytr_data, epochs=100, batch_size=1, verbose=2)

#%% Test

# make predictions
trainPredict = model.predict(lstm_xtr_data)
testPredict = model.predict(lstm_xts_data)


#%% Figure

fig=plt.figure()
plt.plot(Ytr)
plt.plot(trainPredict,'r.')
fig.show()



fig=plt.figure()
plt.plot(Yts)
plt.plot(testPredict,'r.')
fig.show()