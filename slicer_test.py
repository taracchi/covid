# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:42:46 2022

@author: klaat
"""

import numpy as np
from covid_lib import read_covid_data

from aidam.math_utils import moving_function_rev1

#import matplotlib.pyplot as plt

import plotly.io as pio
pio.renderers.default='browser'
import plotly.graph_objects as go


from keras import Sequential
from keras.layers import Dense, Dropout

#%% Lettura trend

trend,regioni,giorni,giorno_settimana,popolazione,day_counter=read_covid_data('stored_data.pkl')

trend['perc_tamponi_positivi']['Italia'][0]=trend['perc_tamponi_positivi']['Italia'][1]

#%% Funzioni di supporto

def get_slice(v,cut_point,lw,rw,mode='safe'):
    
    if (cut_point+rw>len(v)) or (cut_point-lw<0):
        # caso in cui sforo
        if mode=='safe':
            raise ValueError('Slice goes out of the array!')
        else:
            return None
    else:
        # qui estraggo la fetta
        return v[cut_point-lw:cut_point+rw],(cut_point-lw,cut_point+rw-1)
    



#%% Test - Creazione dataset


main_trend=moving_function_rev1(trend['nuovi_positivi']['Italia'],np.mean,3,3)
eso_trend=moving_function_rev1(trend['perc_tamponi_positivi']['Italia'],np.mean,3,3)

# finestra di input e output

input_window=14
target_window=7



# preparazione dei dati per modello

endo_input=[]
exo1_input=[]
target_data=[]

indici_input=[]
indici_target=[]

len_data_serie=len(trend['nuovi_positivi']['Italia'])


# PER TRAINING DATA
for i in range(input_window,len_data_serie-target_window+1):
#for i in range(input_window,len_data_serie):
    temp_endodata,_=get_slice(main_trend,i,input_window,0,mode='unsafe')
    temp_esodata,_=get_slice(eso_trend,i,input_window,0,mode='unsafe')
    temp_target,_=get_slice(main_trend,i,0,target_window,mode='unsafe')
    endo_input.append(temp_endodata)
    exo1_input.append(temp_esodata)
    target_data.append(temp_target)
    '''
    
    indici_input.append(temp_in_indice)
    indici_target.append(temp_tar_indice)
    '''
    
input_data=np.hstack((np.array(endo_input),np.array(exo1_input)))
target_data=np.array(target_data)

#%% modello



def create_model(num_input,num_output,num_hidden=10):
    model1 = Sequential()
    model1.add(Dropout(0.1, input_dim=num_input))
    model1.add(Dense(35,  activation='relu'))
    #model1.add(Dense(20, input_dim=num_input, activation='relu'))
    model1.add(Dense(18, activation='relu'))
    model1.add(Dense(num_output))
    return model1

mlp_model=create_model(input_data.shape[1],target_data.shape[1])
mlp_model.compile(loss='mean_absolute_percentage_error', optimizer='adam')

mlp_model.summary()

mlp_model.fit(input_data, target_data, epochs=800, batch_size=25)




last_data=np.hstack((main_trend[-input_window:],eso_trend[-input_window:]))
future_days=mlp_model.predict(last_data.reshape(1,-1))


#%% plot


fig = go.Figure()
fig.add_trace(go.Scatter(x=mydays,y=time_serie_data[-lower_data_limit:]))
fig.add_trace(go.Scatter(x=mydays[-target_window:],y=future_days[0]))
fig.show()