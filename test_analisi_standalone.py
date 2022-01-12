# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:30:58 2022

@author: klaat
"""

import pandas as pd
import numpy as np
#import datetime
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')


data_file = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv'

data=pd.read_csv(data_file)

data['data'].apply(lambda x: x[0:10])

giorni=pd.unique(data['data'])
day_counter=list(range(len(giorni)))
print('Giorni osservati: %d'%len(giorni))
print('Da: %s'%giorni[0][0:10])
print(' A: %s'%giorni[-1][0:10])
regioni=pd.unique(data['denominazione_regione'])

for colname in data.columns:
    print(colname,end=', ')
    
data_italia = pd.DataFrame(data.groupby(['data']).sum())
data_italia.reset_index(inplace=True)

# aggiunta info da colonne calcolate

data_italia['deceduti_giornalieri']=data_italia['deceduti'].diff()