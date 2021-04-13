# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 09:22:19 2021


Legge da Git repo i dati riguardanti la popolazione e li memorizza in
un dizionario.

https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-statistici-riferimento/popolazione-istat-regione-range.csv


@author: klaat
"""


import pandas as pd
import numpy as np
import pickle as pkl

#%% Parametri

data_address='https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-statistici-riferimento/popolazione-istat-regione-range.csv'



#%% Lettura

data=pd.read_csv(data_address)


#%% Formazione dizionario

regioni=np.unique(data['denominazione_regione'])

popolazione={}

for r in regioni:
    data_regione=data[data['denominazione_regione']==r]
    popolazione[r]=data_regione['totale_generale'].sum()
    print('%15s: %d'%(r,popolazione[r]))
    
# sostituzione chiavi (per uniformità con altri dati)
popolazione['P.A. Trento'] = popolazione['Trento']
del popolazione['Trento']
popolazione['P.A. Bolzano'] = popolazione['Bolzano']
del popolazione['Bolzano']


#%% Salvataggio

pf=open('popolazione.pkl','wb')
pkl.dump(popolazione, pf)
pf.close()