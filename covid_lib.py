# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:37:45 2020

Questo file contiene la funzione che legge da file PICKLE tutti
i dati necessari alle analisi

@author: klaat
"""


import pickle as pkl
from aidam.miscellanea_utils import find_matlabstyle


def read_covid_data(data_file,verbose=True):
    infile=open(data_file,'rb')
    trend=pkl.load(infile)
    regioni=pkl.load(infile)
    giorni=pkl.load(infile)
    giorno_settimana=pkl.load(infile)
    popolazione=pkl.load(infile)
    infile.close()
    day_counter=list(range(len(giorni)))
    if verbose:
        print('Giorni osservati: %d'%len(giorni))
        print('Primo giorno: %s'%giorni[0])
        print('Ultimo giorno: %s'%giorni[-1])
    return trend,regioni,giorni,giorno_settimana,popolazione,day_counter



def find_day_index(days_array,day_string):
    gi=find_matlabstyle(days_array, lambda x: x[0:len(day_string)]==day_string)
    return gi



