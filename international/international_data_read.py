# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:39:53 2020

@author: klaat
"""

import pandas as pd


# Beg ===== Params ===== 
data_file='owid-covid-data.xlsx'


# End ===== Params ===== 


# Lettura
data=pd.read_excel(data_file)
col_names=data.columns