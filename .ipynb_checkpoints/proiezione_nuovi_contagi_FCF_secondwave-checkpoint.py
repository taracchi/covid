#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pickle as pkl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from aidam.miscellanea_utils import find_matlabstyle

import seaborn as sns

from aidam.faga.curve_fit import FagaCurveFit

from aidam.math_utils import moving_function_rev1
from aidam.NbEnhance import tabprint


# ## Lettura dati

# In[28]:


data_file='stored_data.pkl'

infile=open(data_file,'rb')
trend=pkl.load(infile)
regioni=pkl.load(infile)
giorni=pkl.load(infile)
giorno_settimana=pkl.load(infile)
infile.close()

day_counter=list(range(len(giorni)))

print('Giorni osservati: %d'%len(giorni))
print('Primo giorno: %s'%giorni[0])
print('Ultimo giorno: %s'%giorni[-1])


# Preparazione date
# 
# - creo il giorno 0 nel formato *datetime* per poi creare i nuovi plot su questa base
# - creo una lista di giorni (in formato di stringhe) partendo dal giorno0

# In[29]:


giorno0=datetime.strptime(giorni[0][0:10], "%Y-%m-%d")
str_giorni=[]
for d in range(500):
    str_giorni.append((giorno0+timedelta(d)).strftime("%Y-%m-%d"))
str_giorni=np.array(str_giorni)

oggi=datetime.strftime(datetime.now(), "%Y-%m-%d")
indice_oggi=find_matlabstyle(str_giorni,lambda x:x==oggi)[0]


# ## Funzioni base
# 
# Si testano due versioni generalizzate della sigmoide:
# 
# - sigmoide generalizzata
# - sigmoide generalizzata estesa

# In[30]:


def generalized_sigmoid(t,a,b,M,alpha):
    y=M/(1+np.exp(-a*t+b))**alpha
    return y

def generalized_sigmoid_ext(t,a,b,c,M,alpha):
    y=(M+c*t)/(1+np.exp(-a*t+b))**alpha
    return y


def two_waves_generalized_sigmoids_ext(t,t_sw,a1,b1,c1,M1,alpha1,a2,b2,c2,M2,alpha2):    
    if t<t_sw:
        y=(M1+c1*t)/(1+np.exp(-a1*t+b1))**alpha1
    else:
        #y=(M1+c1*t)/(1+np.exp(-a1*t+b1))**alpha1+(M2+c2*t)/(1+np.exp(-a2*(t-t_sw)+b2))**alpha2
        y1=(M1+c1*t_sw)/(1+np.exp(-a1*t_sw+b1))**alpha1
        y2=(M2+c2*(t-t_sw-1))/(1+np.exp(-a2*(t-t_sw-1)+b2))**alpha2
        y=y1+y2
    return y


def two_waves_generalized_sigmoids(t,t_sw,a1,b1,M1,alpha1,a2,b2,M2,alpha2):
    t_sw=np.round(t_sw)
    if t<t_sw:
        y=(M1)/(1+np.exp(-a1*t+b1))**alpha1    
    else:        
        y1=(M1)/(1+np.exp(-a1*t_sw+b1))**alpha1
        y2=(M2)/(1+np.exp(-a2*(t-t_sw)+b2))**alpha2
        y=y1+y2
    return y



employed_model=two_waves_generalized_sigmoids_ext


# ### Test plots
test_pars=[115,
           0.1, 0.05, 5,2.5e+5, 2,
           0.1, 0,    2, 2.5e+5, 2]
           
y_pred=np.array(list(map(lambda x:employed_model(x,*test_pars),day_counter)))

plt.plot(day_counter,y_pred)
# ## Fitting

# In[31]:


target_data=np.cumsum(trend['nuovi_positivi']['Italia'])
#target_data=np.cumsum(norm_np_it)
target_data_filtered=moving_function_rev1(target_data,np.mean,3,3)

fig,ax=plt.subplots(1,1,figsize=(8,4))
ax.plot(day_counter,target_data,'o')
ax.plot(day_counter,target_data_filtered,'-')
ax.set_xticks(list(range(0,len(target_data),7)))
ax.set_title('Target: casi totali')
ax.set_xticklabels(str_giorni[list(range(0,len(target_data),7))],rotation=90)
ax.grid()


# ## GA-based Curve fitting
# 
# E' possibile utilizzare solo **parzialmente** i dati per il tuning del modello: tramite il parametro `days_training` si decide quanti giorni usare effettivamente dall'inizio del dataset
# 

# In[43]:


days_training=len(target_data)#-28

# Italia
a_range=[0,1]
b_range=[0, 1]
c_range=[0,20]
M_range=[1e+5,5e+5]
alpha_range=[2,15] # 5 - 15
t_sw_range=[150,180]

'''
# dominio per generalized_sigmoid_ext
dominio=np.vstack((a_range,
                   b_range,
                   c_range,
                   M_range,
                   alpha_range))
'''



# dominio per two_waves_generalized_sigmoids_ext
#M_range=[0,3e+5]
dominio=np.vstack((t_sw_range,
                   a_range,
                   b_range,
                   c_range,
                   M_range,
                   alpha_range,
                   a_range,
                   b_range,
                   c_range,
                   M_range,
                   alpha_range))


# dominio per two_waves_generalized_sigmoids
#M_range=[0,3e+5]
dominio=np.vstack((t_sw_range,
                   a_range,
                   b_range,
                   M_range,
                   alpha_range,
                   a_range,
                   b_range,
                   M_range,
                   alpha_range))

                   
dominio=dominio.T

tabprint(dominio,digits=10,precision=1,index=['Min','Max'])


# In[44]:


def error_measure(predicted,actual):
    #return np.mean(np.abs(predicted-actual))+np.std(np.abs(predicted-actual))
    #return 0.7*np.mean(np.abs(predicted-actual))+0.3*np.mean(np.abs(predicted[-14:]-actual[-14:]))
    #return 0.7*np.mean(np.abs(predicted[:-30]-actual[:-30]))+0.15*np.mean(np.abs(predicted[-14:-7]-actual[-14:-7]))+0.15*np.mean(np.abs(predicted[-7:]-actual[-7:]))
    return np.percentile(np.abs(predicted-actual),80)+np.mean(np.abs(predicted-actual))
    #return np.mean(np.abs(predicted-actual))

np.random.seed(42)
    
fct=FagaCurveFit()

fct.fit(employed_model, 
        np.arange(days_training),
        target_data[0:days_training],
        dominio,
        ga_generations=1200, ga_population=100,print_info=200,
        error_function=error_measure,
        num_mutations=1)


# In[45]:


solution=fct.opt_params

print('\nOptimal solution is: ')

print(solution)
# Interpretazione della soluzione
print('Inizio second wave: giorno %d, %s'%(solution[0],giorni[int(np.round(solution[0]))]))


# In[46]:


fig,ax=plt.subplots(1,2,figsize=(14,5))
ax[0].plot(fct.fitness_tracker['average'],label='Average')
ax[0].plot(fct.fitness_tracker['max'],label='Max')
ax[0].legend()
ax[1].plot(fct.fitness_tracker['std'],label='Std')
ax[1].legend()
fig.suptitle('Fitness trends')


# Adesso applico il modello tunato sia ai giorni passati sia a giorni futuri.
# 
# *forecast_days* sono i giorni per cui lo applico: iniziano dal giorno 0 e finoscono quando si vuole nel futuro

# In[47]:


forecast_days=np.arange(300)
y_pred=np.array(list(map(lambda x:employed_model(x,*solution),forecast_days)))


# #### Plots

# In[48]:


# preparazione palette colori
lista_colori=list(mcolors.TABLEAU_COLORS.items())
color_names=[cn[0] for cn in lista_colori]

fig,ax=plt.subplots(2,1,figsize=(12,10))
ax[0].plot(day_counter,target_data,label='Reale',linewidth=4)
ax[0].plot(forecast_days,y_pred,'r--',label='Predetto')
ax[0].axvline(days_training-1,c='g',ls='--',label='Fine tuning')
ax[0].set_xticks(list(range(0,len(forecast_days),5)))
ax[0].set_xticklabels(str_giorni[list(range(0,len(forecast_days),5))],rotation=90)
ax[0].set_ylabel('Totale infetti')
ax[0].legend()
ax[0].grid()

ax[1].bar(day_counter,y_pred[day_counter]-target_data)
ax[1].axvline(days_training-1)
ax[1].set_title('Errore giornaliero')
fig.tight_layout()
fig.show()


# ### Adesso vediamo, secondo il modello, quando non avremo più contagiati

# In[49]:


nuovi_contagi_pred=np.diff(y_pred)

nuovi_contagi_reali=np.diff(target_data)
nuovi_contagi_reali_filtered=moving_function_rev1(nuovi_contagi_reali,np.mean,3,3)#np.diff(target_data_filtered)


# In[50]:


fig,ax=plt.subplots(1,1,figsize=(14,6))
ax.plot(nuovi_contagi_pred,'-',label='Nuovi contagi predetti')
ax.set_ylabel('Nuovi contagi')

ax.plot(nuovi_contagi_reali,'o',label='Contagi reali')
#ax.plot(nuovi_contagi_reali_filtered,'+',label='Contagi reali filt.')
ax.axvline(x=days_training,linewidth=2, color='k',linestyle='--',label='Fine fitting')
    
ax.set_xticks(list(range(0,len(forecast_days),10)))
ax.set_xticklabels(str_giorni[list(range(1,len(forecast_days),10))],rotation=90)
ax.grid()
ax.legend()


# In[40]:


import plotly.graph_objects as go


# In[41]:


fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=day_counter,
                         y=nuovi_contagi_reali,
                         mode='markers',
                         name='Reali'))
fig.add_trace(go.Scatter(x=day_counter,
                         y=nuovi_contagi_pred,
                         mode='lines',
                         name='Predetti'))
fig.add_trace(go.Scatter(x=day_counter,
                         y=nuovi_contagi_reali_filtered,
                         mode='markers',
                         name='Reali filtrati'))
fig.update_layout(title='Nuovi contagi',
                  xaxis_tickmode = 'array',
                  xaxis_tickvals = np.arange(0,len(day_counter),7),
                  xaxis_ticktext = [g[0:10] for g in giorni[np.arange(0,len(day_counter),7)]])
fig.show()


# In[42]:


from scipy.signal import savgol_filter


fig,ax=plt.subplots(1,1,figsize=(12,6))
errore_giornaliero=nuovi_contagi_pred[0:len(nuovi_contagi_reali)]-nuovi_contagi_reali
ax.bar(day_counter[0:len(nuovi_contagi_reali)],errore_giornaliero)
#ax.plot(day_counter[0:len(nuovi_contagi_reali)],moving_function_rev1(errore_giornaliero,np.mean,5,2),'r',linewidth=3)
ax.plot(day_counter[0:len(nuovi_contagi_reali)],savgol_filter(errore_giornaliero,13,1),'r',linewidth=3)
ax.axvline(days_training-2)
ax.set_title('Nuovi contagi - Errore')
ax.grid()
fig.tight_layout()
fig.show()


# In[ ]:





# In[ ]:




