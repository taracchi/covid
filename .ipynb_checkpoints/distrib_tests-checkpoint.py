# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:43:01 2020

@author: klaat
"""



import numpy as np
import pickle as pkl
from datetime import datetime, timedelta


#import matplotlib.colors as mcolors

from aidam.miscellanea_utils import find_matlabstyle
from aidam.math_utils import moving_function_rev1

from aidam.faga.faga import Faga

from aidam.faga.init_functions_repo import uniform_init
from aidam.faga.terminal_functions_repo import TF_max_generations, TF_max_elapsed_time
from aidam.faga.recomb_functions_repo import constant_recombination_rates
#from fitscaling_function_repo import inverted_scaler
from aidam.faga.selection_functions_repo import *
from aidam.faga.crossover_functions_repo import *
from aidam.faga.mutation_functions_repo import *

from scipy.signal import savgol_filter

import time


#%% Lettura dati


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


giorno0=datetime.strptime(giorni[0][0:10], "%Y-%m-%d")
str_giorni=[]
for d in range(500):
    str_giorni.append((giorno0+timedelta(d)).strftime("%Y-%m-%d"))
str_giorni=np.array(str_giorni)

oggi=datetime.strftime(datetime.now(), "%Y-%m-%d")
indice_oggi=find_matlabstyle(str_giorni,lambda x:x==oggi)[0]


target_data=trend['nuovi_positivi']['Italia']
target_data_filtered=moving_function_rev1(target_data,np.mean,3,3)

fig,ax=plt.subplots(1,1,figsize=(8,4))
ax.plot(day_counter,target_data,'o')
ax.plot(day_counter,target_data_filtered,'-')
ax.set_xticks(list(range(0,len(target_data),7)))
ax.set_title('Target: casi totali')
ax.set_xticklabels(str_giorni[list(range(0,len(target_data),7))],rotation=90)
ax.grid()


#%%

def generate_global_distribution(pp_list,num_points,robustness_factor=10):
    # pp_list: lista di liste dove ogni elemento è
    # composto da [media,std,altezza]    
    out_distrib=np.zeros(num_points)
    # prendo le distribuzioni una alla volta
    for dist_par in pp_list:
        # genero la distribuzione
        y=np.random.normal(dist_par[0],dist_par[1],robustness_factor*num_points)
        # ne estraggo l'istogramma
        count,_=np.histogram(y,np.arange(0,num_points+1))
        # lo normalizzo e riscalo sul parametro del picco
        count=dist_par[2]*count/np.max(count)
        # sommo al collettore
        out_distrib=out_distrib+count
    return out_distrib



#%% Set-up genetico


num_giorni=len(day_counter)
pop_cardinality=120

max_generations=1000

# search space come righe di minimi e massimi valori dei parametri
search_space_2waves=np.array([[10,2,1000,180,2,8000],
              [100, num_giorni/2,10000,num_giorni, num_giorni/2,20000]])

search_space_3waves=np.array([[10 , 10,           1000,  100,        10,           1000,  150,        5,           5000],
                              [100, num_giorni/2,10000, num_giorni, num_giorni/2,20000, num_giorni, num_giorni/2,20000]])

search_space_3waves_samestd=np.array([[0 , 10,           1000,  10,        5,           0,  150,        5000],
                              [90, num_giorni/2,10000, num_giorni, num_giorni/2,10000, num_giorni,20000]])


search_space=search_space_3waves.copy()

#   Popolazione iniziale
initial_population=uniform_init(pop_cardinality,
                             search_space.shape[1],
                             search_space[0,:],search_space[1,:])


#    termination
def custom_termination(gao):
    return TF_max_generations(gao,max_generations)

def custom_stop_time(gao):
    return TF_max_elapsed_time(gao,10)



# crossover
def mixed_weighted_crossover(c1,c2,gao):
    scelta=np.random.choice([1,2])
    if scelta==1:
        return weighted_averaging_crossover(c1,c2,gao)
    else:
        return weighted_gene_mixing_crossover(c1,c2,gao)
        

#   recombination
rr=[0.2, 0.8, 0.06]
def basic_recombination(gao):
    rr_copy=rr.copy()
    return constant_recombination_rates(gao,rr_copy)

#   mutazione
def custom_mutation(chromosome_index,gao):
    return random_mutation_in_range(chromosome_index,gao,search_space.T,num_mutations=2)

#   fitness
def daily_infected_fitness(candidates):
    evals=np.zeros(candidates.shape[0])
    for i,c in enumerate(candidates):        
        pred_infetti=generate_global_distribution(c.reshape(-1,3),num_giorni)
        err=np.mean(np.abs(target_data-pred_infetti))
        evals[i]=-err
    return evals



#   fitness
def daily_infected_fitness_modstd(candidates):
    evals=np.zeros(candidates.shape[0])
    for i,c in enumerate(candidates):
        c=np.insert(c,len(c)-1,c[1])        
        pred_infetti=generate_global_distribution(c.reshape(-1,3),num_giorni)
        err=np.mean(np.abs(target_data-pred_infetti))
        evals[i]=-err
    return evals




#%% Creazione ottimizzatore e fit
              
ganedo=Faga(initial_population,
            fitness_fun=daily_infected_fitness,                 
            selection_fun=tophalf_selection,
            crossover_fun=mixed_weighted_crossover,
            mutation_fun=custom_mutation,
            termination_fun=custom_termination,
            recomb_rates_fun=basic_recombination,
            elite=3,            
            print_info=50,
            verbose=False)


start_time=time.time()    
solution,performance=ganedo.solve()
time.time()-start_time

#☺solution=np.insert(solution,len(solution)-1,solution[1])   

print('Achieved fitness: %.5g'%performance)  
print('Solution: ')
print(solution.reshape(-1,3))


pred_infetti=generate_global_distribution(solution.reshape(-1,3),num_giorni+30)

pred_infetti_sm=savgol_filter(pred_infetti, 11, 1)

plt.figure()
plt.plot(target_data,'ro',label='Actual')
plt.plot(target_data_filtered,'g--',linewidth=2,label='Actual filtered')
plt.plot(pred_infetti,'b-',label='Predicted')
plt.plot(pred_infetti_sm,'m--',linewidth=2,label='Predicted filtered')
plt.show()



fig,ax=plt.subplots(1,1)
ax.plot(target_data,'.',label='Contagi reali')
ax.plot(pred_infetti,'-',label='Predetti grezzi',linewidth=0.7)
ax.plot(pred_infetti_sm,'-',label='Predetti',linewidth=2)
ax.set_ylabel('Nuovi contagi')


#ax.plot(nuovi_contagi_reali_filtered,'+',label='Contagi reali filt.')
#ax.axvline(x=days_training,linewidth=2, color='k',linestyle='--',label='Fine fitting')
    
ax.set_xticks(list(range(0,num_giorni+30,30)))
ax.set_xticklabels(str_giorni[list(range(1,num_giorni+30,30))],rotation=90)
ax.grid()
ax.legend()


