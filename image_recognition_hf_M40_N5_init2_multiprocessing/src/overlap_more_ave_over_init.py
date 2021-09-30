#=================================================================================================
#Code name: overlaps_more.py
#Author: Gang Huang
#Date: 2021-4-23
#Version : 0
#Before running this code, one have to calculate all the basic overlaps vof J (or J_hidden) and S.
#=================================================================================================
import sys
sys.path.append('/public1/home/sc91981/py_functions/')
#sys.path.append('/home/gang/Github/Yoshino/py_functions/')
from utilities import *
from utilities_overlap import *

#===================================
#LOAD ALL BASIC RESULTS FOR OVERLAPS
from Network import l_list, tw_list
#TEMPERAY
#tw_list = [0,1024,4096, 8192,65536]

#=======
# Module
#=======
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl

#===========
# Parameters
#===========
beta = 66.7
L = 10 
N = 5 
N_in = 784
N_out = 10
D = 0

M = 10 

tw_list = [0, 1024, 8192, 65536, 2097152]
tot_steps_list = [1024, 1024, 1024, 1024,1024, 1024, 1024 ] 
tot_steps__list = []

if __name__ == '__main__':
    _tw = 0 #CONSTANT
    init = 2 # CONSTANT  

    #BASIC PARAMETERS
    import argparse
    mpl.use('Agg')
    ext_index = 0 
    
    # to find the locations of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir) # There is only one directory now. So set the index i=0.
    i = 0
    ##------------------------------------------------------------------------
    ## M,L,N,tot_steps, we can obtain these parameter from the first direcotry
    ## The following 5 lines do this thing. 
    ##------------------------------------------------------------------------
    #para_list_basic = np.load('{}/{}/para_list_basic.npy'.format(data_dir,timestamp_list[0])) # The para_list_basic.npy file is the last updated one.

    para_list_basic=np.array( [[L, M, N, N_in, N_out, tot_steps_list[0], tw_list[0], init],
    [L,       M,        N,      N_in,       N_out,    tot_steps_list[1], tw_list[1], init],
    [L,       M,        N,      N_in,       N_out,    tot_steps_list[2], tw_list[2], init],
    [L,       M,        N,      N_in,       N_out,    tot_steps_list[3], tw_list[3], init],
    [L,       M,        N,      N_in,       N_out,    tot_steps_list[4], tw_list[4], init]])
    #[L,       M,        N,      N_in,       N_out,    tot_steps_list[5], tw_list[5], init],
    #[L,       M,        N,      N_in,       N_out,    tot_steps_list[6], tw_list[6], init]])

    print(para_list_basic)
    para_list = para_list_basic[0] 

    SQ_N = N ** 2
    num_hidden_node_layers = L - 1 
    num_hidden_bond_layers = L - 2
    num_variables = N * M * num_hidden_node_layers 
    num_bonds = N * N_in + SQ_N * num_hidden_bond_layers + N_out * N
    num_variables = int(num_variables) 
    num_bonds = int(num_bonds)
    num = num_variables + num_bonds

    BIAS = 1
    tot_steps = max(tot_steps_list)
    tot_steps_ = int(np.log2(tot_steps * num + BIAS)) # Rescale 

    #==========================================================================
    # WE WILL LOAD ALL THE CALCULATED Overlaps, AND COMBINE THEM INTO NEW ARRAYS, NAMMED grand_J AND grand_S. THEREFORE, WE
    # create two arrays: the shape of J or S, ref averaged res_J and res_S in overlaps_twX.py.
    grand_J = np.zeros((len(tw_list), L, tot_steps_))
    grand_S = np.zeros((len(tw_list), L-1, tot_steps_))
   
    #load with loops
    print("init{}:",init)
    for index_tw, tw in enumerate(tw_list):
        grand_J[index_tw] = np.load('{}/{}/overlap_J_ave_over_init_{:s}_tw{:d}_L{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],tw,L,N,beta,tot_steps_list[index_tw]))
        grand_S[index_tw] = np.load('{}/{}/overlap_S_ave_over_init_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],tw,L,M,N,beta,tot_steps_list[index_tw]))

    # Save the averaged overlaps
    # The grand_S and grand_J are the averaged overlaps of J and S over different initial configurations.
    np.save('{}/{:s}/grand_J_{:s}_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],L,M,N,beta,tot_steps),grand_J)
    np.save('{}/{:s}/grand_S_{:s}_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],L,M,N,beta,tot_steps),grand_S)
   
    #=====================================================
    # plot the overlaps for a fixed layer (eg, 2), to see the waiting time-dependence of the overlaps Q(t,l) and q(t,l).
    #=====================================================
    for l_index in l_list: 
        plot_overlap_J_tw_X_ave_over_init(grand_J[0],grand_J[1],grand_J[2],grand_J[3],grand_J[4],timestamp_list[i],l_index,L,M,N,beta,tot_steps_,tot_steps)
        plot_overlap_S_tw_X_ave_over_init(grand_S[0],grand_S[1],grand_S[2],grand_S[3],grand_S[4],timestamp_list[i],l_index,L,M,N,beta,tot_steps_,tot_steps)
