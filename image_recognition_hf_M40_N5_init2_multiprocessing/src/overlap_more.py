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
from Network import l_list,tw_list
#Temperay
tw_list = [0, 1024, 8192, 65536]

#=======
# Module
#=======
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl

if __name__ == '__main__':
    _tw = 0 #CONSTANT
    _init = 2 # CONSTANT  

    #BASIC PARAMETERS
    import argparse
    mpl.use('Agg')
    ext_index = 0 
    
    # to find the locations of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir) # There is only one directory now. So set the index i=0.
    i = 0
    # IMPORT THE TEMPERATURE
    beta_tmp = np.load('{}/{}/para_list_beta.npy'.format(data_dir,timestamp_list[i]))
    beta = beta_tmp[0]
    #------------------------------------------------------------------------
    # M,L,N,tot_steps, we can obtain these parameter from the first direcotry
    # The following 5 lines do this thing. 
    #------------------------------------------------------------------------
    para_list_basic = np.load('{}/{}/para_list_basic.npy'.format(data_dir,timestamp_list[0])) # The para_list_basic.npy file is the last updated one.
    para_list = para_list_basic[0] 
    L = para_list[0]
    M = para_list[1]
    N = para_list[2]
    N_in = para_list[3]
    N_out = para_list[4]
    tot_steps = para_list[5]

    SQ_N = N ** 2
    num_hidden_node_layers = L - 1 
    num_hidden_bond_layers = L - 2
    num_variables = N * M * num_hidden_node_layers 
    num_bonds = N * N_in + SQ_N * num_hidden_bond_layers + N_out * N
    num_variables = int(num_variables) 
    num_bonds = int(num_bonds)
    num = num_variables + num_bonds

    BIAS = 1
    tot_steps_ = int(np.log2(tot_steps * num + BIAS)) # Rescale 
    init = para_list[7]

    #==========================================================================
    #LOAD ALL BASIC RESULTS FOR OVERLAPS
    #overlap_J_1618938013_init0_tw0_L10_M50_N10_beta66.7_step4200.npy (EXAMPLE)
    #==========================================================================
    # WE WILL LOAD ALL THE CALCULATED Overlaps, AND COMBINE THEM INTO NEW ARRAYS, NAMMED grand_J AND grand_S. THEREFORE, WE
    # create two arrays: the shape of J or S, ref res_J and res_S in overlaps_twX.py.
    grand_J = np.zeros((len(tw_list), L-2, tot_steps_))
    grand_S = np.zeros((len(tw_list), L-1, tot_steps_))
    #load with loops
    print("init{}:",init)
    for index_tw, tw in enumerate(tw_list):
        grand_J[index_tw] = np.load('{}/{}/overlap_J_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],init,tw,L,N,beta,tot_steps))
        grand_S[index_tw] = np.load('{}/{}/overlap_S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],init,tw,L,M,N,beta,tot_steps))

    np.save('{}/{:s}/grand_J_{:s}_init{:d}_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],init,L,M,N,beta,tot_steps),grand_J)
    np.save('{}/{:s}/grand_S_{:s}_init{:d}_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],init,L,M,N,beta,tot_steps),grand_S)
    #=====================================================
    # plot the overlaps for a fixed layer (eg, 2), to see the waiting time-dependence of the overlaps Q(t,l) and q(t,l).
    #=====================================================
    # Case 
    for l_index in l_list: 
        #plot_overlap_J_tw(grand_J[0],grand_J[1],grand_J[2],grand_J[3],grand_J[4],grand_J[5],grand_J[6],timestamp_list[i],init,l_index,L,M,N,beta,tot_steps_,tot_steps)
        #plot_overlap_S_tw(grand_S[0],grand_S[1],grand_S[2],grand_S[3],grand_S[4],grand_S[5],grand_S[6],timestamp_list[i],init,l_index,L,M,N,beta,tot_steps_,tot_steps)
        plot_overlap_J_tw_4(grand_J[0],grand_J[1],grand_J[2],grand_J[3],timestamp_list[i],init,l_index,L,M,N,beta,tot_steps_,tot_steps)
        plot_overlap_S_tw_4(grand_S[0],grand_S[1],grand_S[2],grand_J[3],timestamp_list[i],init,l_index,L,M,N,beta,tot_steps_,tot_steps)
    ## Case 
    #l_index = 2
    #plot_overlap_J_tw(grand_J[0],grand_J[1],grand_J[2],grand_J[3],grand_J[4],timestamp_list[i],init,l_index,L,M,N,beta,tot_steps_)
    #plot_overlap_S_tw(grand_S[0],grand_S[1],grand_S[2],grand_S[3],grand_S[4],timestamp_list[i],init,l_index,L,M,N,beta,tot_steps_)
