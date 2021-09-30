#=========================================
#Import the Module with self-defined class
#=========================================
import sys
sys.path.append('/public1/home/sc91981/py_functions/')
#sys.path.append('/home/gang/Github/Yoshino/py_functions/')
from utilities import *
from utilities_overlap import *
from Network import l_list, tw_list
#Temperay
#tw_list = [0, 1024, 8192, 65536, 262144, 1048576, 2094152, 4194304, 8388608]
tw_list = [0, 1024, 4096, 8192, 65536]

#==============
#Import Modules 
#==============
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    D = 0
    #Before running, PLEASE SET the number of replica by hand
    R = 0
    parser.add_argument('-D', nargs='?', const=D, type=int, default=D, \
                        help="the waiting time.")
    parser.add_argument('-R', nargs='?', const=R, type=int, default=R, \
                        help="the number of replicas.")
    args = parser.parse_args()
    _tw,n_replica = args.D, args.R

    #PARAMETERS
    n_pairs = n_replica * (n_replica-1)/2
    import argparse
    mpl.use('Agg')
    ext_index = 0 
    
    # to find the locations of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir) # There is only one directory.
    #------------------------------------------------------------------------
    # M,L,N,tot_steps, we can obtain these parameter from the first direcotry
    # The following 5 lines do this thing. 
    #------------------------------------------------------------------------
    para_list_basic = np.load('{}/{}/para_list_basic.npy'.format(data_dir,timestamp_list[0])) 
    beta_list = np.load('{}/{}/para_list_beta.npy'.format(data_dir,timestamp_list[0]))
    print('para_list_basic:')
    print(para_list_basic)
    #===========================================================================
    #IF YOU NEED TEMPARY SETTING, MODIFY THIS BLOCK
    #para_list_basic=np.array( [[10, 80, 5, 784, 10, 8192, 1024, 2],
    #[10,       80,        5,      784,       10,     8192,     8192,        2],
    #[10,       80,        5,      784,       10,     8192,    65536,        2],
    #[10,       80,        5,      784,       10,     8192,   524288,        2],
    #[10,       80,        5,      784,       10,     8192,  4194304,        2]])
    #print(para_list_basic)
    #===========================================================================
    #IF YOU NEED TEMPARY SETTING
    para_list = para_list_basic[0] 

    L = para_list[0]
    M = para_list[1]
    N = para_list[2]
    N_in = para_list[3]
    N_out = para_list[4]
    tot_steps = para_list[5]
    init = para_list[7]
    # Inverse temperature (beta)
    beta = beta_list[0]   
 
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
     
    ave_traj_JJ0 = np.zeros((tot_steps_,L-2,N,N),dtype='float32')
    ave_traj_SS0 = np.zeros((tot_steps_,M,L-1,N),dtype='float32')
    
    #aim: To find the indices for replicas
    #method1: match pattern with glob
    #method2: match pattern with startwith, endwith
    import glob
    i = 0
    path = '/'.join([data_dir,timestamp_list[i]])
    #prefixed = [filename for filename in os.listdir(path) if filename.startswith("J_hidden_")]
    #prefixed = [filename for filename in glob.glob('/'.join([path,'J_hidden_*_*tw{:d}_*npy'.format(tw)]))]
    prefixed = [filename for filename in glob.glob('/'.join([path,'overlap_*npy']))]
    grand_J = np.load('{}/{:s}/grand_J_{:s}_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],L,M,N,beta,tot_steps))
    grand_S = np.load('{}/{:s}/grand_S_{:s}_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],L,M,N,beta,tot_steps))
    shape = grand_J.shape
    grand_tau_J = np.zeros((shape[0],shape[1]))
    grand_tau_S = np.zeros((shape[0],shape[1]))
    for tw_index in range(shape[0]):
        print("tw index:")
        print(tw_index)
        for l_index in range(1,shape[1]):
            print("l index:")
            print(l_index)
            grand_tau_J[tw_index,l_index] = relaxation_time(grand_J[tw_index,l_index])
            grand_tau_S[tw_index,l_index] = relaxation_time(grand_S[tw_index,l_index])
            print("grand_tau_J:")
            print(grand_tau_J)
            print("grand_tau_S:")
            print(grand_tau_S)
            grand_tau_J = grand_tau_J/num
            grand_tau_S = grand_tau_S/num
            print(grand_tau_J[tw_index,l_index])
            print(grand_tau_S[tw_index,l_index])
    #np.save('{}/{:s}/grand_tau_J_{:s}_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],L,M,N,beta,tot_steps), grand_tau_J)
    #np.save('{}/{:s}/grand_tau_S_{:s}_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],L,M,N,beta,tot_steps), grand_tau_S)
    np.save('{}/grand_tau_J_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,L,M,N,beta,tot_steps), grand_tau_J)
    np.save('{}/grand_tau_S_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,L,M,N,beta,tot_steps), grand_tau_S)
    #Plot tau_J and tau_S
    plot_tau_J_tw_X(grand_tau_J,timestamp_list[i],L,M,N,beta,tot_steps)
    plot_tau_S_tw_X(grand_tau_S,timestamp_list[i],L,M,N,beta,tot_steps)
