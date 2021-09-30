#=========================================
#Import the Module with self-defined class
#=========================================
import sys
sys.path.append('/public1/home/sc91981/py_functions/')
#sys.path.append('/home/gang/Github/Yoshino/py_functions/')
from Network import tw_list
from utilities import *
from utilities_overlap import *

#==============
#Import Modules 
#==============
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl

import datetime
import multiprocessing as mp
from time import sleep
#===========
# Parameters
#===========
beta = 66.7
init = 2 
L = 10 
M = 40 
N = 5 
N_in = 784
N_out = 10
tot_steps = 2048 
D = 0
#Before running, PLEASE SET the number of replica by hand
n_replica = 8 
tw = 0

def overlap_log_ave_over_init(L=L, M=M, N=N, tot_steps=tot_steps, beta=beta, N_in=N_in, N_out=N_out, tw=tw, n_replica=n_replica):
    n_pairs = n_replica * (n_replica-1)/2
    import argparse
    mpl.use('Agg')
    ext_index = 0
     
    # to find the locations of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir) # There is only one directory.

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
    ave_traj_JJ0_in = np.zeros((tot_steps_,N,N_in),dtype='float32')
    ave_traj_JJ0_out = np.zeros((tot_steps_,N_out,N),dtype='float32')
    ave_traj_SS0 = np.zeros((tot_steps_,M,L-1,N),dtype='float32')
    
    i = 0
    #========================================================================
    # For getting the shape of res_J and res_S arrays, we load overlap_X.npy
    #========================================================================
    res_J=np.load('{}/{}/overlap_J_{:s}_init2_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],tw,L,N,beta,tot_steps))
    res_S=np.load('{}/{}/overlap_S_{:s}_init2_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],tw,L,M,N,beta,tot_steps))
    print("shape of J, and of S:")
    print(res_J.shape)
    print(res_S.shape)

    init_list = [1,2,3]
    res_J_ave_over_init = np.zeros(res_J.shape)
    res_S_ave_over_init = np.zeros(res_S.shape)
    temp_res_J = np.zeros((len(init_list),res_J.shape[0],res_J.shape[1]))
    temp_res_S = np.zeros((len(init_list),res_S.shape[0],res_J.shape[1]))
    for j,init in enumerate(init_list):     
        data_dir_ = '../../image_recognition_hf_M{:d}_N{:d}_init{:d}_multiprocessing/data1'.format(M,N,init)
        timestamp_list = list_only_naked_dir(data_dir_) # There is only one directory.
        temp_res_J[j] = np.load('{}/{}/overlap_J_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir_,timestamp_list[i],timestamp_list[i],init,tw,L,N,beta,tot_steps))
        temp_res_S[j] = np.load('{}/{}/overlap_S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir_,timestamp_list[i],timestamp_list[i],init,tw,L,M,N,beta,tot_steps))
    #=====================================================
    # Do the average over different dynamic path from the same initial configuration, and the same waiting time.
    # np.mean() can average over the first axis, 
    # therefore, 'axis=0' is used.
    #=====================================================
    res_J_ave_over_init = np.mean(temp_res_J,axis=0) # "axis=0" is required
    res_S_ave_over_init = np.mean(temp_res_S,axis=0)

    #Maybe this lines are not required.
    # Go back to the current location: to find the locations of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir) # There is only one directory.

    #Save the average overlaps of J and S: 
    np.save('{}/{}/overlap_J_ave_over_init_{:s}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],tw,L,N,beta,tot_steps),res_J_ave_over_init)
    np.save('{}/{}/overlap_S_ave_over_init_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],tw,L,M,N,beta,tot_steps),res_S_ave_over_init)
    print("shape of J, and of S:")
    print(res_J.shape)
    print(res_S.shape)
    print("shape of average J, and of S:")
    print(res_J_ave_over_init.shape)
    print(res_S_ave_over_init.shape)
    
    #Plot
    plot_ave_overlap_J(res_J_ave_over_init,timestamp_list[i],tw,L,M,N,beta,tot_steps_,tot_steps)
    plot_ave_overlap_S(res_S_ave_over_init,timestamp_list[i],tw,L,M,N,beta,tot_steps_,tot_steps) 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-B', nargs='?', const=beta, type=float, default=beta, \
                        help="the inverse temperautre.")
    #parser.add_argument('-C', nargs='?', const=init, type=int, default=init, \
    #                    help="the index of initial configurations.  (the initial Conifiguration index)")
    parser.add_argument('-I', nargs='?', const=N_in, type=int, default=N_in, \
                        help="the number of bits for input.")
    parser.add_argument('-J', nargs='?', const=N_out, type=int, default=N_out, \
                        help="the number of classes for output.")
    parser.add_argument('-L', nargs='?', const=L, type=int, default=L, \
                        help="the number of layers.(Condition: L > 1)")
    parser.add_argument('-M', nargs='?', const=M, type=int, default=M, \
                        help="the number of samples.")
    parser.add_argument('-N', nargs='?', const=N, type=int, default=N, \
                        help="the number of nodes per layer.")
    parser.add_argument('-R', nargs='?', const=n_replica, type=int, default=n_replica, \
                        help="the number of replicas.")
    parser.add_argument('-S', nargs='?', const=tot_steps, type=int, default=tot_steps, \
                        help="the number of total steps.")
    args = parser.parse_args()
    M,L,N,beta,tot_steps,n_replica = args.M,args.L,args.N,args.B,args.S,args.R
    N_in,N_out = args.I,args.J

    #================================
    #To calculate the average overlap
    #================================

    start_t = datetime.datetime.now()
     
    num_cores = int(mp.cpu_count()) 
    print("The computer has " + str(num_cores) + " cores.")

    param_tuple = [(L,M,N,tot_steps, beta, N_in, N_out, tw_list[0], n_replica),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[1], n_replica),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[2], n_replica),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[3], n_replica),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[4], n_replica),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[5], n_replica),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[6], n_replica),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[7], n_replica)]
    #The number of cores required is the number of tw's 
    for i,term in enumerate(param_tuple):
        print("Now start process ({}).".format(i))
        mp.Process(target=overlap_log_ave_over_init, args=term).start() #start now
        sleep(1)
