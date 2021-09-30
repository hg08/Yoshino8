#=========================================
#Import the Module with self-defined class
#=========================================
import sys
sys.path.append('/public1/home/sc91981/py_functions/')
#sys.path.append('/home/gang/Github/Yoshino/py_functions/')
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

#===========
# Parameters
#===========
beta = 66.7
init = 2 
L = 10 
M = 10 
N = 5 
N_in = 784
N_out = 10
S = 64 
D = 0
#Before running, PLEASE SET the number of replica by hand
R = 64 
tw = 0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-B', nargs='?', const=beta, type=float, default=beta, \
                        help="the inverse temperautre.")
    parser.add_argument('-C', nargs='?', const=init, type=int, default=init, \
                        help="the index of initial configurations.  (the initial Conifiguration index)")
    parser.add_argument('-D', nargs='?', const=tw, type=int, default=tw, \
                        help="the waiting time. (the time Delay)")
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
    parser.add_argument('-R', nargs='?', const=R, type=int, default=R, \
                        help="the number of replicas.")
    parser.add_argument('-S', nargs='?', const=S, type=int, default=S, \
                        help="the number of total steps.")
    args = parser.parse_args()
    M,L,N,beta,tot_steps,tw,init,n_replica = args.M,args.L,args.N,args.B,args.S,args.D,args.C,args.R
    N_in,N_out = args.I,args.J

    #PARAMETERS
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
    ave_traj_SS0 = np.zeros((tot_steps_,M,L-1,N),dtype='float32')
    
    ##aim: To find the indices for replicas
    ##method1: match pattern with glob
    ##method2: match pattern with startwith, endwith
    #import glob
    i = 0
    #path = '/'.join([data_dir,timestamp_list[i]])
    ## step 1:list the files including the indices for replicas, avoiding to obtain files that incuds repeated replica indices.
    ##prefixed = [filename for filename in os.listdir(path) if filename.startswith("J_hidden_")]
    #prefixed = [filename for filename in glob.glob('/'.join([path,'J_hidden_*_*tw{:d}_*npy'.format(tw)]))]
    #str_replica_index_list = []
    #str_temp_list = []
    #for term in prefixed:
    #    str_temp_list.append(term.split("/")[-1])
    #print(str_temp_list)
    #for term in prefixed:
    #    str_replica_index_list.append(term.split("_",3)[2])
    #print(str_replica_index_list)
    
    #beta_tmp = np.load('{}/{}/para_list_beta.npy'.format(data_dir,timestamp_list[i]))
    #beta = beta_tmp[0]
    #print("beta:")
    #print(beta)

    #========================================================================
    # For getting the shape of res_J and res_S arrays, we load overlap_X.npy
    #========================================================================
    res_J=np.load('{}/{}/overlap_J_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],init,tw,L,N,beta,tot_steps))
    res_S=np.load('{}/{}/overlap_S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],init,tw,L,M,N,beta,tot_steps))
    print("shape of J, and of S:")
    print(res_J.shape)
    print(res_S.shape)

    init_list = [1,2,3]
    res_J_ave_over_init = np.zeros(res_J.shape)
    res_S_ave_over_init = np.zeros(res_S.shape)
    temp_res_J = np.zeros((len(init_list),res_J.shape[0],res_J.shape[1]))
    temp_res_S = np.zeros((len(init_list),res_S.shape[0],res_J.shape[1]))
    for j,init in enumerate(init_list): 
        data_dir = '../../image_recognition_hf_M{:d}_N{:d}_init{:d}/data1'.format(M,N,init)
        timestamp_list = list_only_naked_dir(data_dir) # There is only one directory.
        temp_res_J[j] = np.load('{}/{}/overlap_J_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],init,tw,L,N,beta,tot_steps))
        temp_res_S[j] = np.load('{}/{}/overlap_S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],init,tw,L,M,N,beta,tot_steps))
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
    np.save('{}/{}/overlap_J_ave_over_init_{:s}_tw{:d}_L{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],tw,L,N,beta,tot_steps),res_J_ave_over_init)
    np.save('{}/{}/overlap_S_ave_over_init_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],tw,L,M,N,beta,tot_steps),res_S_ave_over_init)
    print("shape of J, and of S:")
    print(res_J.shape)
    print(res_S.shape)
    print("shape of average J, and of S:")
    print(res_J_ave_over_init.shape)
    print(res_S_ave_over_init.shape)
    
    #Plot
    plot_ave_overlap_J(res_J_ave_over_init,timestamp_list[i],tw,L,M,N,beta,tot_steps_,tot_steps)
    plot_ave_overlap_S(res_S_ave_over_init,timestamp_list[i],tw,L,M,N,beta,tot_steps_,tot_steps) 
