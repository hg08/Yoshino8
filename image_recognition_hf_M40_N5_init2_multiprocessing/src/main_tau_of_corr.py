#=========================================
#Import the Module with self-defined class
#=========================================
import sys
sys.path.append('/public1/home/sc91981/py_functions/')
#sys.path.append('/home/gang/Github/Yoshino/py_functions/')
from utilities import *
from utilities_overlap import *
from Network import l_list, tw_list

#==============
#Import Modules 
#==============
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl

beta = 66.7
L = 10 
N = 5 
N_in = 784
N_out = 10
D = 0

init = 2
M = 10 
tot_steps = 4096 
n_replica = 8

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-B', nargs='?', const=beta, type=float, default=beta, \
                        help="the inverse temperautre.")
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

    import argparse
    parser = argparse.ArgumentParser()

    #PARAMETERS
    n_pairs = n_replica * (n_replica-1)/2
    tot_steps_list = [tot_steps, tot_steps, tot_steps, tot_steps, tot_steps, tot_steps, tot_steps, tot_steps] 
    import argparse
    mpl.use('Agg')
    ext_index = 0 
    
    # to find the locations of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir) # There is only one directory.
    #------------------------------------------------------------------------
    # M,L,N,tot_steps, we can obtain these parameter from the first direcotry
    # The following 5 lines do this thing. 
 
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
    grand_J = np.load('{}/{:s}/grand_J_{:s}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],L,M,N,beta,tot_steps))
    grand_S = np.load('{}/{:s}/grand_S_{:s}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],L,M,N,beta,tot_steps))
    shape = grand_J.shape
    print("shape:{} !".format(shape))
    grand_tau_J = np.zeros((shape[0],shape[1]))
    grand_tau_S = np.zeros((shape[0],shape[1]))
    for tw_index in range(shape[0]):
        print("tw index:")
        print(tw_index)
        for l_index in range(1,shape[1]-1):
            print("l index:")
            print(l_index)
            #grand_tau_J[tw_index,l_index] = relaxation_time(grand_J[tw_index,l_index])
            grand_tau_S[tw_index,l_index] = relaxation_time_p(grand_S[tw_index,l_index], tot_steps)
            print("grand_tau_S:")
            print(grand_tau_S)
            #print("grand_tau_J:")
            #print(grand_tau_J)
            print(grand_tau_S[tw_index,l_index])
            #print(grand_tau_J[tw_index,l_index])
    #np.save('{}/{:s}/grand_tau_J_{:s}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],L,M,N,beta,tot_steps), grand_tau_J)
    #np.save('{}/{:s}/grand_tau_S_{:s}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],L,M,N,beta,tot_steps), grand_tau_S)
    grand_tau_S = grand_tau_S/num
    print("grand_tau_S:")
    print(grand_tau_S)
    np.save('{}/grand_tau_J_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,L,M,N,beta,tot_steps), grand_tau_J)
    np.save('{}/grand_tau_S_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,L,M,N,beta,tot_steps), grand_tau_S)
    #Plot tau_J and tau_S
    plot_tau_J_tw_X(grand_tau_J,timestamp_list[i],L,M,N,beta,tot_steps,tw_list)
    plot_tau_S_tw_X(grand_tau_S,timestamp_list[i],L,M,N,beta,tot_steps,tw_list)
