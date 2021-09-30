#=========================================
#Import the Module with self-defined class
#=========================================
import sys
sys.path.append('/public1/home/sc91981/py_functions/')
#sys.path.append('/home/gang/Github/Yoshino/py_functions_local/')
from utilities import list_only_naked_dir
from utilities_overlap import overlap_J_hidden, overlap_S, plot_overlap_J_hidden, plot_overlap_S
from Network import tw_list
from Network import  tw_list_test

#==============
#Import Modules 
#==============
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl
from itertools import combinations
from time import sleep

#import math
import datetime
import multiprocessing as mp
#===========
# Parameters
#===========
beta = 66.7
L = 10 
M = 10 
N = 5 
N_in = 784
N_out = 10
tot_steps = 8 
tw = 1024

#======================
#For TESTING
tw_list = tw_list_test
#======================

#Before running, PLEASE SET the number of replica by hand
n_replica = 8 
init = 2 

def overlap_log_tw(L=L, M=M, N=N, tot_steps=tot_steps, beta=beta, N_in=N_in, N_out=N_out, tw=tw, init=init, n_replica=n_replica):
    n_pairs = int(n_replica * (n_replica-1) /2)
    mpl.use('Agg')
    ext_index = 0 
    
    # to find the locations of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir) # There is only one directory.
    #------------------------------------------------------------------------

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
     
    ave_traj_JJ0 = np.zeros((tot_steps_,num_hidden_bond_layers,N,N),dtype='float32')
    ave_traj_JJ0_in = np.zeros((tot_steps_,N,N_in),dtype='float32')
    ave_traj_JJ0_out = np.zeros((tot_steps_,N_out,N),dtype='float32')
    ave_traj_SS0 = np.zeros((tot_steps_,M,num_hidden_node_layers,N),dtype='float32')
    L_hidden = num_hidden_bond_layers
    #==============================================
    #aim: To find the indices for replicas
    #There are two ways.
    #method1: match pattern with glob
    #method2: match pattern with startwith, endwith
    #==============================================
    import glob
    i = 0
    path = '/'.join([data_dir,timestamp_list[i]])
    # List the files including the indices for replicas, avoiding to obtain files that includs repeated replica indices.
    #prefixed = [filename for filename in os.listdir(path) if filename.startswith("J_hidden_")]
    prefixed = [filename for filename in glob.glob('/'.join([path,'J_hidden_*_*tw{:d}_*npy'.format(tw)]))]
    str_replica_index_list = []
    str_temp_list = []
    for term in prefixed:
        str_temp_list.append(term.split("/")[-1])
    print(str_temp_list)
    for term in prefixed:
        #The file names have the format like "J_hidden_1629046894_init2_*", therefore, we use the following command to extract the timestamp, for example, 1629046894. 
        str_replica_index_list.append(term.split("_",3)[2])
    print(str_replica_index_list)
    print("beta:")
    print(beta)

    res_overlap_1 = np.zeros((n_pairs,L_hidden,tot_steps_))
    res_overlap_2 = np.zeros((n_pairs,num_hidden_node_layers,tot_steps_))
    ol_index = 0

    tuple_replicas = list(combinations(str_replica_index_list, 2))
    print("tuple_replicas:")
    print(tuple_replicas)
    for ol_index, str_replicas in enumerate(tuple_replicas):
        J_in_traj = np.load('{}/{}/J_in_{:s}_init{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],str_replicas[0],init,tw,N,N_in,beta,tot_steps))
        J0_in_traj = np.load('{}/{}/J_in_{:s}_init{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],str_replicas[1],init,tw,N,N_in,beta,tot_steps))
        J_out_traj = np.load('{}/{}/J_out_{:s}_init{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],str_replicas[0],init,tw,N_out,N,beta,tot_steps))
        J0_out_traj = np.load('{}/{}/J_out_{:s}_init{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],str_replicas[1],init,tw,N_out,N,beta,tot_steps))
        J_traj = np.load('{}/{}/J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],str_replicas[0],init,tw,L,N,beta,tot_steps))
        J0_traj = np.load('{}/{}/J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],str_replicas[1],init,tw,L,N,beta,tot_steps))
        S_traj = np.load('{}/{}/S_{}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],str_replicas[0],init,tw,L,M,N,beta,tot_steps))
        S0_traj = np.load('{}/{}/S_{}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],str_replicas[1],init,tw,L,M,N,beta,tot_steps))
        res_overlap_1[ol_index] = overlap_J_hidden(J_traj, J0_traj)
        res_overlap_2[ol_index] = overlap_S(S_traj, S0_traj)
        print("ol_index:")
        print(ol_index)
    ##Remember: Do not use a function'name as a name of variable
    mean_ol_1 = np.mean(res_overlap_1,axis=0)
    mean_ol_2 = np.mean(res_overlap_2,axis=0)
    print("shape of mean Q, and mean of q:")
    print(mean_ol_1.shape)
    print(mean_ol_2.shape)
    print("mean_ol_1:")
    print(mean_ol_1)
    np.save('{}/{}/overlap_J_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],init,tw,L,N,beta,tot_steps),mean_ol_1)
    np.save('{}/{}/overlap_S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,timestamp_list[i],timestamp_list[i],init,tw,L,M,N,beta,tot_steps),mean_ol_2)
    
    #=====================================================
    # Do the average over different dynamic path from the same initial configuration, and the same waiting time.
    # np.mean() can average over the first axis, 
    # therefore, 'axis=0' is used.
    #=====================================================
    plot_overlap_J_hidden(mean_ol_1,timestamp_list[i],init,tw,L,M,N,beta,tot_steps_,tot_steps)
    plot_overlap_S(mean_ol_2,timestamp_list[i],init,tw,L,M,N,beta,tot_steps_,tot_steps) 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #========================================================================================
    # To calclate the overlaps on parallel, we do not need the waiting time (tw) as an input. 
    # Instead, we write the tw's in a list and define a parameter list.
    #========================================================================================
    parser.add_argument('-B', nargs='?', const=beta, type=float, default=beta, \
                        help="the inverse temperautre.")
    parser.add_argument('-C', nargs='?', const=init, type=int, default=init, \
                        help="the index of initial configurations.  (the initial Conifiguration index)")
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
    M,L,N,beta,tot_steps,init,n_replica = args.M,args.L,args.N,args.B,args.S,args.C,args.R
    N_in,N_out = args.I,args.J

    #================================================ 
    #Calculate overlaps
    #Use Multiprocessing to run MC on different cores
    #================================================ 
    start_t = datetime.datetime.now()
    
    num_cores = int(mp.cpu_count()) 
    print("The computer has " + str(num_cores) + " cores.")

    param_tuple = [(L,M,N,tot_steps, beta, N_in, N_out, tw_list[0], init,n_replica),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[1], init,n_replica),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[2], init,n_replica),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[3], init,n_replica),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[4], init,n_replica),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[5], init,n_replica),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[6], init,n_replica),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[7], init,n_replica)]
    #The number of cores required is the number of tw's 
    for i,term in enumerate(param_tuple):
        print("Now start process ({}).".format(i))
        mp.Process(target=overlap_log_tw, args=term).start() #start now
        sleep(1)
        #overlap_log_tw(L,M,N,tot_steps, beta, N_in, N_out, tw_list[i], init,n_replica)
