#============
# main.py
# Date: 21-8-9
#=============
# host.py 
# 1. Run a MC dynamics and save the trajectories for J_in, S, etc.
# 2. Save the seeds for J_in, J_out, S, etc at t = 2**N, where N = 2, 4, 8, 16, ... 
#========
# guest.py 
# 1. Run a MC dynamics and save the trajectories for J_in, S, etc. Do not save seeds.
#========
#Module 1
#========
import sys
sys.path.append('/public1/home/sc91981/py_functions/')
#sys.path.append('/home/gang/Github/Yoshino/py_functions_local/')

from utilities import calc_ener, generate_S_in_and_out, list_only_naked_dir
from HostNetwork import HostNetwork
from Network import Network,tw_list, tw_list_test

#import math
import datetime
import multiprocessing as mp

#========
#Module 2 
#========
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from random import choice
from random import randrange
import scipy as sp
from scipy.stats import norm
import tensorflow as tf
from time import sleep 
from time import time


#===========
# Metedata
#===========
wait0 = 600 # For different network, we use different wait0, which is the time have to wait to generate all the seeds.
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
tot_steps = 8 
tw = 0

#If test, turn on the flowing line
tw_list = tw_list_test
#=========
#Functions
#=========
def host(L=L, M=M, N=N, tot_steps=tot_steps, beta=beta, N_in=N_in, N_out=N_out, tw=tw, init=init):
    replica_index = int(time())
    str_replica_index = str(replica_index)
    # Parameters for rescaling J
    # Obtain the timestamp list
    start_time_int = int(time())
    data_dir = '../data'
    timestamp_list = list_only_naked_dir(data_dir)

    j = 0
    timestamp = timestamp_list[j] # j is a index, but this index should given by job.sh
    str_timestamp = str(timestamp)
    
    print("L={}".format(L))
    print("M={}".format(M))
    print("N={}".format(N))
    print("N_in={}".format(N_in))
    print("N_out={}".format(N_out))
    print("tot_steps={}".format(tot_steps))

    # Initilize an instance of network.
    o = Network(init,tw,L,M,N,N_in,N_out,tot_steps,beta,timestamp)
    # Load data 
    o.S_in = np.load('../data/{:s}/seed_S_in_M{:d}_N_in{:d}_beta{:4.2f}.npy'.format(str_timestamp,M,N_in,beta))
    o.S_out = np.load('../data/{:s}/seed_S_out_M{:d}_N_out{:d}_beta{:4.2f}.npy'.format(str_timestamp,M,N_out,beta))
    print("shape of S_in:")
    print(o.S_in.shape)
    #========================================================================================================================================================
    # Import the nodes and bonds at t=tw. Motivation: make sure the initial configuraton of the host machine is the same as the guest machines.
    o.J_in = np.load('../data/{:s}/seed_J_in_{:s}_init{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}.npy'.format(str_timestamp,str_timestamp,init,tw,N,N_in,beta))
    o.J_out = np.load('../data/{:s}/seed_J_out_{:s}_init{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}.npy'.format(str_timestamp,str_timestamp,init,tw,N_out,N,beta))
    o.J_hidden = np.load('../data/{:s}/seed_J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(str_timestamp,str_timestamp,init,tw,L,N,beta))
    o.S = np.load('../data/{:s}/seed_S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(str_timestamp,str_timestamp,init,tw,L,M,N,beta))
    #=========================================================================================================================================================
    print("tw (host):")
    print(tw)

    o.tot_steps = tot_steps

    #=======================================================================================
    # Define some variables 
    # Ref: He Yujian's book, Fig. 3.2, m-layer network; Yoshino2020, Fig.1. L-layer network.
    # We ASSUME that each hidden layer has N neurons.
    #=======================================================================================
    print("o.S = ")
    print(o.S)
    print("o.J_hidden = ")
    print(o.J_hidden)
    o.new_S = copy.copy(o.S) # for storing temperay array when update
    o.new_J_hidden = copy.copy(o.J_hidden) # for storing temperay array when update
    o.new_J_in = copy.copy(o.J_in) # for storing temperay array when update
    o.new_J_out = copy.copy(o.J_out) # for storing temperay array when update 

    o.r_hidden = o.gap_hidden_init() # The initial gap
    o.r_in = o.gap_in_init() # The initial gap
    o.r_out = o.gap_out_init() # The initial gap
    
    # We do not need to define o.S_in_traj_hyperfine or o.S_out_traj_hyperfine.
    o.J_in_traj_hyperfine[0,:,:] = o.J_in # The shape of J_in : (N, N_in) 
    o.S_traj_hyperfine[0,:,:,:] = o.S # Note that self.S_traj will independent of self.S from now on. This o.S is the state of S at the end of last epoch of training
    o.J_hidden_traj_hyperfine[0,:,:,:] = o.J_hidden # This o.J is the state of J at the end of last epoch of training
    o.J_out_traj_hyperfine[0,:,:] = o.J_out # The shape of o.J_out:  (N_out, N) 
    
    o.H_hidden = calc_ener(o.r_hidden) # The energy
    o.H_in = calc_ener(o.r_in) # The energy
    o.H_out = calc_ener(o.r_out) # The energy

    o.H_hidden_traj_hyperfine[1] = o.H_hidden # H_traj[0] will be neglected
    
    # Run MC on the host machine
    o.mc_main_random_update_hyperfine_2(str_timestamp)        

    # Save the state of S and J at this end of the epoch of training
    np.save('../data/{:s}/S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(str_timestamp,str_replica_index,o.init,o.tw,o.L,o.M,o.N,o.beta,tot_steps),o.S_traj_hyperfine)
    np.save('../data/{:s}/J_in_{:s}_init{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}_step{:d}.npy'.format(str_timestamp,str_replica_index,o.init,o.tw,o.N,o.N_in,o.beta,tot_steps),o.J_in_traj_hyperfine)
    np.save('../data/{:s}/J_out_{:s}_init{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(str_timestamp,str_replica_index,o.init,o.tw,o.N_out,o.N,o.beta,tot_steps),o.J_out_traj_hyperfine)
    np.save('../data/{:s}/J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(str_timestamp,str_replica_index,o.init,o.tw,o.L,o.N,o.beta,tot_steps),o.J_hidden_traj_hyperfine)
    np.save('../data/{:s}/ener_new_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(str_timestamp,str_replica_index,o.init,o.tw,o.L,o.M,o.N,o.beta,tot_steps),o.H_hidden_traj_hyperfine)
    
    #====================================================
    # Test: see how large the delta_e for each flip/shift
    #====================================================
    o.ave_energy_diff_induced_by_flip()
    o.ave_energy_diff_induced_by_shift()

    #=========
    # Finished
    #=========
    print("MC simulations (host) done!")

def guest(L=L, M=M, N=N, tot_steps=tot_steps, beta=beta, N_in=N_in, N_out=N_out, tw=tw, init=init):
    replica_index = int(time())
    str_replica_index = str(replica_index)
    MC_index = 0
    # Obtain the timestamp list
    start_time_int = int(time())
    data_dir = '../data'
    timestamp_list = list_only_naked_dir(data_dir)

    j = 0
    timestamp = timestamp_list[j] # j=0 is a index, means there is only one timestamp
    str_timestamp = str(timestamp)

    # Initilize an instance of network.
    o = Network(init,tw,L,M,N,N_in,N_out,tot_steps,beta,timestamp)
    # Load data
    o.S_in = np.load('../data/{:s}/seed_S_in_M{:d}_N_in{:d}_beta{:4.2f}.npy'.format(str_timestamp,M,N_in,beta))
    o.S_out = np.load('../data/{:s}/seed_S_out_M{:d}_N_out{:d}_beta{:4.2f}.npy'.format(str_timestamp,M,N_out,beta))
    print("shape of S_in:")
    print(o.S_in.shape)
    #==========================================================================================================================================================
    # Import the nodes and bonds at t=tw. Motivation: make sure the initial configuraton of all machine is the same as the other machines.
    o.S = np.load('../data/{:s}/seed_S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(str_timestamp,str_timestamp,init,tw,L,M,N,beta))
    o.J_in = np.load('../data/{:s}/seed_J_in_{:s}_init{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}.npy'.format(str_timestamp,str_timestamp,init,tw,N,N_in,beta))
    o.J_out = np.load('../data/{:s}/seed_J_out_{:s}_init{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}.npy'.format(str_timestamp,str_timestamp,init,tw,N_out,N,beta))
    o.J_hidden = np.load('../data/{:s}/seed_J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(str_timestamp,str_timestamp,init,tw,L,N,beta))
    #==========================================================================================================================================================

    #=======================
    # Define some parameters 
    # Ref: He Yujian's book, Fig. 3.2, m-layer network; Yoshino2020, Fig.1. L-layer network.
    # We ASSUME that each hidden layer has N neurons.
    o.new_S = copy.copy(o.S) # for storing temperay array when update
    o.new_J_hidden = copy.copy(o.J_hidden) # for storing temperay array when update
    o.new_J_in = copy.copy(o.J_in) # for storing temperay array when update
    o.new_J_out = copy.copy(o.J_out) # for storing temperay array when update 

    o.r_hidden = o.gap_hidden_init() # The initial gap
    o.r_in = o.gap_in_init() # The initial gap
    o.r_out = o.gap_out_init() # The initial gap
    
    o.S_traj_hyperfine[0,:,:,:] = o.S # Note that self.S_traj will independent of self.S from now on. This o.S is the state of S at the end of last epoch of training
    o.J_hidden_traj_hyperfine[0,:,:,:] = o.J_hidden # This o.J is the state of J at the end of last epoch of training
    o.J_in_traj_hyperfine[0,:,:] = o.J_in # The shape of J_in : (N, N_in) 
    o.J_out_traj_hyperfine[0,:,:] = o.J_out # The shape of o.J_out:  (N_out, N) 
    
    o.H_hidden = calc_ener(o.r_hidden) # The energy
    o.H_in = calc_ener(o.r_in) # The energy
    o.H_out = calc_ener(o.r_out) # The energy

    #o.H_in_traj[1] = o.H_in # H_traj[0] will be neglected
    o.H_hidden_traj_hyperfine[1] = o.H_hidden # H_traj[0] will be neglected
    #o.H_out_traj[1] = o.H_out # H_traj[0] will be neglected

    # Run MC simulations
    o.mc_random_update_hyperfine_2(str_timestamp) 
    # Save the state of S and J at this end of the epoch of training
    np.save('../data/{:s}/S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(str_timestamp,str_replica_index,o.init,o.tw,o.L,o.M,o.N,o.beta,tot_steps),o.S_traj_hyperfine)
    np.save('../data/{:s}/J_in_{:s}_init{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}_step{:d}.npy'.format(str_timestamp,str_replica_index,o.init,o.tw,o.N,o.N_in,o.beta,tot_steps),o.J_in_traj_hyperfine)
    np.save('../data/{:s}/J_out_{:s}_init{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(str_timestamp,str_replica_index,o.init,o.tw,o.N_out,o.N,o.beta,tot_steps),o.J_out_traj_hyperfine)
    np.save('../data/{:s}/J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(str_timestamp,str_replica_index,o.init,o.tw,o.L,o.N,o.beta,tot_steps),o.J_hidden_traj_hyperfine)
    np.save('../data/{:s}/ener_new_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(str_timestamp,str_replica_index,o.init,o.tw,o.L,o.M,o.N,o.beta,tot_steps),o.H_hidden_traj_hyperfine)

    #====================================================
    # Test: see how large the delta_e for each flip/shift
    #====================================================
    o.ave_energy_diff_induced_by_flip()
    o.ave_energy_diff_induced_by_shift()
 
    #=========
    # Finished
    #=========
    print("MC simulations (guest) done!")

def init_clean(L=L, M=M, N=N, tot_steps=tot_steps, beta=beta, N_in=N_in, N_out=N_out, tw=tw, init=init):
    """ """
    timestamp = int(time())
    print("starting time:{}".format(timestamp))
    # In general, the input layer and output layer is NOT of the same size of the hidden layers
    # Assumption: layer=0 denotes input layer; layer=L denotes the output layer; layer=1,..., L-1 denotes the hidden layers. 
    # Assumption: The output layer has N_out nodes. J_out has N * N_out bonds. 
    # We need to define two extra arrays for the input layer J_in and the output layer J_out
    J_in = np.zeros((N,N_in))
    J_out = np.zeros((N_out,N))

    # Initilize an instance of network.
    o = HostNetwork(init,tw,L,M,N,N_in,N_out,tot_steps,beta,timestamp)
    # Load data from tf.keras
    o.S_in,o.S_out = generate_S_in_and_out(init,M,N_in,N_out)
    #=====================================
    # Make a new directory named timestamp
    #=====================================
    str_timestamp = str(timestamp)
    list_dir = ['../data/', str_timestamp]
    data = "../data"
    name_dir = "".join(list_dir)
    #==========================
    # Create directory name_dir
    #==========================
    os.makedirs(name_dir,exist_ok=True)

    src_dir = os.path.dirname(__file__) # <-- absolute dir where the script is in

    ##=========================================
    ## Save the arrays in the created directory
    ##=========================================
    # Save the initial configures:
    np.save('{}/{:s}/seed_S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(data,str_timestamp,str_timestamp,init,tw,L,M,N,beta),o.S)
    np.save('{}/{:s}/seed_J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(data,str_timestamp,str_timestamp,init,tw,L,N,beta),o.J_hidden)
    np.save('{}/{:s}/seed_J_in_{:s}_init{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}.npy'.format(data,str_timestamp,str_timestamp,init,tw,N,N_in,beta),o.J_in)
    np.save('{}/{:s}/seed_J_out_{:s}_init{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}.npy'.format(data,str_timestamp,str_timestamp,init,tw,N_out,N,beta),o.J_out)
    #ASSUME that S_in and S_out are fixed during the training. They are independent on the initial configurations (init). (For simplicity)
    np.save('{}/{:s}/seed_S_in_M{:d}_N_in{:d}_beta{:4.2f}.npy'.format(data,str_timestamp,M,N_in,beta),o.S_in)
    np.save('{}/{:s}/seed_S_out_M{:d}_N_out{:d}_beta{:4.2f}.npy'.format(data,str_timestamp,M,N_out,beta),o.S_out)

    print('Initial configurations are generated.')
    

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-B', nargs='?', const=beta, type=float, default=beta, \
                        help="the inverse temperautre.")
    parser.add_argument('-C', nargs='?', const=init, type=int, default=init, \
                        help="the index of initial configurations.  (the initial Conifiguration index)")
    #parser.add_argument('-D', nargs='?', const=tw, type=int, default=tw, \
    #                    help="the waiting time. (the time Delay)")
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
    parser.add_argument('-S', nargs='?', const=tot_steps, type=int, default=tot_steps, \
                        help="the number of total steps.")
    args = parser.parse_args()
    M,L,N,beta,tot_steps,init = args.M,args.L,args.N,args.B,args.S,args.C
    N_in,N_out = args.I,args.J
    
    init_clean(L,M,N,tot_steps, beta, N_in, N_out, tw, init) 

    #================================================ 
    #Use Multiprocessing to run MC on multiple cores
    #================================================ 
    start_t = datetime.datetime.now()
    num_tw = len(tw_list)
    num_cores = int(mp.cpu_count()) 
    n_replica = int(num_cores / num_tw)
 
    print("Number of waiting times:{}. They are:".format(num_tw))
    print(tw_list)
    print("Number of replicas:{}.".format(n_replica))
    print("The computer has " + str(num_cores) + " cores.")

    param_tuple = [(L,M,N,tot_steps, beta, N_in, N_out, tw_list[0], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[0], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[0], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[0], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[0], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[0], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[0], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[0], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[1], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[1], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[1], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[1], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[1], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[1], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[1], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[1], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[2], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[2], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[2], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[2], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[2], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[2], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[2], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[2], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[3], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[3], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[3], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[3], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[3], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[3], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[3], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[3], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[4], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[4], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[4], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[4], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[4], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[4], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[4], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[4], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[5], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[5], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[5], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[5], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[5], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[5], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[5], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[5], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[6], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[6], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[6], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[6], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[6], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[6], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[6], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[6], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[7], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[7], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[7], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[7], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[7], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[7], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[7], init),
                   (L,M,N,tot_steps, beta, N_in, N_out, tw_list[7], init)]

    # MC simulations for tw as host
    print("Now start process ({}).".format(0))
    mp.Process(target=host, args=param_tuple[0]).start() #start now
    sleep(wait0)
    print("host is running and some seeds are prepared after {} seconds!".format(wait0))

    # MC simulations for other tw's as gustes
    for k in range(1, num_cores):
        print("Now start process ({}).".format(k))
        mp.Process(target=guest, args=param_tuple[k]).start() #start now
        sleep(2)
    print("DONE!")
