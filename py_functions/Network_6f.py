#======
#Module
#======
import sys
sys.path.append('/public1/home/sc91981/py_functions/')
#sys.path.append('/home/gang/Github/Yoshino/py_functions/')

from random import choice
import copy
from functools import wraps
import math
import numpy as np
from scipy.stats import norm
import os
import matplotlib.pyplot as plt
from random import randrange
import scipy as sp
from time import time
from utilities import *

l_list = [1,2,3,4,5,6,7,8]
l_list_all = [0,1,2,3,4,5,6,7,8,9]
l_index_list = [0,1,2,3,4,5,6,7]
l_S_list = [1,2,3,4,5,6,7,8]
step_list = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728]

tw_list = [0, 4096, 8192, 65536, 262144, 524288, 1048576, 2097152]
tw_list_M20 = [0, 4096, 8192, 65536, 262144, 524288, 1048576, 2097152]
tw_list_M40 = [0, 4096, 8192, 65536, 262144, 524288, 1048576, 2097152]

M_list = [10,20,30,40,50,60,70,80,90]
alpha_list = [1,2,3,4,5,6,7,8,9] # These values are from alpha=M/N, where M = 10,20,40,80,160, N=5.

def timethis(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        start = time()
        result = fun(*args,**kwargs)
        end = time()
        print(fun.__name__, end-start)
        return result

    return wrapper

#class Network:
#    def __init__(self,init,tw,L,M,N,N_in,N_out,tot_steps,beta,timestamp):
#        """Since Yoshino_3.0, when update the energy, we do not calculate all the gaps, but only calculate the part affected by the flip of a SPIN (S)  or a shift of
#           a BOND (J). This will accelerate the calculation by hundreds of times. (2) Besides, we note that we do NOT need to define a functon: remain(), which records
#           the new MC steps' S, J and H, even though one MC move is rejected."""
#        # Parameters used in the host machine (No. 0)
#        self.init = int(init)
#        self.tw = int(tw)
#        self.L = int(L)
#        self.M = int(M)
#        self.N = int(N)
#        self.N_in = int(N_in)
#        self.N_out = int(N_out)
#        self.tot_steps = int(tot_steps)
#        self.beta = beta
#        self.timestamp = timestamp
#        self.num_hidden_node_layers = self.L - 1 # After distingush num_hidden_node_layers and num_hidden_bond_layers, then I found the clue is much clear!
#        self.num_hidden_bond_layers = self.L - 2
#        # Define new parameters; T (technically required, to save memory)
#        self.BIAS = 1 # For avoiding x IS NOT EQUAL TO ZERO in log2(x)
#        self.BIAS2 = 5 # For obtaing long enough list list_k.
#        self.T = int(np.log2(self.tot_steps+self.BIAS)) # we keep the initial state in the first step
#        #self.L_hidden = self.L - 2 (DATED)
#
#        self.H = 0 # for storing energy
#        self.new_H = 0 # for storing temperay energy when update
#
#        # Energy difference caused by update of sample mu
#        self.delta_H = 0
#
#        # Intialize S,J_hidden,J_in and J_out by the saved arrays, which are saved from the host machine. 
#        # We use S, instead of S_hidden, because S[0], S[L-2] still have interaction to J_in and J_out. Ref. Yoshino eq (8).
#        data_dir = '../data'
#        str_timestamp = str(timestamp)
#        self.S = np.load('../data/{:s}/seed_S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:3.1f}.npy'.format(str_timestamp,str_timestamp,init,tw,L,M,N,beta))
#        self.J_hidden = np.load('../data/{:s}/seed_J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:3.1f}.npy'.format(str_timestamp,str_timestamp,init,tw,L,N,beta))
#        self.J_in = np.load('{}/{:s}/seed_J_in_{:s}_init{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:3.1f}.npy'.format(data_dir, str_timestamp, str_timestamp, init, tw, N, N_in, beta))
#        self.J_out = np.load('{}/{:s}/seed_J_out_{:s}_init{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:3.1f}.npy'.format(data_dir,str_timestamp,str_timestamp,init,tw,N_out,N,beta))
#        
#        print("self.S: shape")
#        print(self.S.shape)
#        print("self.J_hidden: shape")
#        print(self.J_hidden.shape)
#        print("self.J_in: shape")
#        print(self.J_in.shape)
#        print("self.J_out :shape")
#        print(self.J_out.shape)
#        # Define J_in and J_out
#        self.S_in = np.zeros((self.M,self.N_in))  
#        self.S_out = np.zeros((self.M,self.N_out))  
#        print("self.S_in shape:")
#        print(self.S_in.shape)
#        print("self.S_out shape")
#        print(self.S_out.shape)
#
#        self.new_S = 0 # for storing temperay array when update
#        self.new_J = 0 # for storing temperay array when update
#
#        self.new_J_in = np.zeros((self.N,self.N_in)) 
#        self.new_J_out = np.zeros((self.N_out,self.N))  
#      
#        # Initialize the inner parameters: num_bonds, num_variables
#        self.SQ_N = (self.N) ** 2
#        num_variables = self.N * self.M * self.num_hidden_node_layers 
#        self.T = int(np.log2(self.tot_steps+self.BIAS)) # we keep the initial state in the first step
#        num_bonds = self.N * self.N_in + self.SQ_N * self.num_hidden_bond_layers + self.N_out * self.N
#
#        print("self.num_hidden_node_layers:")
#        print(self.num_hidden_node_layers)
#        print("self.T:")
#        print(self.T)
#
#        self.num_variables = int(num_variables) 
#        self.num_bonds = int(num_bonds)
#        self.num = self.num_variables + self.num_bonds
#
#        print("self.num_bonds:")
#        print(self.num_bonds)
#
#        self.ind_save = 0
#        self.count_MC_step = 0 
#
#        self.T_2 = int(np.log2(self.tot_steps * self.num + self.BIAS)) # we keep the initial state in the first step
#        #======================================================================================================================================== 
#        # The arrays for storing MC trajectories of S, J and H (hyperfine)
#        # Note that we DO NOT nedd to define arrays self.S_in_traj_hyperfine, or S_out_traj_hyperfine, because self.S_in and self.S_out are fixed.
#        self.J_hidden_traj_hyperfine = np.zeros((self.T_2, self.num_hidden_bond_layers, self.N, self.N), dtype='float32') 
#        self.S_traj_hyperfine = np.zeros((self.T_2, self.M, self.num_hidden_node_layers, self.N), dtype='int8')
#        self.H_hidden_traj_hyperfine = np.zeros(self.T_2, dtype='float32') 
#        self.J_in_traj_hyperfine = np.zeros((self.T_2,self.N,self.N_in), dtype='float32')
#        self.J_out_traj_hyperfine = np.zeros((self.T_2,self.N_out,self.N), dtype='float32')
#        #======================================================================================================================================== 
#
#        # DEFINE SOME PARAMETERS
#        self.EPS = 1e-10
#        self.RAT = 0.1 # r: Yoshino2019 Eq(35)
#        self.RESCALE_J = 1.0/np.sqrt(1 + self.RAT**2)
#        self.SQRT_N = np.sqrt(self.N)
#        self.SQRT_N_IN = np.sqrt(self.N_in)
#        self.PROB = self.num_variables/self.num 
#        self.cutoff1 = self.N*self.N_in
#        self.cutoff2 = self.cutoff1 + self.num_hidden_bond_layers * self.SQ_N
#        #print("self.RESCALE_J:")
#        #print(self.RESCALE_J)
#        #print("self.SQRT_N:")
#        #print(self.SQRT_N)
#        #print("self.SQRT_N_IN:")
#        #print(self.SQRT_N_IN)
#        #print("self.cutoff1")
#        #print(self.cutoff1)
#        #print("self.cutoff2")
#        #print(self.cutoff2)
#
#        # EXAMPLE: list_k = [4,8,16,64,256,1024,4096,16384,65536]
#        self.list_k = [2**(i) for i in range(int(np.log2(self.tot_steps+self.BIAS) + self.BIAS2))]
#        self.list_k_4_hyperfine = [2**(i) for i in range(int(np.log2(self.tot_steps * self.num + self.BIAS) + self.BIAS2))]
#
#    def gap_in_init(self):
#        '''Ref: Yoshino2019, eqn (31b)'''
#        M = self.M
#        N = self.N
#        N_in = self.N_in
#        r_in = np.zeros((M,N),dtype='float32')
#        for mu in range(M):
#            for n2 in range(N):
#                r_in[mu,n2] = (np.sum(self.J_in[n2,:] * self.S_in[mu,:])/self.SQRT_N_IN) * self.S[mu,0,n2]
#        return r_in
#
#    def gap_hidden_init(self):
#        '''Ref: Yoshino2019, eqn (31b)'''
#        L = self.L
#        M = self.M
#        N = self.N
#        r_hidden = np.zeros((M,L,N),dtype='float32')
#        for mu in range(M):
#            for l in range(self.num_hidden_bond_layers): # l = 2,...,L-1
#                index_node_layer = l # Distinguish different index for S and r is important!
#                for n2 in range(N):
#                    r_hidden[mu,l,n2] = (np.sum(self.J_hidden[l,n2,:] * self.S[mu,index_node_layer,:])/self.SQRT_N) * self.S[mu,index_node_layer + 1, n2]
#        return r_hidden
#
#    def gap_out_init(self):
#        M = self.M
#        N = self.N
#        N_out = self.N_out
#        r_out = np.zeros((M,N_out),dtype='float32')
#        for mu in range(M):
#            for n2 in range(N_out):
#                r_out[mu,n2] = (np.sum(self.J_out[n2,:] * self.S[mu,-1,:])/self.SQRT_N) * self.S_out[mu,n2]
#        return r_out
#
#    def flip_spin(self,mu,l,n):
#        '''flip_spin() will flip S at a given index tuple (l,mu,n). We add l,mu,n as parameters, for parallel programming. Note: any spin can be update except the input/output.'''
#        # Update self.new_S
#        self.new_S = copy.copy(self.S)
#        self.new_S[mu][l][n] = -1 * self.S[mu][l][n]
#
#    def random_update_J(self):
#        """Method 1"""
#        # Const.s
#        x = np.random.normal(loc=0,scale=1.0,size=None)
#        x = round(x,10)
#        EPS = self.EPS
#        SQ_N = (self.N) ** 2
#        cutoff1 = self.cutoff1        
#        cutoff2 = self.cutoff2        
#        P1 = cutoff1/self.num_bonds
#        P2 = cutoff2/self.num_bonds
#        rand1, rand2 = np.random.random(1), np.random.random(1)
#
#        if rand1 < P1:
#            n2,n1=randrange(self.N),randrange(self.N_in)
#            # Update J_in[n2,n1]
#            self.shift_bond_in(n2,n1,x)
#            self.decision_bond_in_by_n2_n1(n2,n1,EPS,rand2)
#        elif rand1 > P1 and rand1 < P2:
#            l,n2,n1 = randrange(self.num_hidden_bond_layers),randrange(self.N),randrange(self.N) 
#            self.shift_bond_hidden(l,n2,n1,x)
#            self.decision_bond_hidden_by_l_n2_n1(l,n2,n1,EPS,rand2)
#        else:
#            n2,n1=randrange(self.N_out),randrange(self.N)
#            self.shift_bond_out(n2,n1,x)
#            self.decision_bond_out_by_n2_n1(n2,n1,EPS,rand2)
#    def random_update_J_method_V2(self):
#        """Method 2"""
#        # Generate a random integer
#        index_J = randrange(self.num_bonds)
#        if index_J < self.cutoff1:
#            n2,n1 = index_J//self.N_in, index_J%self.N_in
#            # Update J_in[n2,n1]
#            self.shift_bond_in(n2,n1,x)
#            self.decision_bond_in_by_n2_n1(n2,n1,EPS,np.random.random(1))
#        elif index_J > self.cutoff1-1 and index_J < cutoff2:
#            index_J_2 = index_J - self.cutoff1
#            l,n2,n1 = index_J_2//SQ_N, (index_J_2%SQ_N)//self.N, (index_J_2%SQ_N)%self.N 
#            self.shift_bond_hidden(l,n2,n1,x)
#            self.decision_bond_hidden_by_l_n2_n1(l,n2,n1,EPS,np.random.random(1))
#        elif index_J > cutoff2-1:
#            index_J_3 = index_J - cutoff2
#            n2,n1 = index_J_3//self.N, index_J_3%self.N
#            self.shift_bond_out(n2,n1,x)
#            self.decision_bond_out_by_n2_n1(n2,n1,EPS,np.random.random(1))
#        else:
#            pass
#
#    def shift_bond_hidden(self,l,n2,n1,x):
#        '''shift_bond_hidden() will shift the element of J with a given index to another value. We add l,n2,n1 as parameters, for parallel programming..'''
#        self.new_J_hidden = copy.copy(self.J_hidden)
#        N = self.N
#        # scale denotes standard deviation
#        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
#        self.new_J_hidden[l][n2][n1] = (self.J_hidden[l][n2][n1] + x * self.RAT) * self.RESCALE_J
#        # rescaling 
#        t = self.new_J_hidden[l][n2] 
#        N_prim = np.sum(t*t)
#        SCALE = np.sqrt(N / N_prim)
#        self.new_J_hidden[l][n2] = self.new_J_hidden[l][n2] * SCALE
#
#    def shift_bond_in(self,n2,n1,x):
#        '''shift_bond_in() will shift the element of J_in with a given index to another value. We add n2,n1 as parameters, for parallel programming..'''
#        self.new_J_in = copy.copy(self.J_in)
#        N_in = self.N_in
#        N = self.N
#        # scale denotes standard deviation
#        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
#        self.new_J_in[n2,n1] = (self.J_in[n2,n1] + x * self.RAT) * self.RESCALE_J
#        # step2:rescaling 
#        t = self.new_J_in[n2]  
#        N_prim = np.sum(t*t)
#        SCALE = np.sqrt(N_in / N_prim)
#        self.new_J_in[n2] = self.new_J_in[n2] * SCALE
#    def shift_bond_out(self,n2,n1,x):
#        '''shift_bond_out() will shift the element of J_out with a given index to another value. We add n2,n1 as parameters, for parallel programming..'''
#        self.new_J_out = copy.copy(self.J_out)
#        N = self.N
#        # scale denotes standard deviation
#        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
#        self.new_J_out[n2,n1] = (self.J_out[n2,n1] + x * self.RAT) * self.RESCALE_J
#        # rescaling 
#        t = self.new_J_out[n2] 
#        N_prim = np.sum(t*t) # Use sort, because we want to avoid the larger values 'eats' the smaller ones. But I do not need to use sort in np.sum(), I believe.
#        SCALE = np.sqrt(N / N_prim)
#        #print("The sum of J_ij^2:{}: [OLD] ".format(N_prim))
#        self.new_J_out[n2] = self.new_J_out[n2] * SCALE
#    
#    # The following accept function is used if S is flipped.
#    def accept_by_mu_l_n(self,mu,l,n):
#        """This accept function is used if S is flipped."""
#        self.S[mu,l,n] = self.new_S[mu,l,n]
#        self.H = self.H + self.delta_H
#    
#    # One of the following accept functions is used if J is shifted.
#    def accept_bond_hidden_by_l_n2_n1(self,l,n2,n1):
#        self.J_hidden[l,n2,n1] = self.new_J_hidden[l,n2,n1]
#        self.H = self.H + self.delta_H
#    def accept_bond_in_by_n2_n1(self,n2,n1):
#        self.J_in[n2,n1] = self.new_J_in[n2,n1]
#        self.H = self.H + self.delta_H
#    def accept_bond_out_by_n2_n1(self,n2,n1):
#        self.J_out[n2,n1] = self.new_J_out[n2,n1]
#        self.H = self.H + self.delta_H
#    
#    # One of the gap function is used if S is flipped.
#    def part_gap_hidden_before_flip(self,mu,l,n):
#        '''l: index for hidden layers of the node (hidden S), l = 1, 2,...,L-3.
#           Ref: Yoshino2019, eqn (31b)
#           When hidden S is fliped, only one machine changes its coordinates and it will affect the gap of the node before it and the gaps of the N nodes
#           behind it. Therefore, N+1 gaps contributes to the Delta_H_eff.
#           We define a small array, part_gap, which has N+1 elements. Each elements of part_gap is a r^mu_node. Use part_gap, we can calculate the
#           Energy change coused by the flip of S^mu_node,n.
#        '''
#        N = self.N
#        SQRT_N = self.SQRT_N
#        part_gap = np.zeros(N + 1,dtype='float32')
#
#        part_gap[0] = (np.sum(self.J_hidden[l-1,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n]
#        for n2 in range(N):
#            part_gap[1+n2] = (np.sum(self.J_hidden[l,n2,:] * self.S[mu,l,:])/SQRT_N) * self.S[mu,l+1,n2]
#        return part_gap 
#    def part_gap_hidden_after_flip(self,mu,l,n):
#        """ l: index for layer of the node. l = 1, 2,...,L-3
#            Distinguish different index for S and r is important!
#        """
#        N = self.N
#        SQRT_N = self.SQRT_N
#        part_gap = np.zeros(N + 1,dtype='float32')
#
#        part_gap[0] = (np.sum(self.J_hidden[l-1,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.new_S[mu,l,n]
#        for n2 in range(N):
#            part_gap[1+n2] = (np.sum(self.J_hidden[l,n2,:] * self.new_S[mu,l,:])/SQRT_N) * self.S[mu,l+1,n2]
#        return part_gap 
#    def part_gap_in_before_flip(self,mu,n):
#        '''If a spin in the first layer flips, then r_in will change.
#        '''
#        N = self.N
#        SQRT_N = self.SQRT_N
#        SQRT_N_IN = self.SQRT_N_IN
#        part_gap = np.zeros(N + 1,dtype='float32')
#
#        # effects on previous gap (only 1 gap is affected)
#        part_gap[0] = (np.sum(self.J_in[n,:] * self.S_in[mu,:])/SQRT_N_IN) * self.S[mu,0,n]
#        # effects on the N gaps in the next layer. Remember the assumption: in hidden layers, each layer has N nodes.
#        index_hidden_bond_layer = 0
#        index_hidden_node_layer = index_hidden_bond_layer
#        for n2 in range(N):
#            part_gap[1+n2] = (np.sum(self.J_hidden[index_hidden_bond_layer,n2,:] * self.S[mu,index_hidden_node_layer,:])/SQRT_N) * self.S[mu,index_hidden_node_layer+1,n2]
#        return part_gap  # Only the N+1 elements affect the Delta_H_eff.
#    def part_gap_in_after_flip(self,mu,n):
#        N = self.N
#        SQRT_N = self.SQRT_N
#        SQRT_N_IN = self.SQRT_N_IN
#        part_gap = np.zeros(N + 1,dtype='float32')
#
#        # effects on previous gap (only 1 gap is affected)
#        part_gap[0] = (np.sum(self.J_in[n,:] * self.S_in[mu,:])/SQRT_N_IN) * self.new_S[mu,0,n]
#        # effects on the N gaps in the next layer. Remember the assumption: in hidden layers, each layer has N nodes.
#        index_hidden_bond_layer = 0
#        index_hidden_node_layer = index_hidden_bond_layer
#        for n2 in range(N):
#            part_gap[1+n2] = (np.sum(self.J_hidden[index_hidden_bond_layer,n2,:] * self.new_S[mu,index_hidden_node_layer,:])/SQRT_N) * self.S[mu,index_hidden_node_layer+1,n2]
#        return part_gap  # Only the N+1 elements affect the Delta_H_eff.
#    def part_gap_out_before_flip(self,mu,n):
#        ''' If a spin in the last hidden layer flips, then r_out will change.
#        '''
#        N = self.N
#        N_out = self.N_out
#        SQRT_N = self.SQRT_N
#        part_gap = np.zeros(N_out + 1, dtype='float32')
#
#        part_gap[0] = (np.sum(self.J_hidden[-1,n,:] * self.S[mu,-2,:]/SQRT_N)) * self.S[mu,-1,n]
#        for n2 in range(N_out):
#            part_gap[1+n2] = (np.sum(self.J_out[n2,:] * self.S[mu,-1,:]/SQRT_N)) * self.S_out[mu,n2]
#        return part_gap  # Only (N_out)+1 gaps affect the Delta_H_eff.
#    def part_gap_out_after_flip(self,mu,n):
#        ''' If a spin in the last hidden layer flips, then r_out will change.
#        '''
#        L = self.L
#        M = self.M
#        N_out = self.N_out
#        SQRT_N = self.SQRT_N
#        part_gap = np.zeros(N_out + 1,dtype='float32')
#
#        part_gap[0] = (np.sum(self.J_hidden[-1,n,:] * self.S[mu,-2,:])/SQRT_N) * self.new_S[mu,-1,n]
#        for n2 in range(N_out):
#            part_gap[1+n2] = (np.sum(self.J_out[n2,:] * self.new_S[mu,-1,:])/SQRT_N) * self.S_out[mu,n2]
#        return part_gap # Only (N_out)+1 gaps affect the Delta_H_eff.
#
#    # One of the gap function is used if J is shifted.
#    def part_gap_in_before_shift(self,n):
#        L = self.L
#        M = self.M
#        SQRT_N_IN = self.SQRT_N_IN
#        part_gap = np.zeros(M,dtype='float32')
#
#        for mu in range(M):
#            part_gap[mu] = (np.sum(self.J_in[n,:] * self.S_in[mu,:])/SQRT_N_IN) * self.S[mu,0,n]
#        return part_gap  # Only the M elements affect the Delta_H_eff.
#    def part_gap_in_after_shift(self,n):
#        L = self.L
#        M = self.M
#        SQRT_N_IN = self.SQRT_N_IN
#        part_gap = np.zeros(M,dtype='float32')
#
#        for mu in range(M):
#            part_gap[mu] = (np.sum(self.new_J_in[n,:] * self.S_in[mu,:])/SQRT_N_IN) * self.S[mu,0,n]
#        return part_gap 
#    def part_gap_out_before_shift(self,n_out):
#        L = self.L
#        M = self.M
#        SQRT_N = self.SQRT_N
#        part_gap = np.zeros(M,dtype='float32')
#
#        for mu in range(M):
#            part_gap[mu] = (np.sum(self.J_out[n_out,:] * self.S[mu,L-2,:])/SQRT_N) * self.S_out[mu,n_out]
#        return part_gap 
#    def part_gap_out_after_shift(self,n_out):
#        L = self.L
#        M = self.M
#        SQRT_N = self.SQRT_N
#        part_gap = np.zeros(M,dtype='float32')
#
#        for mu in range(M):
#            # For testing
#            #yy = np.sum(self.new_J_out[n_out,:] * self.S[L-2,mu,:])
#            #xx = yy/SQRT_N * self.S[mu,n_out]
#            #print (xx)
#            part_gap[mu] = (np.sum(self.new_J_out[n_out,:] * self.S[mu,self.L-2,:])/SQRT_N) * self.S_out[mu,n_out]
#        return part_gap 
#    def part_gap_hidden_before_shift(self,l,n):
#        M = self.M
#        SQRT_N = self.SQRT_N
#        part_gap = np.zeros(self.M,dtype='float32')
#
#        for mu in range(M):
#            part_gap[mu] = (np.sum(self.J_hidden[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n]
#        return part_gap  # Only the M elements affect the Delta_H_eff.
#    def part_gap_hidden_after_shift(self,l,n):
#        M = self.M
#        SQRT_N = self.SQRT_N
#        part_gap = np.zeros(self.M,dtype='float32')
#
#        for mu in range(M):
#            part_gap[mu] = (np.sum(self.new_J_hidden[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n]
#        return part_gap # Only the M elements affect the Delta_H_eff.
#
#    def rand_index_for_S(self):
#        # For S: list_index_for_S = [(mu,l,n),...]
#        list_index_for_S = []
#        for _ in range(self.num_variables * (self.tot_steps-1)):
#            list_index_for_S.append([randrange(self.M), randrange(1,self.L-1), randrange(self.N)])
#        res_arr = np.array(list_index_for_S)
#        return res_arr
#    def rand_index_for_J(self):
#        # For generating J: list_index_for_J = [(l,n2,n1),...]
#        list_index_for_J = []
#        for _ in range(self.num_bonds * (self.tot_steps-1)):
#            list_index_for_J.append([randrange(1,self.L), randrange(self.N), randrange(self.N)])
#        res_arr = np.array(list_index_for_J)
#        return res_arr
#    def rand_series_for_x(self):
#        """
#        For generating J: list_for_x = [x1,x2,...]
#        We separate rand_index_for_J() and rand_series_for_x(), instead of merginging them to one function and return a list of four-tuple (l,n2,n1,x).
#        The reason is: x is float and l,n2,n1 are integers, it will induce trouble if one put them (x and l,n2,n1 ) together.
#        """
#        list_for_x = []
#        for _ in range(self.num_bonds * (self.tot_steps-1)):
#            x = np.random.normal(loc=0,scale=1.0,size=None)
#            x = round(x,10)
#            list_for_x.append(x)
#        res_arr = np.array(list_for_x)
#        return res_arr
#
#    def rand_series_for_decision_on_S(self):
#        # For generating J: list_index_for_J = [(l,n2,n1),...]
#        list_for_decision = []
#        for _ in range(self.num_variables * (self.tot_steps-1)):
#            list_for_decision.append(np.random.random(1))
#        res_arr = np.array(list_for_decision)
#        return res_arr
#    def rand_series_for_decision_on_J(self):
#        # For generating J: list_index_for_J = [(l,n2,n1),...]
#        list_for_decision = []
#        for _ in range(self.num_bonds * (self.tot_steps-1)):
#            list_for_decision.append(np.random.random(1))
#        res_arr = np.array(list_for_decision)
#        return res_arr
#
#    def check_and_save_seeds(self,str_timestamp):
#        ind = self.count_MC_step
#        init = self.init
#        if ind==self.list_k[0]:
#            self.list_k.pop(0) 
#            np.save('../data/{:s}/seed_S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:3.1f}.npy'.format(str_timestamp,str_timestamp,init,ind,self.L,self.M,self.N,self.beta),self.S)   
#            np.save('../data/{:s}/seed_J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:3.1f}.npy'.format(str_timestamp,str_timestamp,init,ind,self.L,self.N,self.beta),self.J_hidden)
#            np.save('../data/{:s}/seed_J_in_{:s}_init{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:3.1f}.npy'.format(str_timestamp,str_timestamp,init,ind,self.N,self.N_in,self.beta),self.J_in)
#            np.save('../data/{:s}/seed_J_out_{:s}_init{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:3.1f}.npy'.format(str_timestamp,str_timestamp,init,ind,self.N_out,self.N,self.beta),self.J_out)
#            np.save('../data/{:s}/seed_ener_new_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:3.1f}.npy'.format(str_timestamp,str_timestamp,init,ind,self.L,self.M,self.N,self.beta),self.H_hidden)
#        else:
#            pass
#    def check_and_save(self):
#        ind = self.count_MC_step
#        if 2**self.ind_save == ind and self.ind_save < self.S_traj.shape[0]:
#            self.S_traj[self.ind_save] = self.S
#            self.J_hidden_traj[self.ind_save] = self.J_hidden
#            self.H_hidden_traj[self.ind_save] = self.H_hidden
#            self.ind_save += 1
#        else:
#            pass
#    def check_and_save_seeds_hyperfine(self,update_index,str_timestamp):
#        ind = update_index
#        init = self.init
#        if ind==self.list_k_4_hyperfine[0]:
#            self.list_k_4_hyperfine.pop(0) 
#            np.save('../data/{:s}/seed_S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:3.1f}.npy'.format(str_timestamp,str_timestamp,init,ind,self.L,self.M,self.N,self.beta),self.S)   
#            np.save('../data/{:s}/seed_J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:3.1f}.npy'.format(str_timestamp,str_timestamp,init,ind,self.L,self.N,self.beta),self.J_hidden)
#            np.save('../data/{:s}/seed_J_in_{:s}_init{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:3.1f}.npy'.format(str_timestamp,str_timestamp,init,ind,self.N,self.N_in,self.beta),self.J_in)
#            np.save('../data/{:s}/seed_J_out_{:s}_init{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:3.1f}.npy'.format(str_timestamp,str_timestamp,init,ind,self.N_out,self.N,self.beta),self.J_out)
#            np.save('../data/{:s}/seed_ener_new_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:3.1f}.npy'.format(str_timestamp,str_timestamp,init,ind,self.L,self.M,self.N,self.beta),self.H_hidden)
#        else:
#            pass
#    def check_and_save_hyperfine(self,update_index):
#        ind = update_index
#        if (2**self.ind_save) == ind and self.ind_save < self.S_traj_hyperfine.shape[0]:
#            self.S_traj_hyperfine[self.ind_save] = self.S
#            self.J_in_traj_hyperfine[self.ind_save] = self.J_in
#            self.J_out_traj_hyperfine[self.ind_save] = self.J_out
#            self.J_hidden_traj_hyperfine[self.ind_save] = self.J_hidden
#            self.H_hidden_traj_hyperfine[self.ind_save] = self.H_hidden
#            self.ind_save += 1
#        else:
#            pass
#
#    def check_and_save_base10(self):
#        ind = self.count_MC_step
#        if 2**self.ind_save == ind and self.ind_save < self.S_traj.shape[0]:
#            self.S_traj[self.ind_save] = self.S
#            self.J_in_traj[self.ind_save] = self.J_in
#            self.J_out_traj[self.ind_save] = self.J_out
#            self.J_hidden_traj[self.ind_save] = self.J_hidden
#            self.H_hidden_traj[self.ind_save] = self.H_hidden
#            self.ind_save += 1
#        else:
#            pass
#    def check_and_save_hyperfine_base10(self,update_index):
#        ind = update_index
#        if (2**self.ind_save) == ind and self.ind_save < self.S_traj_hyperfine.shape[0]:
#            self.S_traj_hyperfine[self.ind_save] = self.S
#            self.J_in_traj_hyperfine[self.ind_save] = self.J_in
#            self.J_out_traj_hyperfine[self.ind_save] = self.J_out
#            self.H_hidden_traj_hyperfine[self.ind_save] = self.H_hidden
#            self.ind_save += 1
#        else:
#            pass
#
#    @timethis
#    def mc_main(self,str_timestamp,replica_index):
#        """MC for the host machine, i.e., it will save seeds for different waiting time."""
#        str_replica_index = str(replica_index)
#        rel_path='J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:3.1f}_step{:d}_host.dat'.format(str_replica_index,self.init,self.tw,self.L,self.N,self.beta,self.tot_steps)
#        src_dir = os.path.dirname(__file__) 
#        abs_file_path=os.path.join(src_dir, rel_path)
#        #file1 = open(abs_file_path,'w')
#
#        EPS = self.EPS
#        SQ_N = (self.N) ** 2
#        cutoff1 = self.cutoff1        
#        cutoff2 = self.cutoff2        
#        P1 =  cutoff1/self.num_bonds
#        P2 =  cutoff2/self.num_bonds
#         
#        print("list_k:")
#        print(self.list_k)
#        # MC siulation starts
#        for MC_index in range(1,self.tot_steps):
#            #print("Updating S:")
#            for update_index in range(0, self.num_variables):
#                mu,l,n = randrange(self.M),randrange(self.num_hidden_node_layers),randrange(self.N)
#                self.flip_spin(mu,l,n)
#                if l == 0:
#                    self.decision_node_in_by_mu_n(mu,n,EPS,np.random.random(1))
#                elif l == self.num_hidden_node_layers - 1:
#
#                    self.decision_node_out_by_mu_n(mu,n,EPS,np.random.random(1))
#                else:
#                    self.decision_by_mu_l_n(mu,l,n,EPS,np.random.random(1))
#            #print("Updating J:")
#            for update_index in range(0, self.num_bonds):
#                self.random_update_J()
#            self.count_MC_step += 1
#            # Check and save the seeds 
#            # IF MC_index EQUALS 2**k, WHERE k = 1,2,3,4,5,...,12, THEN SAVE THE CONFIGURATION OF THE NDOES AND BONDS. 
#            # THIS OPERATION SHUOLD BE ONLY DONE IN HOST MACHINE, DO NOT DO IT IN A GUEST MACHINE.
#            self.check_and_save_seeds(str_timestamp)
#
#            #Check and save the configureation of bonds 
#            self.check_and_save() # save configuration
#
#            #========================================================================================================
#            # To check if the training dynamics have the aging effect.
#            # The basic idea is: After we start training, we save the configurations of J_in, J_out, J_hidden and S 
#            # (ie., S_hidden) at MC_index = 4, (8,) 16, (32,) 64, (128,) 256, (512,) 1024, (2048,) 4096, (8192.)
#            # For each of these restarting configuration, we run N_replica = 10  dependent training (MC 'dynamics'). 
#            # Each replica trajectory should have a label (a name, ie, 0, 1, 2, 3, ..., 9).
#            # They can be paired to N_replica * (N_replica-1)/2 pairs. In each pair,
#            # one trajectory is palyed the role of 'host_restart_MC_index' and the other is 'guest_restart_MC_index'.
#            # After all these dynamics are obtained, we can calculate Q(t,l), q(t,l) for each trajectory pair.
#            # Similiarly, we can obtain the tau(l) function for each trajectory.
#            # Then we can know if there is aging effect.
#            #
#            # There are totally (12 + 1) * 10 = 130 training dynamics should be generated. If we only run the MC dynamics 
#            # for MC_index = 8, 32, 128, 512, 2048, 8192, we will totally run (1+6)*N_replica = 70 trajectories.
#            # Each trajectory is of the total step 20,000.
#            # Especially, for Initial host.py, we will run guest.py 9 times (N_replica - 1 = 10-1 = 9).
#
#    @timethis
#    def mc_odd(self,replica_index):
#        """MC for guest machines, i.e., it will not save seeds for different waiting time."""
#        str_replica_index = str(replica_index)
#        rel_path='J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:3.1f}_step{:d}_odd.dat'.format(str_replica_index,self.init,self.tw,self.L,self.N,self.beta,self.tot_steps)
#        src_dir = os.path.dirname(__file__) 
#        abs_file_path=os.path.join(src_dir, rel_path)
#        #file1 = open(abs_file_path,'w')
#
#        EPS = self.EPS
#        SQ_N = (self.N) ** 2
#        cutoff1 = self.cutoff1        
#        cutoff2 = self.cutoff2        
#        P1 =  cutoff1/self.num_bonds
#        P2 =  cutoff2/self.num_bonds
#        # MC siulation starts
#        for MC_index in range(1,self.tot_steps):
#            #print("MC step: {:d}".format(MC_index))
#            #print("Updating S:")
#            #for update_index in range((MC_index-1) * self.num_variables, MC_index * self.num_variables):
#            for update_index in range(0, self.num_variables):
#                mu,l,n = randrange(self.M),randrange(self.num_hidden_node_layers),randrange(self.N)
#                self.flip_spin(mu,l,n)
#                if l == 0:
#                    self.decision_node_in_by_mu_n(mu,n,EPS,np.random.random(1))
#                elif l == self.num_hidden_node_layers - 1:
#
#                    self.decision_node_out_by_mu_n(mu,n,EPS,np.random.random(1))
#                else:
#                    self.decision_by_mu_l_n(mu,l,n,EPS,np.random.random(1))
#            #print("Updating J:")
#            #for update_index in range((MC_index-1)*self.num_bonds, MC_index*self.num_bonds):
#            for update_index in range(0, self.num_bonds):
#                #========
#                #Method 1
#                #========
#                self.random_update_J()
#                ##========
#                ##Method 2
#                ##========
#            self.count_MC_step += 1
#            #Check and save the configureation of bonds 
#            self.check_and_save() # save configuration
#
#            #FOR TESTING
#            #file1.write("{}\n".format(str(self.count_MC_step)))
#            #for j in range(self.num_hidden_bond_layers):
#            #    for i in range(self.N):
#            #        for ii in range(self.N):
#            #            file1.write("{}\n".format(str(self.J_hidden[j][i][ii])))
#        #file1.close()
#
#    @timethis
#    def mc_even(self,replica_index):
#        """MC for guest machines, i.e., it will not save seeds for different waiting time."""
#        str_replica_index = str(replica_index)
#        rel_path='J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:3.1f}_step{:d}_even.dat'.format(str_replica_index,self.init,self.tw,self.L,self.N,self.beta,self.tot_steps)
#        src_dir = os.path.dirname(__file__) 
#        abs_file_path=os.path.join(src_dir, rel_path)
#        #file1 = open(abs_file_path,'w')
#
#        EPS = self.EPS
#        SQ_N = (self.N) ** 2
#        cutoff1 = self.cutoff1        
#        cutoff2 = self.cutoff2        
#        P1 =  cutoff1/self.num_bonds
#        P2 =  cutoff2/self.num_bonds
#        # MC siulation starts
#        for MC_index in range(1,self.tot_steps):
#            #print("MC step: {:d}".format(MC_index))
#            #print("Updating J:")
#            #for update_index in range((MC_index-1)*self.num_bonds, MC_index*self.num_bonds):
#            for update_index in range(0, self.num_bonds):
#                #=========
#                #Method 1 
#                #=========
#                self.random_update_J()
#                ##=========
#                ##Method 2 
#                ##=========
#                ## Generate a random integer
#                #index_J = randrange(self.num_bonds)
#                #if index_J < self.cutoff1:
#                #    n2,n1 = index_J//self.N_in, index_J%self.N_in
#                #    # Update J_in[n2,n1]
#                #    self.shift_bond_in(n2,n1,x)
#                #    self.decision_bond_in_by_n2_n1(n2,n1,EPS,np.random.random(1))
#                #elif index_J > self.cutoff1-1 and index_J < cutoff2:
#                #    index_J_2 = index_J - self.cutoff1
#                #    l,n2,n1 = index_J_2//SQ_N, (index_J_2%SQ_N)//self.N, (index_J_2%SQ_N)%self.N 
#                #    self.shift_bond_hidden(l,n2,n1,x)
#                #    self.decision_bond_hidden_by_l_n2_n1(l,n2,n1,EPS,np.random.random(1))
#                #elif index_J > cutoff2-1:
#                #    index_J_3 = index_J - cutoff2
#                #    n2,n1 = index_J_3//self.N, index_J_3%self.N
#                #    self.shift_bond_out(n2,n1,x)
#                #    self.decision_bond_out_by_n2_n1(n2,n1,EPS,np.random.random(1))
#                #else:
#                #    pass
#            #print("Updating S:")
#            #for update_index in range((MC_index-1) * self.num_variables, MC_index * self.num_variables):
#            for update_index in range(0, self.num_variables):
#                mu,l,n = randrange(self.M),randrange(self.num_hidden_node_layers),randrange(self.N)
#                self.flip_spin(mu,l,n)
#                if l == 0:
#                    self.decision_node_in_by_mu_n(mu,n,EPS,np.random.random(1))
#                elif l == self.num_hidden_node_layers - 1:
#
#                    self.decision_node_out_by_mu_n(mu,n,EPS,np.random.random(1))
#                else:
#                    self.decision_by_mu_l_n(mu,l,n,EPS,np.random.random(1))
#            self.count_MC_step += 1
#            #Check and save the configureation of bonds 
#            self.check_and_save() # save configuration
#            
#            #FOR TESTING 
#            #file1.write("{}\n".format(str(self.count_MC_step)))
#            #for j in range(self.num_hidden_bond_layers):
#            #    for i in range(self.N):
#            #        for ii in range(self.N):
#            #            file1.write("{}\n".format(str(self.J_hidden[j][i][ii])))
#        #file1.close()
#
#    @timethis
#    def mc_main_hyperfine(self,str_timestamp,replica_index):
#        """MC for the host machine, i.e., it will save seeds for different waiting time."""
#        str_replica_index = str(replica_index)
#        rel_path='J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:3.1f}_step{:d}_host.dat'.format(str_replica_index,self.init,self.tw,self.L,self.N,self.beta,self.tot_steps)
#        src_dir = os.path.dirname(__file__) 
#        abs_file_path=os.path.join(src_dir, rel_path)
#        #file1 = open(abs_file_path,'w') # IF you want to test, use this line (step 1-2)
#
#        EPS = self.EPS
#        SQ_N = (self.N) ** 2
#        
#        cutoff1 = self.cutoff1        
#        cutoff2 = self.cutoff2        
#        P1 = cutoff1/self.num_bonds
#        P2 = cutoff2/self.num_bonds
#        print("list_k_4_hyperfine:")
#        print(self.list_k_4_hyperfine)
#        # MC siulation starts
#        for MC_index in range(1,self.tot_steps):
#            #print("MC step: {:d}".format(MC_index))
#            #print("Updating S:")
#            start_index_variables = (MC_index-1) * self.num
#            end_index_variables = start_index_variables + self.num_variables 
#            for update_index in range(start_index_variables, end_index_variables):
#                mu,l,n = randrange(self.M),randrange(self.num_hidden_node_layers),randrange(self.N)
#                self.flip_spin(mu,l,n)
#                if l == 0:
#                    self.decision_node_in_by_mu_n(mu,n,EPS,np.random.random(1))
#                elif l == self.num_hidden_node_layers - 1:
#                    self.decision_node_out_by_mu_n(mu,n,EPS,np.random.random(1))
#                else:
#                    self.decision_by_mu_l_n(mu,l,n,EPS,np.random.random(1))
#                #Check and save the configureation of variables
#                self.check_and_save_hyperfine(update_index) # save configuration
#                #Check and save the seeds 
#                self.check_and_save_seeds_hyperfine(update_index,str_timestamp)
#
#            #print("Updating J:")
#            start_index_bonds = end_index_variables
#            end_index_bonds = MC_index * self.num
#            for update_index in range(start_index_bonds, end_index_bonds):
#            #for update_index in range(0, self.num_bonds):
#                #=========
#                #Method 1 
#                #=========
#                self.random_update_J()
# 
#                #Check and save the configureation of bonds 
#                self.check_and_save_hyperfine(update_index) # save configuration
#                #Check and save the seeds 
#                self.check_and_save_seeds_hyperfine(update_index,str_timestamp)
#
#            ##TESTING # IF you want to test, use these line (step 2-2)
#            #file1.write("{}\n".format(str(self.count_MC_step)))
#            #for j in range(self.num_hidden_bond_layers):
#            #    for i in range(self.N):
#            #        for ii in range(self.N):
#            #            file1.write("{}\n".format(str(self.J_hidden[j][i][ii])))
#        #file1.close()
#
#    @timethis
#    def mc_odd_hyperfine(self,str_timestamp,replica_index):
#        """MC for the guest machine, i.e., it will save seeds for different waiting time."""
#        str_replica_index = str(replica_index)
#        rel_path='J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:3.1f}_step{:d}_odd.dat'.format(str_replica_index,self.init,self.tw,self.L,self.N,self.beta,self.tot_steps)
#        src_dir = os.path.dirname(__file__) 
#        abs_file_path=os.path.join(src_dir, rel_path)
#        #file1 = open(abs_file_path,'w')
#
#        EPS = self.EPS
#        SQ_N = (self.N) ** 2
#        
#        cutoff1 = self.cutoff1        
#        cutoff2 = self.cutoff2        
#        print("list_k:")
#        print(self.list_k)
#        # MC siulation starts
#        for MC_index in range(1,self.tot_steps):
#            #print("MC step: {:d}".format(MC_index))
#            #print("Updating S:")
#            start_index_variables = (MC_index-1) * self.num
#            end_index_variables = start_index_variables + self.num_variables 
#            for update_index in range(start_index_variables, end_index_variables):
#                # Const.s
#                rand1 = np.random.random(1)
#
#                mu,l,n = randrange(self.M),randrange(self.num_hidden_node_layers),randrange(self.N)
#                self.flip_spin(mu,l,n)
#                if l == 0:
#                    self.decision_node_in_by_mu_n(mu,n,EPS,rand1)
#                elif l == self.num_hidden_node_layers - 1:
#                    self.decision_node_out_by_mu_n(mu,n,EPS,rand1)
#                else:
#                    self.decision_by_mu_l_n(mu,l,n,EPS,rand1)
#                #Check and save the configureation of variables
#                self.check_and_save_hyperfine(update_index) # save configuration
#
#            #print("Updating J:")
#            start_index_bonds = end_index_variables
#            end_index_bonds = MC_index * self.num
#            for update_index in range(start_index_bonds, end_index_bonds):
#            #for update_index in range(0, self.num_bonds):
#                x = np.random.normal(loc=0,scale=1.0,size=None)
#                x = round(x,10)
#                #=========
#                #Method 1
#                #=========
#                rand_num = np.random.random(1)
#                rand2 = np.random.random(1)
#                
#                if rand_num < P1:
#                    n2,n1=randrange(self.N),randrange(self.N_in)
#                    # Update J_in[n2,n1]
#                    self.shift_bond_in(n2,n1,x)
#                    self.decision_bond_in_by_n2_n1(n2,n1,EPS,rand2)
#                elif rand_num > P1 and rand_num < P2:
#                    l,n2,n1 = randrange(0,self.L-2),randrange(self.N),randrange(self.N) 
#                    self.shift_bond_hidden(l,n2,n1,x)
#                    self.decision_bond_hidden_by_l_n2_n1(l,n2,n1,EPS,rand2)
#                else:
#                    n2,n1=randrange(self.N_out),randrange(self.N)
#                    self.shift_bond_out(n2,n1,x)
#                    self.decision_bond_out_by_n2_n1(n2,n1,EPS,rand2)
#                #Check and save the configureation of bonds 
#                self.check_and_save_hyperfine(update_index) # save configuration
#
#    @timethis
#    def mc_even_hyperfine(self,str_timestamp,replica_index):
#        """MC for the host machine, i.e., it will save seeds for different waiting time."""
#        str_replica_index = str(replica_index)
#        rel_path='J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:3.1f}_step{:d}_even.dat'.format(str_replica_index,self.init,self.tw,self.L,self.N,self.beta,self.tot_steps)
#        src_dir = os.path.dirname(__file__) 
#        abs_file_path=os.path.join(src_dir, rel_path)
#        #file1 = open(abs_file_path,'w')
#
#        EPS = self.EPS
#        SQ_N = (self.N) ** 2
#        
#        cutoff1 = self.cutoff1        
#        cutoff2 = self.cutoff2        
#        P1 =  cutoff1/self.num_bonds
#        P2 =  cutoff2/self.num_bonds
#        print("list_k:")
#        print(self.list_k)
#        # MC siulation starts
#        for MC_index in range(1,self.tot_steps):
#            #print("Updating J:")
#            start_index_bonds = (MC_index-1) * self.num
#            end_index_bonds = start_index_bonds + self.num_bonds 
#            for update_index in range(start_index_bonds, end_index_bonds):
#                self.random_update_J()
#                #Check and save the configureation of bonds 
#                self.check_and_save_hyperfine(update_index) # save configuration
#
#            #print("MC step: {:d}".format(MC_index))
#            #print("Updating S:")
#            start_index_variables = end_index_bonds
#            end_index_variables = MC_index * self.num
#            for update_index in range(start_index_variables, end_index_variables):
#                # Const.s
#                rand2 = np.random.random(1)
#
#                mu,l,n = randrange(self.M),randrange(self.num_hidden_node_layers),randrange(self.N)
#                self.flip_spin(mu,l,n)
#                if l == 0:
#                    self.decision_node_in_by_mu_n(mu,n,EPS,rand2)
#                elif l == self.num_hidden_node_layers - 1:
#                    self.decision_node_out_by_mu_n(mu,n,EPS,rand2)
#                else:
#                    self.decision_by_mu_l_n(mu,l,n,EPS,rand2)
#                #Check and save the configureation of variables
#                self.check_and_save_hyperfine(update_index) # save configuration
#
#    def mc_update_J_or_S(self):
#        # Const.s
#        EPS = self.EPS
#        cutoff1 = self.cutoff1        
#        cutoff2 = self.cutoff2        
#        P1 = cutoff1/self.num_bonds
#        P2 = cutoff2/self.num_bonds
#        rand1 = np.random.random(1)
#        rand2 = np.random.random(1)
#
#        if rand1 < self.PROB:
#            # Flip one spin and make a decision: 
#            mu,l,n = randrange(self.M),randrange(self.num_hidden_node_layers),randrange(self.N)
#            self.flip_spin(mu,l,n)
#            if l == 0:
#                self.decision_node_in_by_mu_n(mu,n,EPS,rand2)
#            elif l == self.num_hidden_node_layers - 1:
#                self.decision_node_out_by_mu_n(mu,n,self.EPS,rand2)
#            else:
#                self.decision_by_mu_l_n(mu,l,n,self.EPS,rand2)
#        else:
#            self.random_update_J()
#
#    def decision_by_mu_l_n(self,mu,l,n,EPS,rand):
#        # Const.s
#        rand1 = np.random.random(1)
#
#        self.delta_H = calc_ener(self.part_gap_after_flip(mu,l,n)) - calc_ener(self.part_gap_before_flip(mu,l,n))
#        delta_e = round(self.delta_H,10)
#        if delta_e < EPS:
#            self.accept_by_mu_l_n(mu,l,n)
#            #print("[S] Delta E:{:12.10f}".format(delta_e))
#        else:
#            if rand1 < np.exp(-delta_e * self.beta):
#                self.accept_by_mu_l_n(mu,l,n)
#                #print("[S] Delta E:{:12.10f}".format(delta_e))
#            else:
#                pass
#    @timethis
#    def mc_main_random_update_hyperfine(self,str_timestamp):
#        """MC for the main machine, i.e., it will save seeds for different waiting time."""
#        print("list_k:")
#        print(self.list_k)
#        # MC siulation starts
#        STEPS = self.tot_steps * self.num
#        for update_index in range(STEPS):
#            #print("Updating S and J randomely, with a fixed probability:")
#            self.mc_update_J_or_S()
#
#            #Check and save the configureation of variables
#            self.check_and_save_hyperfine(update_index) # save configuration
#
#            #Check and save the seeds (ONLY IN mc_main) 
#            self.check_and_save_seeds_hyperfine(update_index,str_timestamp)
#
#    @timethis
#    def mc_random_update_hyperfine(self,str_timestamp):
#        """MC for the guest machine, i.e., it will save seeds for different waiting time."""
#        print("list_k:")
#        print(self.list_k)
#        # MC siulation starts
#        STEPS = self.tot_steps * self.num
#        for update_index in range(STEPS):
#            #print("Updating S and J randomely, with a fixed probability:")
#            self.mc_update_J_or_S()
#
#            #Check and save the configureation of bonds 
#            self.check_and_save_hyperfine(update_index) # save configuration

class Network_6f:
    def __init__(self,init,tw,L,M,N,tot_steps,beta,timestamp):
        """Since Yoshino_3.0, when update the energy, we do not calculate energy for all the gaps, but only calculate these part affected by the flipping of a SPIN (S)  or shifting of 
           a BOND (J). This will accelerate the calculation by hundreds of times. (2) Besides, we note that we do NOT need to define a functon: remain(), which records
           the new MC steps' S, J and H, even though one MC move is rejected."""
        self.init = int(init)
        self.tw = int(tw)
        self.L = int(L) # number of layers
        self.M = int(M) # number of samples
        self.N = int(N) # the number of neurons at each layer is a constant.
        self.tot_steps = int(tot_steps) # number of total MC simulation steps
        self.timestamp = timestamp  
        self.beta = beta # inverse temperature
        
        # Define new parameters: T (technically required)
        T = self.tot_steps+1  # we keep the initial state in the first step 
        
        self.H = 0 # for storing energy when update
        self.new_H = 0 # for storing temperay energy when update

        # Define J_in and J_out
        self.S_in = np.zeros((self.M,self.N))  
        self.S_out = np.zeros((self.M,self.N))  

        # Energy difference caused by update of variables J or S 
        self.delta_H= 0 
        self.num_hidden_node_layers = self.L - 1
        # Initialize the inner parameters: num_bonds, num_variables
        self.SQ_N = (self.N) ** 2
        self.SQRT_N = np.sqrt(self.N)
        self.num_variables = int(self.N * self.M * self.num_hidden_node_layers) 
        self.BIAS = 1 # For avoiding x IS NOT EQUAL TO ZERO in log2(x)
        self.BIAS2 = 5 # For obtaing long enough list list_k.
        self.T = int(np.log2(self.tot_steps + self.BIAS)) # we keep the initial state in the first step
        self.num_bonds = int(self.SQ_N * self.L) 
        self.num = self.num_variables + self.num_bonds
        
        self.ind_save = 0
        self.count_MC_step = 0 

        self.T_2 = int(np.log2(self.tot_steps * self.num + self.BIAS)) # we keep the initial state in the first step

        # The arrays for storing MC trajectories of S, J and H
        self.J_traj_hyperfine = np.zeros((self.T_2, self.L, self.N, self.N), dtype='float32') 
        self.S_traj_hyperfine = np.zeros((self.T_2, self.M, self.num_hidden_node_layers, self.N), dtype='int8')
        self.H_traj_hyperfine = np.zeros(self.T_2, dtype='float32') 

        # print("self.num_hidden_node_layers:")
        # print(self.num_hidden_node_layers)
        # print("self.T:")
        # print(self.T)

        self.EPS = 1e-10
        self.RAT = 0.1 # r: Yoshino2019 Eq(35)
        self.RESCALE_J = 1.0 / np.sqrt(1 + self.RAT**2)
        
        
        self.PROB = self.num_variables/self.num # Float type 

        # Parameters used in the host machine
        # For recording which kind of coordinate is activated
        self.S_active = False 
        self.J_active = False 
        #Intialize S and J by the array S and J. 
        #Note: Both S and J are the coordinates of a machine.
        self.S = init_S(self.M, self.L, self.N) # L layers
        self.J = init_J(self.L, self.N) # L layers 
        self.r = self.gap_init_6f() # the initial gap is returned from a function.

        self.new_S = copy.copy(self.S) # for storing temperay array when update 
        self.new_J = copy.copy(self.J) # for storing temperay array when update  
        self.new_r = copy.copy(self.r)

        self.count_MC_step = 0           

        # For recording which layer is updating
        self.updating_layer_index = None 
        # For recording which node is updating for S

        # EXAMPLE: list_k = [4,8,16,64,256,1024,4096,16384,65536]
        self.list_k = [2**(i) for i in range(int(np.log2(self.tot_steps+self.BIAS) + self.BIAS2))]
        self.list_k_4_hyperfine = [2**(i) for i in range(int(np.log2(self.tot_steps * self.num + self.BIAS) + self.BIAS2))]

    def gap_init_6f(self):
        '''Ref: Yoshino2019, eqn (31b)
           In the 6f version, both input and output have N bits. 
        '''
        r = np.zeros((self.M,self.L,self.N))
        SQRT_N = self.SQRT_N
        for mu in range(self.M):
            # in
            l = 0
            for n2 in range(self.N):
                r[mu,l,n2] = (np.sum(self.J[l,n2,:] * self.S_in[mu,:])/SQRT_N) * self.S[mu,l,n2] 
            # out
            l = self.L-1 
            for n2 in range(self.N):
                r[mu,l,n2] = (np.sum(self.J[l,n2,:] * self.S[mu,l-1:])/SQRT_N) * self.S_out[mu,n2] 
            # hidden
            for l in range(1,self.L-1): # l = 1,...,L-2
                for n2 in range(self.N):
                    r[mu,l,n2] = (np.sum(self.J[l,n2,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n2] 
        return r    
    def flip_spin(self,mu,l,n):
        '''flip_spin() will flip S at a given index tuple (mu,l,n). We add mu,l,n as parameters, for parallel programming. Note: any spin can be update except the input/output.'''
        # Update self.new_S
        self.new_S = copy.copy(self.S)
        self.new_S[mu][l][n] = -1 * self.S[mu][l][n]  
    def shift_bond(self,l,n2,n1,x):
        '''shift_bond() will shift the element of J with a given index to another value. We add l,n2,n1 as parameters, for parallel programming..'''
        self.new_J = copy.copy(self.J)
        N = self.N
        # scale denotes standard deviation
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J[l][n2][n1] = (self.J[l][n2][n1] + x * self.RAT) * self.RESCALE_J
        # rescaling 
        t = self.new_J[l][n2] 
        N_prim = np.sum(t*t)
        SCALE = np.sqrt(N / N_prim)
        self.new_J[l][n2] = self.new_J[l][n2] * SCALE
    def accept_by_mu_l_n(self,mu,l,n):
        """This accept function is used if S is flipped."""
        self.S[mu,l,n] = self.new_S[mu,l,n]
        self.H = self.H + self.delta_H
        #print("ENERGY: {}".format(self.H))
    def accept_by_l_n2_n1(self,l,n2,n1):
        """This accept function is used if J is shifted."""
        self.J[l,n2,n1] = self.new_J[l,n2,n1]
        self.H = self.H + self.delta_H
        #print("ENERGY: {}".format(self.H))
    def part_gap_before_flip(self,mu,l,n):
        '''Ref: Yoshino2019, eqn (31b)
           When S is fliped, only one machine changes its coordinates and it will affect the gap of the node before it and the gaps of the N nodes
           behind it. Therefore, N+1 gaps contributes to the Delta_H_eff. l = 0,1, ..., L-1. 
           We define a small array, part_gap, which has N+1 elements. Each elements of part_gap is a r^mu_node. Use part_gap, we can calculate the 
           Energy change coused by the flip of S^mu_node,n. 
        '''
        SQRT_N = self.SQRT_N
        part_gap = np.zeros(self.N + 1)
        if l == 0:
            part_gap[0] = (np.sum( self.J[l,n,:] * self.S_in[mu,:])/SQRT_N) * self.S[mu,l,n] 
            for n2 in range(self.N):
                part_gap[1+n2] = (np.sum(self.J[l+1,n2,:] * self.S[mu,l,:])/SQRT_N) * self.S[mu,l+1,n2] 
        if l == self.num_hidden_node_layers -1: #  = L - 2
            part_gap[0] = (np.sum( self.J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n] 
            for n2 in range(self.N):
                # -1: the last layer
                part_gap[1+n2] = (np.sum(self.J[-1,n2,:] * self.S[mu,l,:])/SQRT_N) * self.S_out[mu,n2] 
        else:
            part_gap[0] = (np.sum( self.J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n] 
            for n2 in range(self.N):
                part_gap[1+n2] = (np.sum(self.J[l+1,n2,:] * self.S[mu,l,:])/SQRT_N) * self.S[mu,l+1,n2] 
        return part_gap  # Only the N+1 elements affect the Delta_H_eff. 
    def part_gap_after_flip(self,mu,l,n):
        SQRT_N = self.SQRT_N
        part_gap = np.zeros(self.N + 1)
        if l == 0:
            part_gap[0] = (np.sum( self.J[l,n,:] * self.S_in[mu,:])/SQRT_N) * self.new_S[mu,l,n] 
            for n2 in range( self.N):
                part_gap[1+n2] = (np.sum(self.J[l+1,n2,:] * self.new_S[mu,l,:])/SQRT_N) * self.S[mu,l+1,n2] 
        if l == self.num_hidden_node_layers - 1:
            part_gap[0] = (np.sum( self.J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.new_S[mu,l,n] 
            for n2 in range( self.N):
                # -1: the last layer
                part_gap[1+n2] = (np.sum(self.J[-1,n2,:] * self.new_S[mu,l,:])/SQRT_N) * self.S_out[mu,n2] 
        else:
            part_gap[0] = (np.sum( self.J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.new_S[mu,l,n] 
            for n2 in range( self.N):
                part_gap[1+n2] = (np.sum(self.J[l+1,n2,:] * self.new_S[mu,l,:])/SQRT_N) * self.S[mu,l+1,n2] 
        return part_gap  # Only the N+1 elements affect the Delta_H_eff. 
    def part_gap_before_shift(self,l,n): 
        SQRT_N = self.SQRT_N
        part_gap = np.zeros(self.M)
        if l == 0:
            for mu in range(self.M):
                part_gap[mu] = (np.sum(self.J[l,n,:] * self.S_in[mu,:])/SQRT_N) * self.S[mu,l,n]
        elif l == self.L-1:
            for mu in range(self.M):
                part_gap[mu] = (np.sum(self.J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S_out[mu,n] 
        else:
             for mu in range(self.M):
                part_gap[mu] = (np.sum(self.J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n] 
        return part_gap  # Only the M elements affect the Delta_H_eff. 
    def part_gap_after_shift(self,l,n): 
        SQRT_N = self.SQRT_N
        part_gap = np.zeros(self.M)
        if l == 0:
            for mu in range(self.M):
                part_gap[mu] = (np.sum(self.new_J[l,n,:] * self.S_in[mu,:])/SQRT_N) * self.S[mu,l,n]
        elif l == self.L-1:
            for mu in range(self.M):
                part_gap[mu] = (np.sum(self.new_J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S_out[mu,n] 
        else:
             for mu in range(self.M):
                part_gap[mu] = (np.sum(self.new_J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n] 
        return part_gap  # Only the M elements affect the Delta_H_eff.        
    def part_gap_after_shift_dated(self,l,n): 
        SQRT_N = self.SQRT_N
        part_gap = np.zeros(self.M)
        for mu in range(self.M):
            part_gap[mu] = (np.sum(self.new_J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n] 
        return part_gap # Only the M elements affect the Delta_H_eff. 

    # One of the follwing decision functions is used if S is flipped.
    def decision_by_mu_l_n_V2(self,mu,l,n,EPS,rand):
        # Const.s
        rand1 = np.random.random(1)

        self.delta_H = calc_ener(self.part_gap_after_flip(mu,l,n)) - calc_ener(self.part_gap_before_flip(mu,l,n))
        delta_e = round(self.delta_H,10)
        if delta_e < EPS:
            self.accept_by_mu_l_n(mu,l,n)
            #print("[S] Delta E:{:12.10f}".format(delta_e))
        else:
            if rand1 < np.exp(-delta_e * self.beta):
                self.accept_by_mu_l_n(mu,l,n)
                #print("[S] Delta E:{:12.10f}".format(delta_e))
            else:
                pass
    def decision_by_mu_l_n_SIMPLE(self,mu,l,n):
        """If use this decision_by_mu_l_n_SIMPLE() function, the parameter EPS is not needed in the input."""
        # Const.s
        rand1 = np.random.random(1)

        self.delta_H = calc_ener(self.part_gap_after_flip(mu,l,n)) - calc_ener(self.part_gap_before_flip(mu,l,n))
        delta_e = self.delta_H
        
        if delta_e > 0:
            if rand1 < np.exp(-delta_e * self.beta):
                self.accept_by_mu_l_n(mu,l,n)
            else:
                pass
        else:
            self.accept_by_mu_l_n(mu,l,n) 
    def decision_by_mu_l_n(self,mu,l,n,EPS,rand):
        self.delta_H = calc_ener(self.part_gap_after_flip(mu,l,n)) - calc_ener(self.part_gap_before_flip(mu,l,n))
        delta_e = round(self.delta_H,10)
        if delta_e < EPS:
            self.accept_by_mu_l_n(mu,l,n)
            #print("[S] Delta E:{:12.10f}".format(delta_e))
        else:
            if rand < np.exp(-delta_e * self.beta):
                self.accept_by_mu_l_n(mu,l,n)
                #print("[S] Delta E:{:12.10f}".format(delta_e))
            else:
                pass
    def decision_node_in_by_mu_n(self,mu,n,EPS,rand):
        self.delta_H = calc_ener(self.part_gap_in_after_flip(mu,n)) - calc_ener(self.part_gap_in_before_flip(mu,n))
        print("delta_e: {} (before round)".format(self.delta_H))
        delta_e = round(self.delta_H,10)
        print("delta_e: {} (after round)".format(delta_e))
        temp_l = 0
        if delta_e < EPS:
            self.accept_by_mu_l_n(mu,temp_l,n)
        else:
            if rand < np.exp(-delta_e * self.beta):
                self.accept_by_mu_l_n(mu,temp_l,n)
            else:
                pass
    def decision_node_out_by_mu_n(self,mu,n,EPS,rand):
        self.delta_H = calc_ener(self.part_gap_out_after_flip(mu,n)) - calc_ener(self.part_gap_out_before_flip(mu,n))
        delta_e = round(self.delta_H,10)
        l = -1
        if delta_e < EPS:
            self.accept_by_mu_l_n(mu,l,n)
        else:
            if rand < np.exp(-delta_e * self.beta):
                self.accept_by_mu_l_n(mu,l,n)
            else:
                pass

    # One of the follwing decision functions is used if J is shifted.
    def decision_bond_hidden_by_l_n2_n1(self,l,n2,n1,EPS,rand):
        self.delta_H = calc_ener(self.part_gap_after_shift(l,n2)) - calc_ener(self.part_gap_before_shift(l,n2))
        delta_e = round(self.delta_H,10)
        if delta_e < EPS:
            # Replace o.S by o.new_S:
            self.accept_bond_hidden_by_l_n2_n1(l,n2,n1)
            #print("[J] Delta E:{:12.10f}".format(delta_e))
        else:
            if rand < np.exp(-delta_e * self.beta):
                self.accept_bond_hidden_by_l_n2_n1(l,n2,n1)
                #print("[J] Delta E:{:12.10f}".format(delta_e))
            else:
                pass # We do not need a "remain" function
    def decision_bond_in_by_n2_n1(self,n2,n1,EPS,rand):
        self.delta_H = calc_ener(self.part_gap_in_after_shift(n2)) - calc_ener(self.part_gap_in_before_shift(n2))
        delta_e = round(self.delta_H,10)
        if delta_e < EPS:
            # Replace o.S by o.new_S:
            self.accept_bond_in_by_n2_n1(n2,n1)
            #print("[J] Delta E:{:12.10f}".format(delta_e))
        else:
            if rand < np.exp(-delta_e * self.beta):
                self.accept_bond_in_by_n2_n1(n2,n1)
                #print("[J] Delta E:{:12.10f}".format(delta_e))
            else:
                pass # We do not need a "remain" function
    def decision_bond_out_by_n2_n1(self,n2,n1,EPS,rand):
        self.delta_H = calc_ener(self.part_gap_out_after_shift(n2)) - calc_ener(self.part_gap_out_before_shift(n2))
        delta_e = round(self.delta_H,10)
        if delta_e < EPS:
            self.accept_bond_out_by_n2_n1(n2,n1)
        else:
            if rand < np.exp(-delta_e * self.beta):
                self.accept_bond_out_by_n2_n1(n2,n1)
            else:
                pass # We do not need a "remain" function
    def decision_by_l_n2_n1(self,l,n2,n1,EPS,rand):
        self.delta_H = calc_ener(self.part_gap_after_shift(l,n2)) - calc_ener(self.part_gap_before_shift(l,n2))
        delta_e = round(self.delta_H,10)
        if delta_e < EPS:
            self.accept_bond_hidden_by_l_n2_n1(l,n2,n1)
        else:
            if rand < np.exp(-delta_e * self.beta):
                self.accept_bond_hidden_by_l_n2_n1(l,n2,n1)
            else:
                pass # We do not need a "remain" function
    def decision_by_l_n2_n1(self,l,n2,n1,EPS,rand):
        self.delta_H = calc_ener(self.part_gap_after_shift(l,n2)) - calc_ener(self.part_gap_before_shift(l,n2))
        delta_e = round(self.delta_H,10)
        if delta_e < EPS:
            self.accept_by_l_n2_n1(l,n2,n1)
        else:
            if rand < np.exp(-delta_e * self.beta):
                self.accept_by_l_n2_n1(l,n2,n1)
            else:
                pass # We do not need a "remain" function
    def update_spin(self,ind): 
        self.flip_spin(ind[0],ind[1],ind[2])
        self.decision_by_mu_l_n(MC_index,ind[0],ind[1],ind[2])

    @timethis
    def mc_main_random_update_hyperfine_6f(self,str_timestamp):
        """MC for the main machine, i.e., it will save seeds for different waiting time."""
        print("list_k:")
        print(self.list_k)
        # MC siulation starts
        STEPS = self.tot_steps * self.num
        for update_index in range(STEPS):
            #print("Updating S and J randomely, with a fixed probability:")
            self.mc_update_J_or_S_6f()

            #Check and save the configureation of variables
            self.check_and_save_hyperfine_6f(update_index) # save configuration

            #Check and save the seeds (ONLY IN mc_main) 
            self.check_and_save_seeds_hyperfine_6f(update_index,str_timestamp)
   
    def check_and_save_seeds_hyperfine_6f(self,update_index,str_timestamp):
        ind = update_index
        init = self.init
        if ind==self.list_k_4_hyperfine[0]:
            self.list_k_4_hyperfine.pop(0) 
            np.save('../data/{:s}/seed_S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:3.1f}.npy'.format(str_timestamp,str_timestamp,init,ind,self.L,self.M,self.N,self.beta),self.S)   
            np.save('../data/{:s}/seed_J_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:3.1f}.npy'.format(str_timestamp,str_timestamp,init,ind,self.L,self.N,self.beta),self.J)
            np.save('../data/{:s}/seed_ener_new_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:3.1f}.npy'.format(str_timestamp,str_timestamp,init,ind,self.L,self.M,self.N,self.beta),self.H)
        else:
            pass

    def check_and_save_hyperfine_6f(self,update_index):
        ind = update_index
        if (2**self.ind_save) == ind and self.ind_save < self.S_traj_hyperfine.shape[0]:
            self.S_traj_hyperfine[self.ind_save] = self.S
            self.J_traj_hyperfine[self.ind_save] = self.J
            self.H_traj_hyperfine[self.ind_save] = self.H
            self.ind_save += 1
        else:
            pass

    @timethis
    def mc_random_update_hyperfine_6f(self,str_timestamp):
        """MC for the guest machine, i.e., it will save seeds for different waiting time."""
        # Const.s
        STEPS = self.tot_steps * self.num
        print("list_k:")
        print(self.list_k)
        # MC siulation starts
        for update_index in range(STEPS):
            #print("Updating S and J randomely, with a fixed probability:")
            self.mc_update_J_or_S_6f()

            #Check and save the configureation of bonds 
            self.check_and_save_hyperfine_6f(update_index) # save configuration
    def mc_update_J_or_S_6f(self):
        EPS = self.EPS
        p1 = np.random.random(1)
        if p1 < self.PROB:
            # Flip on node and make a decision:
            self.random_update_S_6f()
        else:
            # Shift one bond and make a decision: 
            self.random_update_J_6f()

    def random_update_J_6f(self):
        """Method 1"""
        # Const.s
        x = np.random.normal(loc=0,scale=1.0,size=None)
        x = round(x,10)
        EPS = self.EPS
        SQ_N = (self.N) ** 2
        l,n2,n1 = randrange(self.L),randrange(self.N),randrange(self.N) 
        p2 = np.random.random(1)
        # Shift a bond and make a decision: 
        self.shift_bond(l,n2,n1,x)
        self.decision_by_l_n2_n1(l,n2,n1,EPS,p2)
    def random_update_S_6f(self):
        # Const.s
        EPS = self.EPS
        mu,l,n = randrange(self.M),randrange(self.num_hidden_node_layers),randrange(self.N)
        p2 = np.random.random(1)
        # Flip a node and make a decision: 
        self.flip_spin(mu,l,n)
        self.decision_by_mu_l_n(mu,l,n,EPS,p2)

