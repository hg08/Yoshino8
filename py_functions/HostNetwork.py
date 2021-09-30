#======
#Module
#======
import sys

sys.path.append('/public1/home/sc91981/py_functions/')
#sys.path.append('/home/gang/Github/Yoshino/py_functions/')

from utilities import *

from random import choice
import copy
import math
import numpy as np
import tensorflow as tf # Import dataset from tf.keras, instead of from keras directly.
from scipy.stats import norm
import os
import matplotlib.pyplot as plt
from random import randrange
import scipy as sp
from time import time

class HostNetwork:
    def __init__(self,init,tw,L,M,N,N_in,N_out,tot_steps,beta,timestamp):
        """Since Yoshino_3.0, when update the energy, we do not calculate all the gaps, but only calculate the part affected by the flip of a SPIN (S)  or a shift of
           a BOND (J). This will accelerate the calculation by hundreds of times. (2) Besides, we note that we do NOT need to define a functon: remain(), which records
           the new MC steps' S, J and H, even though one MC move is rejected."""
        self.init= int(init)
        self.tw = int(tw)
        self.L = int(L)
        self.M = int(M)
        self.N = int(N)
        self.tot_steps = int(tot_steps)
        self.beta = beta
        self.timestamp = timestamp
        self.N_in = int(N_in)
        self.N_out = int(N_out)
        self.num_hidden_node_layers = self.L - 1 # After distingush num_hidden_node_layers and num_hidden_bond_layers, then I found the clue is much clear!
        self.num_hidden_bond_layers = self.L - 2
        # Define new parameters; T (technically required)
        self.BIAS = 1
        self.T = int(np.log2(self.tot_steps+self.BIAS)) # we keep the initial state in the first step

        self.H = 0 # for storing energy
        self.new_H = 0 # for storing temperay energy when update

        # Energy difference caused by update of sample mu
        self.delta_H= 0

        # Intialize S and J by the array S and J.
        # We use S, instead of S_hidden, because S[0], S[L-2] still have interaction to J_in and J_out. Ref. Yoshino eq (8).
        self.S = init_S(self.M,self.num_hidden_node_layers,self.N)
        self.J_hidden = init_J(self.num_hidden_bond_layers,self.N)
        #TEST
        print("S:")
        print(self.S)
        print("J_hidden:")
        print(self.J_hidden)
        # Define J_in and J_out
        self.S_in = np.zeros((self.M,self.N_in))  
        self.S_out = np.zeros((self.M,self.N_out))  
        self.J_in = init_J_in(self.N,self.N_in)  
        self.J_out = init_J_out(self.N_out,self.N)
        print("J_in:")
        print(self.J_in)
        print("J_out:")
        print(self.J_out)
        #self.new_S = 0 # for storing temperay array when update
        #self.new_J = 0 # for storing temperay array when update

        #self.new_J_in = np.zeros((self.N,self.N_in)) 
        #self.new_J_out = np.zeros((self.N_out,self.N))  
      
        self.r = 0 # the initial gap
        self.r_in = 0 # the initial gap
        self.r_out = 0 # the initial gap

        # Initialize the inner parameters: num_bonds, num_variables
        self.num_variables = 0
        self.num_bonds = 0
        self.num = 0

        self.count_MC_step = 0

    # The following accept function is used if S is flipped.
    def accept_by_mu_l_n(self,mu,l,n):
        """This accept function is used if S is flipped."""
        self.S[mu,l,n] = self.new_S[mu,l,n]
        self.H = self.H + self.delta_H
    
    def rand_index_for_S(self):
        # For S: list_index_for_S = [(mu,l,n),...]
        list_index_for_S = []
        for _ in range(self.num_variables * (self.tot_steps-1)):
            list_index_for_S.append([randrange(self.M), randrange(1,self.L-1), randrange(self.N)])
        res_arr = np.array(list_index_for_S)
        return res_arr
    def rand_index_for_J(self):
        # For generating J: list_index_for_J = [(l,n2,n1),...]
        list_index_for_J = []
        for _ in range(self.num_bonds * (self.tot_steps-1)):
            list_index_for_J.append([randrange(1,self.L), randrange(self.N), randrange(self.N)])
        res_arr = np.array(list_index_for_J)
        return res_arr
    def rand_series_for_x(self):
        """
        For generating J: list_for_x = [x1,x2,...]
        We separate rand_index_for_J() and rand_series_for_x(), instead of merginging them to one function and return a list of four-tuple (l,n2,n1,x).
        The reason is: x is float and l,n2,n1 are integers, it will induce trouble if one put them (x and l,n2,n1 ) together.
        """
        list_for_x = []
        for _ in range(self.num_bonds * (self.tot_steps-1)):
            x = np.random.normal(loc=0,scale=1.0,size=None)
            x = round(x,10)
            list_for_x.append(x)
        res_arr = np.array(list_for_x)
        return res_arr

    def rand_series_for_decision_on_S(self):
        # For generating J: list_index_for_J = [(l,n2,n1),...]
        list_for_decision = []
        for _ in range(self.num_variables * (self.tot_steps-1)):
            list_for_decision.append(np.random.random(1))
        res_arr = np.array(list_for_decision)
        return res_arr
    def rand_series_for_decision_on_J(self):
        # For generating J: list_index_for_J = [(l,n2,n1),...]
        list_for_decision = []
        for _ in range(self.num_bonds * (self.tot_steps-1)):
            list_for_decision.append(np.random.random(1))
        res_arr = np.array(list_for_decision)
        return res_arr

class HostNetwork_6f:
    def __init__(self,init,tw,L,M,N,tot_steps,beta,timestamp):
        """Since Yoshino_3.0, when update the energy, we do not calculate energy for all the gaps, but only calculate these part affected by the flipping of a SPIN (S)  or shifting of 
           a BOND (J). This will accelerate the calculation by hundreds of times. (2) Besides, we note that we do NOT need to define a functon: remain(), which records
           the new MC steps' S, J and H, even though one MC move is rejected."""
        # Parameters used in the host machine
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

        # Energy difference caused by update of variables J or S 
        self.delta_H= 0 

        self.num_hidden_node_layers = self.L - 1 # After distingush num_hidden_node_layers and num_hidden_bond_layers, then I found the clue is much clear!
        self.num_hidden_bond_layers = self.L - 2
        #Intialize S and J by the array S and J. 
        #Note: Both S and J are the coordinates of a machine.
        self.S = init_S(self.M,self.num_hidden_node_layers,self.N)
        self.J = init_J(self.L, self.N) # L layers 
        self.r = 0 # the initial gap is returned from a function.

        self.new_S = copy.copy(self.S) # for storing temperay array when update 
        self.new_J = copy.copy(self.J) # for storing temperay array when update  
        self.new_r = copy.copy(self.r)

        self.count_MC_step = 0           

    def gap_init(self):
        '''Ref: Yoshino2019, eqn (31b)'''
        r = np.zeros((self.M,self.L,self.N))
        for mu in range(self.M):
            for l in range(1,self.L): # l = 0,...,L-1
                for n2 in range(self.N):
                    r[mu,l,n2] = (np.sum(self.J[l,n2,:] * self.S[mu,l-1,:])/np.sqrt(self.N)) * self.S[mu,l,n2] 
        return r    
    def flip_spin(self,mu,l,n):
        '''flip_spin() will flip S at a given index tuple (l,mu,n). We add l,mu,n as parameters, for parallel programming. Note: any spin can be update except the input/output.'''
        # Update self.new_S
        self.new_S = copy.deepcopy(self.S)
        self.new_S[mu][l][n] = -1 * self.S[mu][l][n]  
    def shift_bond(self,l,n2,n1):
        '''shift_bond() will shift the element of J with a given index to another value. We add l,n2,n1 as parameters, for parallel programming..'''
        self.new_J = copy.copy(self.J)
        x = np.random.normal(loc=0,scale=1.0,size=1) 
        # scale denotes standard deviation; 
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J[l][n2][n1] = (self.J[l][n2][n1] + x * rat) / RESCALE_J
        # rescaling 
        t = self.new_J[l][n2] 
        tt = t*t
        N_prim = tt.sum()
        SCALE_J = np.sqrt(N_prim/(self.N))
        print("The sum of J_ij^2:{}: [OLD] ".format(N_prim))
        self.new_J[l][n2][n1] = self.new_J[l][n2][n1]/SCALE_J
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
        part_gap = np.zeros(self.N + 1)
        part_gap[0] = (np.sum( self.J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n] 
        for n2 in range(self.N):
            part_gap[1+n2] = (np.sum(self.J[l+1,n2,:] * self.S[mu,l,:])/SQRT_N) * self.S[mu,l+1,n2] 
        return part_gap  # Only the N+1 elements affect the Delta_H_eff. 
    def part_gap_after_flip(self,mu,l,n):
        part_gap = np.zeros(self.N + 1)
        part_gap[0] = (np.sum( self.J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.new_S[mu,l,n] 
        for n2 in range( self.N):
            part_gap[1+n2] = (np.sum(self.J[l+1,n2,:] * self.new_S[mu,l,:])/SQRT_N) * self.S[mu,l+1,n2] 
        return part_gap  # Only the N+1 elements affect the Delta_H_eff. 
    def part_gap_before_shift(self,l,n): 
        part_gap = np.zeros(self.M)
        for mu in range(self.M):
            part_gap[mu] = (np.sum(self.J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n] 
        return part_gap  # Only the M elements affect the Delta_H_eff. 
    def part_gap_after_shift(self,l,n): 
        part_gap = np.zeros(self.M)
        for mu in range(self.M):
            part_gap[mu] = (np.sum(self.new_J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n] 
        return part_gap # Only the M elements affect the Delta_H_eff. 
    def decision_by_mu_l_n(self,MC_index,mu,l,n):
        self.delta_H = calc_ener(self.part_gap_after_flip(mu,l,n)) - calc_ener(self.part_gap_before_flip(mu,l,n))
        delta_e = self.delta_H
        if delta_e < 0:
            self.accept_by_mu_l_n(mu,l,n) 
        else:
            if np.random.random(1) < np.exp(-delta_e * self.beta):
                self.accept_by_mu_l_n(mu,l,n)
            else:
                pass
    def decision_by_l_n2_n1(self,MC_index,l,n2,n1):
        self.delta_H = calc_ener(self.part_gap_after_shift(l,n2)) - calc_ener(self.part_gap_before_shift(l,n2))
        delta_e = self.delta_H
        if delta_e < 0:
            self.accept_by_l_n2_n1(l,n2,n1) 
        else:
            if np.random.random(1) < np.exp(-delta_e * self.beta):
                self.accept_by_l_n2_n1(l,n2,n1)
            else:
                pass
    def update_spin(self,ind): 
        self.flip_spin(ind[0],ind[1],ind[2])
        self.decision_by_mu_l_n(MC_index,ind[0],ind[1],ind[2])


