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

        # Three arrays for storing MC trajectories of S, J and H
        self.J_traj = 0 
        self.S_traj = 0 
        self.H_hidden_traj = None 

        self.H = 0 # for storing energy
        self.new_H = 0 # for storing temperay energy when update

        # Energy difference caused by update of sample mu
        self.delta_H= 0

        # Intialize S and J by the array S and J.
        # We use S, instead of S_hidden, because S[0], S[L-2] still have interaction to J_in and J_out. Ref. Yoshino eq (8).
        self.S = init_S(self.M,self.num_hidden_node_layers,self.N)
        self.J_hidden = init_J(self.num_hidden_bond_layers,self.N)
        # Define J_in and J_out
        self.S_in = np.empty((self.M,self.N_in))  
        self.J_in = init_J_in(self.N,self.N_in)  
        self.S_out = np.empty((self.M,self.N_out))  
        self.J_out = init_J_out(self.N_out,self.N)
        self.new_S = 0 # for storing temperay array when update
        self.new_J = 0 # for storing temperay array when update

        self.new_J_in = np.empty((self.N,self.N_in)) 
        self.new_J_out = np.empty((self.N_out,self.N))  
      
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

