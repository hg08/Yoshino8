#======
#Module
#======
import copy
import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import random
import scipy as sp
import tensorflow as tf

#=========
# Functions
#=========
def generate_coord(N):
    """Randomly set the initial coordinates,whose components are 1 or -1."""
    v = np.zeros(N,dtype='int8')
    list_spin = [-1,1]
    for i in range(N):
        v[i] = np.random.choice(list_spin)
    return v

def generate_J(N):
    J = generate_rand_normal_numbers(N)
    return J

def generate_rand_normal_numbers(N=4,mu=0,sigma=1):
    """生成N个满足正态分布的随机数,平均值mu,方差sigma."""
    samp = np.random.normal(loc=mu,scale=sigma,size=N)
    return samp

def generate_S_in_and_out(init,M,N_in,N_out):
    """Generate S_in and S_out from tf.keras. S_in and S_out are fixed for all the machines."""
    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    #================================
    # Generate new and CLEAN datasets 
    new_train_X = generate_new_train_X(train_X,train_y,M,init)
    new_train_y = generate_new_train_y(train_y,M,init)
    #=========================================================
    # Generate S_in. Since each figure is a 2D array, new_train_M[:M] is a 3D array. We reshpae the 3D array into 2D array.
    S_in = new_train_X[:M].reshape(-1,N_in)
    # Generate the corresponding S_out, which is a fixed 2D-array, during this epoch of training.
    labels = new_train_y
    S_out = digit_arr2unitvec_arr(labels,N_out)
    return (S_in,S_out)

def generate_random_S_in_and_out(init,M,N_in,N_out):
    """Generate S_in and S_out from self-defined function. S_in and S_out are fixed for all the machines.
       Each site can be +1 or -1.
    """
    # Generate S_in. 
    size_in = M * N_in
    size_out = M * N_out

    S_in = np.random.choice(2, size_in).reshape((M, N_in))
    S_in[S_in < 0.5] = -1

    S_out = np.random.choice(2, size_out).reshape((M, N_out))
    S_out[S_out < 0.5] = -1 

    return (S_in,S_out)

def mergesort(arr1d):
    """mergesort is stable, O(nlogn)."""
    arr = np.sort(arr1d, axis=-1, kind='mergesort')
    return arr
def timsort(arr1d):
    """timsort is stable, O(nlogn)."""
    arr = np.sort(arr1d, axis=-1, kind='timsort')
    return arr
def init_J(L,N):
    """This function normalize the generated array J[i,j,:] to sqrt(tmp), and
       then make sure the constraint J[i,j,:]*J[i,j,:]=N are satisfied exactly.
    """
    J = np.zeros((L,N,N),dtype='float32')
    for i in range(L):
        for j in range(N):         
            #First, generate N random numbers
            J[i,j,:] = generate_rand_normal_numbers(N)
            # Add the constraint np.sum(J[i,j,:] * J[i,j,:]) = N.
            N_prime= np.sum(J[i,j,:] * J[i,j,:])
            J[i,j,:] = J[i,j,:] * np.sqrt(N / N_prime) 
            ##TEST:
            print("J_hidd * J_hidd:")
            print(N_prime)
            print("J_hidd * J_hidd:(normed)")
            print(np.sum(J[i,j,:] * J[i,j,:]))
            print("J_hidd[i,j,:]:")
            print(J[i,j,:])
    return J 

def init_J_v00(L,N):
    """This function normalize the generated array J[i,j,:] to sqrt(tmp), and
     then make sure the constraint J[i,j,:]*J[i,j,:]=N are satisfied exactly."""
    J = np.zeros((L,N,N)) # axis=1 labels the backward-nodes (标记后一层结点), while axis=2 labels the forward-nodes (标记前一层结点)
    # Set the first layer J_{0,x}^{y} = 0 (x, y are any index.)
    for i in range(1,L):
        for j in range(N):         
            #First, generate N random numbers
            J[i,j,:] = generate_rand_normal_numbers(N)
            # Add the constraint np.sum(J[i,j,:] * J[i,j,:]) = N.
            tmp = np.sum(J[i,j,:] * J[i,j,:])
            J[i,j,:] = (J[i,j,:]/np.sqrt(tmp)) * np.sqrt(N) 
    return J 

def init_J_in(N,N_in):
    """Should I define this array J_in? If I should define, should I normalize it?
       Assumption 1: we assume that J_in is normalized.
    """
    J_in = np.zeros((N,N_in),dtype='float32')
    # N_in is the number of nodes for the 0-th layer
    # N is the number of nodes for the first layer
    for j in range(N):         
        #First, generate N_in random numbers
        J_in[j,:] = generate_rand_normal_numbers(N_in)
        # Add the constraint np.sum(J[j,:] * J[j,:]) = N_in.
        N_prime = np.sum(J_in[j,:] * J_in[j,:])
        J_in[j,:] = J_in[j,:] * np.sqrt(N_in / N_prime) 
        ##TEST:
        print("J_in * J_in:")
        print(N_prime)
        print("J_in * J_in:(normed)")
        print(np.sum(J_in[j,:] * J_in[j,:]))
        #print("J_in[j,:]")
        #print(J_in[j,:])
    return J_in 

def init_J_out(N_out,N):
    """Should I define this array J_out? If I should define, should I normalize it?
       Assumption 2: we assume that J_out is normalized."""
    # N_out: number of classes (neurons) in the last layer, i.e.,
    # N_out is the number of nodes for the L-th (the last) layer
    # N is the number of nodes for the (L-1)-th (the second-last) layer
    J_out = np.zeros((N_out,N),dtype='float32')
    for j in range(N_out):         
        #First, generate N random numbers
        J_out[j,:] = generate_rand_normal_numbers(N)
        # Add the constraint np.sum(J[j,:] * J[j,:]) = N.
        N_prime = np.sum(J_out[j,:] * J_out[j,:])
        J_out[j,:] = J_out[j,:] * np.sqrt(N / N_prime) 
        #TEST:
        print("J_out * J_out:")
        print(N_prime)
        print("J_out * J_out:(normed)")
        print(np.sum(J_out[j,:] * J_out[j,:]))
        #print("J_out[j,:] :")
        #print(J_out[j,:])
    return J_out
 
def init_S(M,L,N):
    # M: number of samples
    # L: number of hidden layers.
    # N: number of neurons in each layer
    S = np.zeros((M,L,N),dtype='int8')
    S = np.reshape(generate_coord(M*L*N),(M,L,N))
    #For testing
    print("[For testing the validation of S] Ratio of positive S and negative S: {:7.6f}".format(S[S<0].size / S[S>0].size))
    return S


def ener_for_mu(r_mu):
    '''Ref: Yoshino2019, eqn (31a)
       energy for single sample (mu).'''
    H_mu = soft_core_potential(r_mu).sum()
    return H_mu
def soft_core_potential(h):
    '''Ref: Yoshino2019, eqn (32)
       This function is tested and it is correct.
    '''
    x2 = 1.0
    return  np.heaviside(-h, x2)*(h**2) 
    # The same as the following.
    #epsilon = 1
    #return epsilon * np.heaviside(-h, x2) * np.power(h,2)
    #return  np.heaviside(-h, x2) * np.power(h,2) 

def calc_ener(r):
    '''Ref: Yoshino2019, eqn (31a)'''
    # The argument r can be any array
    H = soft_core_potential(r).sum()
    return H
#======================================================================================================
#The following three functions are used for get the argument of locations of the initial configurations
#======================================================================================================
def list_only_dir(directory):
    """This function list the directorys only, under a given direcotry."""
    import os
    list_dir = next(os.walk(directory))[1]
    full_li = []
    #directory = '../data'
    for item in list_dir:
        li = [directory,item]
        full_li.append("/".join(li))
    return full_li
def list_only_naked_dir(directory):
    """This function list the naked directory names only, under a given direcotry."""
    import os
    list_dir = next(os.walk(directory))[1]
    for item in list_dir:
        li = [directory,item]
    return list_dir
def str2int(list_dir):
    res = []
    for item in range(len(list_dir)):
        res.append(int(list_dir[item]))
    return res

def digit2unitvec(i,N=10):
    '''
    Convert a number (0 to 9) to a unit vector in 10-D Euclidean space.
    '''
    import numpy as np
    vec = np.zeros(N)
    vec[i] = 1
    return vec
    
def digit_arr2unitvec_arr(arr,N=10):
    '''
    Convert an array (1D) of numbers (0 to 9) to an array of unit vector in 10-D Euclidean space.
    N=10, is the total number of digits:0,1,...,9.
    '''
    import numpy as np
    M = arr.shape[0]
    vec = np.zeros((M,N))
    print("vec.shape:")
    print(vec.shape)
    for i, term in enumerate(arr):
        vec[i][int(term)] = 1
    return vec      

def digit_arr2spindist_arr(arr,N=10):
    '''
    Convert an array (1D) of numbers (0 to 9) to an array of spins. The difference between digit_arr2unitvec_arr() and digit_arr2spindist_arr() can be shown in the following example:
    if the result of digit_arr2unitvec_arr() is [[1,0,0,0,0,0,0,0,0,0],...] then the result of digit_arr2spindist_arr() is [[1,-1,-1,-1,-1,-1,-1,-1,-1,-1],...].
    N=10, is the total number of digits:0,1,...,9.
    '''
    import numpy as np
    M = arr.shape[0]
    #print(arr.shape)
    #print(M)
    vec = -np.ones((M,N))
    print("vec.shape:")
    print(vec.shape)
    for i, term in enumerate(arr):
        vec[i][int(term)] = 1
    return vec      

# Calculate the relaxation time tau 
def relaxation_time_p(corr,tot_steps):
    """
    Calculate the relaxation time of a correlation function. 
    Input: a 1-array.
    Output: a real value.
    """
    #from numpy.polynomial import Chebyshev
    from numpy.polynomial import Polynomial
    cc = 0.3679 # critical value of corr, 1/e
    #Generate x_list
    x_list = np.zeros_like(corr)
    for j,iterm in enumerate(corr):
        x_list[j] = 2 ** j
    corr = corr - cc
    p = Polynomial.fit(x_list, corr, 5, domain=[0,tot_steps], window=[0,1])
    res = p.roots()
    return res[0]

# Calculate the relaxation time tau 
def relaxation_time_DATE(corr):
    """
    Calculate the relaxation time of a correlation function. 
    Input: a 1-array.
    Output: a real value.
    #NOTE: 
    #WE do NOT use this funtion, because the denominator in some formula may be quite SMALL, then the result will be not accurate!
    """
    cc = 0.3679 # critical value of corr, 1/e
    #Generate x_list
    x_list = np.zeros_like(corr)
    for j,iterm in enumerate(corr):
        x_list[j] = 2 ** j

    if min(corr) > cc:
        for i,iterm in enumerate(corr):
            # determine tau by interpotation. I have to use a more accuarate method to calculate the relaxation time.
            return (cc - corr[i]) * (x_list[i-1]-x_list[i])/(corr[i-1]-corr[i]) + x_list[i]
    else:
        for i in range(corr.shape[0]):
            print("corr[i]")
            print(corr[i])
            if corr[i] < cc:
                # determine tau by interpotation. I have to use a more accuarate method to calculate the relaxation time.
                return ((corr[i-1] - cc) * (x_list[i]-x_list[i-1]) ) / (corr[i-1]-corr[i])  + x_list[i-1]
            else:
                pass

# Calculate the relaxation time tau (2nd method) 
def relaxation_time(corr):
    """
    Calculate the relaxation time of a correlation function. 
    Input: a 1-array.
    Output: a real value.
    This method is only defined for 2**k time series.
    """
    # Const.s
    cc = 0.3679 # critical value of corr, 1/e
    n = 3

    #Generate x_list
    x_list = np.zeros_like(corr)
    for j,iterm in enumerate(corr):
        x_list[j] = 2 ** j

    if min(corr) > cc:
        # Use LSF: https://www.varsitytutors.com/hotmath/hotmath_help/topics/line-of-best-fit
        #1. select an integer n
        #2. calculate the mean of last n elements of the array x, and y 
        x_mean = np.mean(x_list[-n:])
        y_mean = np.mean(corr[-n:])
        #3. calculat the slope and intersept
        k = np.sum((x_list[-n:]-x_mean) * (corr[-n:]-y_mean)) / np.sum((x_list[-n:]-x_mean)*(x_list[-n:]-x_mean))
        b = y_mean - k * x_mean
        #4. calculate tau
        tau = (cc - b)/k
        return tau
    else:
        for i in range(corr.shape[0]):
            print("corr[i]")
            print(corr[i])
            if corr[i] < cc:
                # Use LSF: https://www.varsitytutors.com/hotmath/hotmath_help/topics/line-of-best-fit
                #1. select an integer n
                #2. calculate the mean of last n elements of the array x, and y 
                x_mean = np.mean(x_list[(i+1)-n:i+1])
                y_mean = np.mean(corr[(i+1)-n:i+1])
                #3. calculat the slope and intersept
                k = np.sum((x_list[i+1-n:i+1]-x_mean) * (corr[i+1-n:i+1]-y_mean)) / np.sum((x_list[i+1-n:i+1]-x_mean)*(x_list[i+1-n:i+1]-x_mean))
                b = y_mean - k * x_mean
                #4. calculate tau
                tau = (cc - b)/k
                return tau
                #=====
                #Note:
                #=====
                #WE do NOT use the follow formula because the denominator may be quite small, then the result will be not accurate!
                #return ((corr[i-1] - cc) * (2**i-2**(i-1)) ) / (corr[i-1]-corr[i])  + 2**(i-1)
                 
def generate_new_train_X(train_X,train_y,M,init):
    """Generate a clean array new_train_X, which have equal number of each digit. 
    train_X: the train_X from MNIST dataset;
    train_y: the train_y from MNIST dataset;
    M: the total number of samples; 
    init : the configuration index. """
    Nout = 10 #The number of classes of digits
    a = int(M/Nout)
    train_X0 = train_X[train_y==0]
    train_X1 = train_X[train_y==1]
    train_X2 = train_X[train_y==2]
    train_X3 = train_X[train_y==3]
    train_X4 = train_X[train_y==4]
    train_X5 = train_X[train_y==5]
    train_X6 = train_X[train_y==6]
    train_X7 = train_X[train_y==7]
    train_X8 = train_X[train_y==8]
    train_X9 = train_X[train_y==9]

    train = np.zeros((M,28,28))
    train[0:a] = train_X0[init*a:(init+1)*a]
    train[a:2*a] = train_X1[init*a:(init+1)*a]
    train[2*a:3*a] = train_X2[init*a:(init+1)*a]
    train[3*a:4*a] = train_X3[init*a:(init+1)*a]
    train[4*a:5*a] = train_X4[init*a:(init+1)*a]
    train[5*a:6*a] = train_X5[init*a:(init+1)*a]
    train[6*a:7*a] = train_X6[init*a:(init+1)*a]
    train[7*a:8*a] = train_X7[init*a:(init+1)*a]
    train[8*a:9*a] = train_X8[init*a:(init+1)*a]
    train[9*a:10*a] = train_X9[init*a:(init+1)*a]
    return train

def generate_new_train_y(train_y,M,init):
    """Generate a clean array new_train_y corresponding to the clean array new_train_X.
    train_X: the train_X from MNIST dataset;
    train_y: the train_y from MNIST dataset;
    M: the total number of samples; 
    init: the configuration index. """
    Nout = 10 #The number of classes of digits
    a = int(M/Nout)
    train_y0 = train_y[train_y==0]
    train_y1 = train_y[train_y==1]
    train_y2 = train_y[train_y==2]
    train_y3 = train_y[train_y==3]
    train_y4 = train_y[train_y==4]
    train_y5 = train_y[train_y==5]
    train_y6 = train_y[train_y==6]
    train_y7 = train_y[train_y==7]
    train_y8 = train_y[train_y==8]
    train_y9 = train_y[train_y==9]
    train = np.zeros(M)
    train[0:a] = train_y0[init*a:(init+1)*a]
    train[a : 2*a] = train_y1[init*a:(init+1)*a]
    train[2*a : 3*a] = train_y2[init*a:(init+1)*a]
    train[3*a : 4*a] = train_y3[init*a:(init+1)*a]
    train[4*a : 5*a] = train_y4[init*a:(init+1)*a]
    train[5*a : 6*a] = train_y5[init*a:(init+1)*a]
    train[6*a : 7*a] = train_y6[init*a:(init+1)*a]
    train[7*a : 8*a] = train_y7[init*a:(init+1)*a]
    train[8*a : 9*a] = train_y8[init*a:(init+1)*a]
    train[9*a : 10*a] = train_y9[init*a:(init+1)*a]
    print("train:")
    print(train) # Testing
    return train

