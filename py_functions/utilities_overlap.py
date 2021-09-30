#=========================================
#Import the Module with self-defined class
#=========================================
import sys

sys.path.append('/public1/home/sc91981/py_functions/')
#sys.path.append('/home/gang/Github/Yoshino/py_functions/')

from utilities import *
from Network import l_list, step_list  

#==============
#Import Modules 
#==============
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl

#==========
# Functions
#==========
def autocorr(x):
    """Correlation Function."""
    x = np.array(x)
    length = np.size(x)
    c = np.ones(length)
    for i in range(1,length):
        c[i]=np.sum(x[:-i]*x[i:])/np.sum(x[:-i]*x[:-i])
    return c
def line2intlist(line):
    line_split=line.strip().split(' ')
    res_list = []
    for x in line_split:
        res_list.append(int(x))
    return res_list
def line2floatlist(line):
    line_split=line.strip().split(' ')
    res_list = []
    for x in line_split:
        res_list.append(float(x))
    return res_list
def get_N_HexCol(N=5):
    """Define N elegant colors and return a list of the N colors. Each element of the list is represented as a string.
       and it can be used as an argument of the kwarg color in plt.plot(), or plt.hist()."""
    import colorsys
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

def plot_overlap_S(q,index,init,tw,L,M,N,beta,tot_steps_,tot_steps):
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((-0.01,1.1))
    x = 0 
    b = -1
    num_color = 8
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],q[x+1][ex:],color_list[0],marker='.',label="l=1")
    ax.plot(time_arr[ex:],q[x+2][ex:],color_list[1],marker='*',label="l=2")
    ax.plot(time_arr[ex:],q[x+3][ex:],color_list[2],marker='o',label="l=3")
    ax.plot(time_arr[ex:],q[x+4][ex:],color_list[3],marker='v',label="l=4")
    ax.plot(time_arr[ex:],q[x+5][ex:],color_list[4],marker='^',label='l=5')
    ax.plot(time_arr[ex:],q[x+6][ex:],color_list[5],marker='<',label='l=6')
    ax.plot(time_arr[ex:],q[x+7][ex:],color_list[6],marker='>',label='l=7')
    ax.plot(time_arr[ex:],q[x+8][ex:],color_list[7],marker='1',label="l=8")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,l)$")
    ax.set_title("init={:d};tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(init,tw,L,M,N,beta))
    plt.savefig("../imag/Overlap_S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,init,tw,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Overlap_S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(index,init,tw,L,M,N,beta,tot_steps),format='png')

def plot_ave_overlap_S(q,index,tw,L,M,N,beta,tot_steps_,tot_steps):
    """overlap_S is the average one over different init's."""
    ex = 0
    num = M*L*N + L*N*N
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((-0.01,1.1))
    x = 0 
    b = -1
    num_color = 9 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],q[x+1][ex:],color_list[x+1],marker='.',label="l=1")
    ax.plot(time_arr[ex:],q[x+2][ex:],color_list[x+2],marker='*',label="l=2")
    ax.plot(time_arr[ex:],q[x+3][ex:],color_list[x+3],marker='o',label="l=3")
    ax.plot(time_arr[ex:],q[x+4][ex:],color_list[x+4],marker='v',label="l=4")
    ax.plot(time_arr[ex:],q[x+5][ex:],color_list[x+5],marker='^',label='l=5')
    ax.plot(time_arr[ex:],q[x+6][ex:],color_list[x+6],marker='<',label='l=6')
    ax.plot(time_arr[ex:],q[x+7][ex:],color_list[x+7],marker='>',label='l=7')
    ax.plot(time_arr[ex:],q[x+8][ex:],color_list[x+8],marker='1',label="l=8")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,l)$")
    ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,tw,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(index,tw,L,M,N,beta,tot_steps),format='png')

def plot_ave_overlap_S_yoshino_setting1(q,index,tw,L,M,N,beta,tot_steps_,tot_steps):
    """overlap_S is the average one over different init's."""
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((-0.01,1.1))
    x = -1 
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],q[x+1][ex:],color_list[x+1],marker='.',label="l=1")
    ax.plot(time_arr[ex:],q[x+2][ex:],color_list[x+2],marker='*',label="l=2")
    ax.plot(time_arr[ex:],q[x+3][ex:],color_list[x+3],marker='o',label="l=3")
    ax.plot(time_arr[ex:],q[x+4][ex:],color_list[x+4],marker='v',label="l=4")
    ax.plot(time_arr[ex:],q[x+5][ex:],color_list[x+5],marker='^',label='l=5')
    ax.plot(time_arr[ex:],q[x+6][ex:],color_list[x+6],marker='<',label='l=6')
    ax.plot(time_arr[ex:],q[x+7][ex:],color_list[x+7],marker='>',label='l=7')
    ax.plot(time_arr[ex:],q[x+8][ex:],color_list[x+8],marker='1',label="l=8")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,l)$")
    ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,tw,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(index,tw,L,M,N,beta,tot_steps),format='png')

def overlap_J(J_traj,J0_traj,J_in_traj,J0_in_traj,J_out_traj,J0_out_traj):
    '''overlap for J_traj and J0_traj Q(t,l). Here we bound J_traj, J_in_traj, J_out_traj all together.'''
    T = J_traj.shape[0]
    L = J_traj.shape[1] + 2
    N = J_traj.shape[-1]
    N_in = J_in_traj.shape[-1]
    N_out = J_out_traj.shape[-2]
    print("N_in:")
    print(N_in)
    print("N_out:")
    print(N_out)
    res = np.zeros((L,T),dtype='float32')
    for l in range(0,L):
        if l == 0:
            for t in range(T):
                res[l][t] = np.sum(J_in_traj[t,:,:] * J_in_traj[t,:,:])/(N*N_in)
                res[l][t] = round(res[l][t],5)
                print("l={}".format(l))
                print("res[l][t]: Q_in")
                print(res[l][t])
        elif l == L-1:
            for t in range(T):
                res[l][t] = np.sum(J_out_traj[t,:,:] * J0_out_traj[t,:,:])/(N_out * N)
                res[l][t] = round(res[l][t],5)
                print( "l={}".format(l))
                print( "res[{}][{}]: Q_out)".format(l,t))
                print(res[l][t])
        else:
            l0 = l - 1 # l0 is the index in J, ie., J_hidden.
            for t in range(T):
                res[l][t] = np.sum(J_traj[t,l0,:,:] * J0_traj[t,l0,:,:] )/(N**2)
                res[l][t] = round(res[l][t],5)
                print("l={}".format(l))
                print("res[{}][{}]: Q_hidden".format(l,t))
                print(res[l][t])
def overlap_J_hidden(J_traj,J0_traj):
    '''overlap for J_traj and J0_traj Q(t,l).'''
    T = J_traj.shape[0]
    L_hidden = J_traj.shape[1] 
    N = J_traj.shape[-1]
    res = np.zeros((L_hidden,T),dtype='float32')
    for l in range(L_hidden):
        for t in range(T):
            res[l][t] = np.sum(J_traj[t,l,:,:] * J0_traj[t,l,:,:] )/(N**2)
            res[l][t] = round(res[l][t],5)
            print("l={}".format(l))
            print("res[{}][{}]: Q_hidden".format(l,t))
            print(res[l][t])
    return res

def overlap_J_6f(J_traj,J0_traj):
    '''overlap for J_traj and J0_traj Q(t,l).'''
    T = J_traj.shape[0]
    L = J_traj.shape[1] 
    N = J_traj.shape[-1]
    res = np.zeros((L,T),dtype='float32')
    for l in range(0,L):
        for t in range(T):
            res[l][t] = np.sum(J_traj[t,l,:,:] * J0_traj[t,l,:,:] )/(N**2)
            res[l][t] = round(res[l][t],5)
            print("l={}".format(l))
            print("res[{}][{}]: Q".format(l,t))
            print(res[l][t])
    return res

def overlap_S(S_traj,S0_traj):
    '''overlap for S_traj and S0_traj q(t,l).'''
    T = S_traj.shape[0]
    M = S_traj.shape[1]
    L = S_traj.shape[2]
    N = S_traj.shape[-1]
    res = np.zeros((L,T),dtype='float32')
    for l in range(0,L): # we do not need the spins in the input layer
        for t in range(T):
            res[l][t] = np.sum(S_traj[t,:,l,:] * S0_traj[t,:,l,:])/(N*M)
            print("l={}".format(l))
            print("res[l][t]: q")
            print(res[l][t])
    return res

#def overlap_JJ0(JJ0_traj,JJ0_in_traj,JJ0_out_traj):
#    '''overlap for J_traj and J0_traj Q(t,l).'''
#    T = JJ0_traj.shape[0]
#    L = JJ0_traj.shape[1] + 2
#    N = JJ0_traj.shape[-1]
#    N_in = JJ0_in_traj.shape[-1]
#    N_out = JJ0_out_traj.shape[-2]
#    res = np.zeros((L,T),dtype='float32')
#    for l in range(0,L):
#        if l == 0:
#            for t in range(T):
#                res[l][t] = np.sum(JJ0_in_traj[t,:,:])/(N*N_in)
#        elif l == L-1:
#            for t in range(T):
#                res[l][t] = np.sum(JJ0_out_traj[t,:,:])/(N*N_out)
#        else:
#            l0 = l - 1
#            for t in range(T):
#                res[l][t] = np.sum(JJ0_traj[t,l0,:,:])/(N**2)
#    return res
#
#def overlap_SS0(SS0_traj):
#    '''overlap for S_traj and S0_traj q(t,l).'''
#    T = SS0_traj.shape[0]
#    M = SS0_traj.shape[1]
#    L = SS0_traj.shape[2]
#    N = SS0_traj.shape[-1]
#    res = np.zeros((L,T),dtype='float32')
#    for l in range(0,L): # we do not need the spins in the input layer
#        for t in range(T):
#            res[l][t] = np.sum(SS0_traj[t,:,l,:])/(N*M)
#    return res

def plot_overlap_J_tw(ol0,ol1,ol2,ol3,ol4,ol5,ol6,timestamp,init,l_index,L,M,N,beta,tot_steps_,tot_steps):
    """olX: the overlap of J. The overlap is not averaged over init yet."""
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    #plt.yscale('log')
    plt.ylim((-0.01,1.1))
    x = -1 # Define a tempary integer
    num_color = 7
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    #ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("init={:d};l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(init,l_index,L,M,N,beta))
    plt.savefig("../imag/Overlap_J_hidden_{:s}_init{:d}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,init,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Overlap_J_hidden_{:s}_init{:d}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(timestamp,init,l_index,L,N,beta,tot_steps),format='png')

def plot_overlap_S_tw(ol0,ol1,ol2,ol3,ol4,ol5,ol6,index,init,l_index,L,M,N,beta,tot_steps_,tot_steps,tw_list):
    """olX: the overlap of S. The overlap is not averaged over init yet."""
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    #plt.yscale('log')
    plt.ylim((-0.01,1.1))
    x = -1 # Define a tempary integer
    num_color = 7 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    #ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,{:d})$".format(l_index))
    ax.set_title("init={:d};l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(init,l_index,L,M,N,beta))
    plt.savefig("../imag/Overlap_S_{:s}_init{:d}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,init,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Overlap_S_{:s}_init{:d}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(index,init,l_index,L,N,beta,tot_steps),format='png')

def plot_overlap_J_tw_ave_over_init(ol0,ol1,ol2,ol3,ol4,ol5,ol6,timestamp,l_index,L,M,N,beta,tot_steps_,tot_steps,tw_list):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((-0.01,1.1))
    x = -1 # Define a tempary integer
    num_color = 7
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    #ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(timestamp,l_index,L,N,beta,tot_steps),format='png')

def plot_overlap_S_tw_ave_over_init(ol0,ol1,ol2,ol3,ol4,ol5,ol6,index,l_index,L,M,N,beta,tot_steps_,tot_steps,tw_list):
    """olX: the overlap of S. The overlap is the averaged overlap over initial configurations."""
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((-0.01,1.1))
    x = -1 # Define a tempary integer
    num_color = 7 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    #ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(index,l_index,L,N,beta,tot_steps),format='png')

def plot_overlap_J_tw_X_ave_over_init(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,timestamp,l_index,L,M,N,beta,tot_steps_,tot_steps,tw_list):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((-0.01,1.1))
    x = -1 # Define a tempary integer
    num_color = 8
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(timestamp,l_index,L,N,beta,tot_steps),format='png')
def plot_overlap_J_tw_X_ave_over_init_yoshino_setting1(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,timestamp,l_index,L,M,N,beta,tot_steps_,tot_steps):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    ex = 0
    num = M*L*N + L*N*N 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((-0.01,1.1))
    x = -1 # Define a tempary integer
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color_list[0],marker='.',label="tw=0")
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color_list[1],marker='*',label="tw=4096")
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color_list[2],marker='o',label="tw=8192")
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color_list[3],marker='v',label="tw=65536") 
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color_list[4],marker='^',label="tw=262,000")
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color_list[5],marker='<',label="tw=524,000")
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color_list[6],marker='>',label="tw=1048,000")
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color_list[7],marker='1',label="tw=2097,000")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(timestamp,l_index,L,N,beta,tot_steps),format='png')

def plot_overlap_S_tw_X_ave_over_init(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,index,l_index,L,M,N,beta,tot_steps_,tot_steps,tw_list):
    """olX: the overlap of S. The overlap is the averaged overlap over initial configurations."""
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((-0.01,1.1))
    x = -1 # Define a tempary integer
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color_list[3],marker='v',label="tw={}".format(tw_list[3])) 
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(index,l_index,L,N,beta,tot_steps),format='png')

def plot_overlap_S_tw_X_ave_over_init_yoshino_setting1(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,index,l_index,L,M,N,beta,tot_steps_,tot_steps):
    """olX: the overlap of S. The overlap is the averaged overlap over initial configurations."""
    ex = 0 
    num = M*L*N + L*N*N 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((-0.01,1.1))
    x = -1 # Define a tempary integer
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color_list[0],marker='.',label="tw=0")
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color_list[1],marker='*',label="tw=4096")
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color_list[2],marker='o',label="tw=8192")
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color_list[3],marker='v',label="tw=65536") #262144, 524288, 1048576, 2097152
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color_list[4],marker='^',label="tw=262,000")
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color_list[5],marker='<',label="tw=524,000")
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color_list[6],marker='>',label="tw=1048,000")
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color_list[7],marker='1',label="tw=2097,000")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(index,l_index,L,N,beta,tot_steps),format='png')

def plot_overlap_J_tw_8_ave_over_init(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,timestamp,l_index,L,M,N,beta,tot_steps_,tot_steps):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((-0.01,1.1))
    x = -1 # Define a tempary integer
    num_color =8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color_list[0],marker='',label="tw=1024")
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color_list[1],marker='',label="tw=8192")
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color_list[2],marker='',label="tw=65536")
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color_list[3],marker='',label="tw=524288")
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color_list[4],marker='',label="tw=2097152")
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color_list[5],marker='',label="tw=4194304")
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color_list[6],marker='',label="tw=16777216")
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color_list[7],marker='',label="tw=33554432")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(timestamp,l_index,L,N,beta,tot_steps),format='png')

def plot_overlap_S_tw_8_ave_over_init(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,index,l_index,L,M,N,beta,tot_steps_,tot_steps):
    """olX: the overlap of S. The overlap is the averaged overlap over initial configurations."""
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((-0.01,1.1))
    x = -1 # Define a tempary integer
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color_list[0],marker='',label="tw=1024")
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color_list[1],marker='',label="tw=8192")
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color_list[2],marker='',label="tw=65536")
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color_list[3],marker='',label="tw=524288")
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color_list[4],marker='',label="tw=2097152")
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color_list[5],marker='',label="tw=4194304")
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color_list[6],marker='',label="tw=16777216")
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color_list[7],marker='',label="tw=33554432")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(index,l_index,L,N,beta,tot_steps),format='png')

def plot_overlap_J_tw_5_ave_over_init(ol0,ol1,ol2,ol3,ol4,timestamp,l_index,L,M,N,beta,tot_steps_,tot_steps):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    #plt.yscale('log')
    plt.ylim((-0.01,1.1))
    x = -1 # Define a tempary integer
    num_color = 7
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color_list[0],marker='',label="tw=0")
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color_list[1],marker='',label="tw=1024")
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color_list[2],marker='',label="tw=8192")
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color_list[3],marker='',label="tw=65536")
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color_list[4],marker='',label="tw=524288")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(timestamp,l_index,L,N,beta,tot_steps),format='png')

def plot_overlap_S_tw_5_ave_over_init(ol0,ol1,ol2,ol3,ol4,index,l_index,L,M,N,beta,tot_steps_,tot_steps):
    """olX: the overlap of S. The overlap is the averaged overlap over initial configurations."""
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((0.0,1.1))
    x = -1 # Define a tempary integer
    num_color = 7 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color_list[0],marker='',label="tw=0")
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color_list[1],marker='',label="tw=1024")
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color_list[2],marker='',label="tw=8192")
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color_list[3],marker='',label="tw=65536")
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color_list[4],marker='',label="tw=524288")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(index,l_index,L,N,beta,tot_steps),format='png')

def plot_overlap_J_tw_base10(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,timestamp,init,l_index,L,M,N,beta,tot_steps_,tot_steps):
    """olX: the overlap of J. The overlap is not averaged over init yet."""
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((0.0,1.1))
    x = -1 # Define a tempary integer
    num_color = 9 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color_list[1],marker='',label="tw=0")
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color_list[2],marker='',label="tw=10")
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color_list[3],marker='',label="tw=100")
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color_list[4],marker='',label="tw=1000")
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color_list[5],marker='',label="tw=10000")
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color_list[6],marker='',label="tw=100000")
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color_list[7],marker='',label="tw=1000000")
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color_list[8],marker='',label="tw=10000000")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("init={:d};l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(init,l_index,L,M,N,beta))
    plt.savefig("../imag/Overlap_J_hidden_{:s}_init{:d}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,init,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Overlap_J_hidden_{:s}_init{:d}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(timestamp,init,l_index,L,N,beta,tot_steps),format='png')

def plot_overlap_S_tw_base10(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,index,init,l_index,L,M,N,beta,tot_steps_,tot_steps):
    """olX: the overlap of S. The overlap is not averaged over init yet."""
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((0.0,1.1))
    x = -1 # Define a tempary integer
    num_color = 9 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color_list[1],marker='',label="tw=0")
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color_list[2],marker='',label="tw=10")
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color_list[3],marker='',label="tw=100")
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color_list[4],marker='',label="tw=1000")
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color_list[5],marker='',label="tw=10000")
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color_list[6],marker='',label="tw=100000")
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color_list[7],marker='',label="tw=1000000")
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color_list[8],marker='',label="tw=10000000")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,{:d})$".format(l_index))
    ax.set_title("init={:d};l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(init,l_index,L,M,N,beta))
    plt.savefig("../imag/Overlap_S_{:s}_init{:d}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,init,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Overlap_S_{:s}_init{:d}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(index,init,l_index,L,N,beta,tot_steps),format='png')

def plot_tau_J_tw_base10(grand_tau,index,init,L,M,N,beta,tot_steps):
    ex = 1
    shape = grand_tau.shape
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    #grand_tau = grand_tau/1000000
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[x+1][ex:],color_list[0],marker='',label="tw=0")
    ax.plot(l_list[ex:],grand_tau[x+2][ex:],color_list[1],marker='',label="tw=10")
    ax.plot(l_list[ex:],grand_tau[x+3][ex:],color_list[2],marker='',label='tw=100')
    ax.plot(l_list[ex:],grand_tau[x+4][ex:],color_list[3],marker='',label='tw=1000')
    ax.plot(l_list[ex:],grand_tau[x+5][ex:],color_list[4],marker='',label="tw=10000")
    ax.plot(l_list[ex:],grand_tau[x+6][ex:],color_list[5],marker='',label="tw=100000")
    ax.plot(l_list[ex:],grand_tau[x+7][ex:],color_list[6],marker='',label="tw=1000000")
    ax.plot(l_list[ex:],grand_tau[x+8][ex:],color_list[7],marker='',label="tw=10000000")
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_J$ ($10^6$)")
    ax.set_title("init={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(init,L,M,N,beta))
    plt.savefig("../imag/tau_J_hidden_{:s}_init{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,init,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_J_hidden_{:s}_init{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.png".format(index,init,L,M,N,beta,tot_steps),format='png')

def plot_tau_S_tw_base10(grand_tau,index,init,L,M,N,beta,tot_steps):
    ex = 1
    shape = grand_tau.shape
    #l_list = [1,2,3,4,5,6,7,8]
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    #grand_tau = grand_tau/10000
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[x+1][ex:],color_list[0],marker='',label="tw=0")
    ax.plot(l_list[ex:],grand_tau[x+2][ex:],color_list[1],marker='',label="tw=10")
    ax.plot(l_list[ex:],grand_tau[x+3][ex:],color_list[2],marker='',label='tw=100')
    ax.plot(l_list[ex:],grand_tau[x+4][ex:],color_list[3],marker='',label='tw=1000')
    ax.plot(l_list[ex:],grand_tau[x+5][ex:],color_list[4],marker='',label="tw=10000")
    ax.plot(l_list[ex:],grand_tau[x+6][ex:],color_list[5],marker='',label="tw=100000")
    ax.plot(l_list[ex:],grand_tau[x+7][ex:],color_list[6],marker='',label="tw=1000000")
    ax.plot(l_list[ex:],grand_tau[x+8][ex:],color_list[7],marker='',label="tw=10000000")
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_S$ ($10^4$)")
    ax.set_title("init={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(init,L,M,N,beta))
    plt.savefig("../imag/tau_S_{:s}_init{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,init,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_S_{:s}_init{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.png".format(index,init,L,M,N,beta,tot_steps),format='png')

def plot_tau_J_tw(grand_tau,index,init,L,M,N,beta,tot_steps):
    ex = 1
    shape = grand_tau.shape
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 7 
    color_list = get_N_HexCol(num_color)
    #grand_tau = grand_tau/1000000
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[x+1][ex:],color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(l_list[ex:],grand_tau[x+2][ex:],color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(l_list[ex:],grand_tau[x+3][ex:],color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(l_list[ex:],grand_tau[x+4][ex:],color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    ax.plot(l_list[ex:],grand_tau[x+5][ex:],color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(l_list[ex:],grand_tau[x+6][ex:],color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(l_list[ex:],grand_tau[x+7][ex:],color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_J$ ($10^6$)")
    ax.set_title("init={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(init,L,M,N,beta))
    plt.savefig("../imag/tau_J_hidden_{:s}_init{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,init,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_J_hidden_{:s}_init{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.png".format(index,init,L,M,N,beta,tot_steps),format='png')

def plot_tau_S_tw(grand_tau,index,init,L,M,N,beta,tot_steps):
    ex = 1
    shape = grand_tau.shape
    #l_list = [1,2,3,4,5,6,7,8]
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    #grand_tau = grand_tau/10000
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[x+1][ex:],color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(l_list[ex:],grand_tau[x+2][ex:],color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(l_list[ex:],grand_tau[x+3][ex:],color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(l_list[ex:],grand_tau[x+4][ex:],color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    ax.plot(l_list[ex:],grand_tau[x+5][ex:],color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(l_list[ex:],grand_tau[x+6][ex:],color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(l_list[ex:],grand_tau[x+7][ex:],color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_S$ ($10^4$)")
    ax.set_title("init={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(init,L,M,N,beta))
    plt.savefig("../imag/tau_S_{:s}_init{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,init,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_S_{:s}_init{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.png".format(index,init,L,M,N,beta,tot_steps),format='png')

def plot_tau_J_tw_X(grand_tau,index,L,M,N,beta,tot_steps,tw_list):
    ex = 0 
    shape = grand_tau.shape
    l_list = [1,2,3,4,5,6,7,8,9,10]
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[x+1][ex:],color_list[0],marker='',label="tw={}".format(tw_list[x+1]))
    ax.plot(l_list[ex:],grand_tau[x+2][ex:],color_list[1],marker='',label="tw={}".format(tw_list[x+2]))
    ax.plot(l_list[ex:],grand_tau[x+3][ex:],color_list[2],marker='',label='tw={}'.format(tw_list[x+3]))
    ax.plot(l_list[ex:],grand_tau[x+4][ex:],color_list[3],marker='',label='tw={}'.format(tw_list[x+4]))
    ax.plot(l_list[ex:],grand_tau[x+5][ex:],color_list[4],marker='',label="tw={}".format(tw_list[x+5]))
    ax.plot(l_list[ex:],grand_tau[x+6][ex:],color_list[5],marker='',label="tw={}".format(tw_list[x+6]))
    ax.plot(l_list[ex:],grand_tau[x+7][ex:],color_list[6],marker='',label="tw={}".format(tw_list[x+7]))
    ax.plot(l_list[ex:],grand_tau[x+8][ex:],color_list[7],marker='',label="tw={}".format(tw_list[x+8]))
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_J$ ($10^6$)")
    ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/tau_J_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_J_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.png".format(index,L,M,N,beta,tot_steps),format='png')

def plot_tau_S_tw_X(grand_tau,index,L,M,N,beta,tot_steps,tw_list):
    ex = 1
    shape = grand_tau.shape
    l_list = [0,1,2,3,4,5,6,7,8]
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[x+1][ex:-1],color_list[0],marker='',label="tw={}".format(tw_list[x+1]))
    ax.plot(l_list[ex:],grand_tau[x+2][ex:-1],color_list[1],marker='',label="tw={}".format(tw_list[x+2]))
    ax.plot(l_list[ex:],grand_tau[x+3][ex:-1],color_list[2],marker='',label='tw={}'.format(tw_list[x+3]))
    ax.plot(l_list[ex:],grand_tau[x+4][ex:-1],color_list[3],marker='',label='tw={}'.format(tw_list[x+4]))
    ax.plot(l_list[ex:],grand_tau[x+5][ex:-1],color_list[4],marker='',label="tw={}".format(tw_list[x+5]))
    ax.plot(l_list[ex:],grand_tau[x+6][ex:-1],color_list[5],marker='',label="tw={}".format(tw_list[x+6]))
    ax.plot(l_list[ex:],grand_tau[x+7][ex:-1],color_list[6],marker='',label="tw={}".format(tw_list[x+7]))
    ax.plot(l_list[ex:],grand_tau[x+8][ex:-1],color_list[7],marker='',label="tw={}".format(tw_list[x+8]))
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_S$ ($10^4$)")
    ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/tau_S_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_S_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.png".format(index,L,M,N,beta,tot_steps),format='png')

def plot_tau_J_tw_8(grand_tau,index,L,M,N,beta,tot_steps,tw_list):
    ex = 1
    shape = grand_tau.shape
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 9 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[x+1][ex:],color_list[0],marker='',label="tw={}".format(tw_list[0]))
    ax.plot(l_list[ex:],grand_tau[x+2][ex:],color_list[1],marker='',label="tw={}".format(tw_list[1]))
    ax.plot(l_list[ex:],grand_tau[x+3][ex:],color_list[2],marker='',label="tw={}".format(tw_list[2]))
    ax.plot(l_list[ex:],grand_tau[x+4][ex:],color_list[3],marker='',label="tw={}".format(tw_list[3]))
    ax.plot(l_list[ex:],grand_tau[x+5][ex:],color_list[4],marker='',label="tw={}".format(tw_list[4]))
    ax.plot(l_list[ex:],grand_tau[x+6][ex:],color_list[5],marker='',label="tw={}".format(tw_list[5]))
    ax.plot(l_list[ex:],grand_tau[x+7][ex:],color_list[6],marker='',label="tw={}".format(tw_list[6]))
    ax.plot(l_list[ex:],grand_tau[x+8][ex:],color_list[7],marker='',label="tw={}".format(tw_list[7]))
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_J$ ($10^6$)")
    ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/tau_J_hidden_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_J_hidden_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.png".format(index,L,M,N,beta,tot_steps),format='png')

def plot_tau_S_tw_8(grand_tau,index,L,M,N,beta,tot_steps,tw_list):
    ex = 1
    shape = grand_tau.shape
    #l_list = [1,2,3,4,5,6,7,8]
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 9 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[x+1][ex:],color_list[0],marker='',label="tw={}".format(tw_list[0]))
    ax.plot(l_list[ex:],grand_tau[x+2][ex:],color_list[1],marker='',label="tw={}".format(tw_list[1]))
    ax.plot(l_list[ex:],grand_tau[x+3][ex:],color_list[2],marker='',label="tw={}".format(tw_list[2]))
    ax.plot(l_list[ex:],grand_tau[x+4][ex:],color_list[3],marker='',label="tw={}".format(tw_list[3]))
    ax.plot(l_list[ex:],grand_tau[x+5][ex:],color_list[4],marker='',label="tw={}".format(tw_list[4]))
    ax.plot(l_list[ex:],grand_tau[x+6][ex:],color_list[5],marker='',label="tw={}".format(tw_list[5]))
    ax.plot(l_list[ex:],grand_tau[x+7][ex:],color_list[6],marker='',label="tw={}".format(tw_list[6]))
    ax.plot(l_list[ex:],grand_tau[x+8][ex:],color_list[7],marker='',label="tw={}".format(tw_list[7]))
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_S$ ($10^4$)")
    ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/tau_S_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_S_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.png".format(index,L,M,N,beta,tot_steps),format='png')

def plot_tau_J_l(grand_tau,index,L,M,N,beta,tot_steps,tw_list):
    """When l is fixed."""
    ex = 1
    shape = grand_tau.shape
    ls = tw_list
    fig = plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau.T
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(ls[ex:],grand_tau[x+1][ex:],color_list[0],marker='.',label="l=1")
    ax.plot(ls[ex:],grand_tau[x+2][ex:],color_list[1],marker='*',label="l=2")
    ax.plot(ls[ex:],grand_tau[x+3][ex:],color_list[2],marker='o',label="l=3")
    ax.plot(ls[ex:],grand_tau[x+4][ex:],color_list[3],marker='v',label="l=4")
    ax.plot(ls[ex:],grand_tau[x+5][ex:],color_list[4],marker='^',label='l=5')
    ax.plot(ls[ex:],grand_tau[x+6][ex:],color_list[5],marker='<',label='l=6')
    ax.plot(ls[ex:],grand_tau[x+7][ex:],color_list[6],marker='>',label='l=7')
    ax.plot(ls[ex:],grand_tau[x+8][ex:],color_list[6],marker='1',label='l=8')
    plt.legend(loc="lower left")
    plt.xlabel("$t_w$")
    plt.ylabel(r"$\tau_J$ ($10^6$)")
    ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/tau_J_l_hidden_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_J_l_hidden_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.png".format(index,L,M,N,beta,tot_steps),format='png')

def plot_tau_S_l(grand_tau,index,L,M,N,beta,tot_steps,tw_list):
    """When l is fixed."""
    ex = 1
    shape = grand_tau.shape
    #tw_list = [0,10,100,1000,10000,100000,1000000,10000000]
    #tw_list = [0,1024,4096,16384,65536,262144]
    ls = tw_list
    fig = plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau.T
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(ls[ex:],grand_tau[x+1][ex:],color_list[0],marker='.',label="l=1")
    ax.plot(ls[ex:],grand_tau[x+2][ex:],color_list[1],marker='*',label="l=2")
    ax.plot(ls[ex:],grand_tau[x+3][ex:],color_list[2],marker='o',label="l=3")
    ax.plot(ls[ex:],grand_tau[x+4][ex:],color_list[3],marker='v',label="l=4")
    ax.plot(ls[ex:],grand_tau[x+5][ex:],color_list[4],marker='^',label="l=5")
    ax.plot(ls[ex:],grand_tau[x+6][ex:],color_list[5],marker='<',label="l=6")
    ax.plot(ls[ex:],grand_tau[x+7][ex:],color_list[6],marker='>',label='l=7')
    ax.plot(ls[ex:],grand_tau[x+8][ex:],color_list[7],marker='1',label='l=8')
    plt.legend(loc="lower left")
    plt.xlabel("$t_w$")
    plt.ylabel(r"$\tau_S$ ($10^4$)")
    ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/tau_S_l_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_S_l_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.png".format(index,L,M,N,beta,tot_steps),format='png')

def plot_tau_J_tw_ave_over_init(grand_tau,L,M,N,beta,tot_steps,tw_list):
    """In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations."""
    ex = 1
    shape = grand_tau.shape
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,100.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 7 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[x+1][ex:],color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(l_list[ex:],grand_tau[x+2][ex:],color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(l_list[ex:],grand_tau[x+3][ex:],color_list[2],marker='o',label='tw={}'.format(tw_list[2]))
    ax.plot(l_list[ex:],grand_tau[x+4][ex:],color_list[3],marker='v',label='tw={}'.format(tw_list[3]))
    ax.plot(l_list[ex:],grand_tau[x+5][ex:],color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(l_list[ex:],grand_tau[x+6][ex:],color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(l_list[ex:],grand_tau[x+7][ex:],color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_J$ ($10^6$)")
    ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/Ave_tau_J_hidden_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_tau_J_hidden_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.png".format(L,M,N,beta,tot_steps),format='png')

def plot_tau_S_tw_ave_over_init(grand_tau,L,M,N,beta,tot_steps,tw_list):
    """In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations."""
    ex = 1
    shape = grand_tau.shape
    #l_list = [1,2,3,4,5,6,7,8]
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[x+1][ex:],color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(l_list[ex:],grand_tau[x+2][ex:],color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(l_list[ex:],grand_tau[x+3][ex:],color_list[2],marker='o',label='tw={}'.format(tw_list[2]))
    ax.plot(l_list[ex:],grand_tau[x+4][ex:],color_list[3],marker='v',label='tw={}'.format(tw_list[3]))
    ax.plot(l_list[ex:],grand_tau[x+5][ex:],color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(l_list[ex:],grand_tau[x+6][ex:],color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(l_list[ex:],grand_tau[x+7][ex:],color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_S$")
    ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/Ave_tau_S_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_tau_S_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.png".format(L,M,N,beta,tot_steps),format='png')

def plot_ggrand_tau_J_tw_ave_over_init(tau,l,L,N,beta,alpha_list,tw_list):
    """In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations.
       we plot the relation between tau(alpha), where tau are the averaged tau over different initial configurations.
    """
    ex = 0
    shape = tau.shape
    print("shape of ggrand_tau_J:")
    print(shape)
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 7 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(alpha_list[ex:],tau[x+1][l],color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(alpha_list[ex:],tau[x+2][l],color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(alpha_list[ex:],tau[x+3][l],color_list[2],marker='o',label='tw={}'.format(tw_list[2]))
    ax.plot(alpha_list[ex:],tau[x+4][l],color_list[3],marker='v',label='tw={}'.format(tw_list[3]))
    ax.plot(alpha_list[ex:],tau[x+5][l],color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(alpha_list[ex:],tau[x+6][l],color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\tau_J$")
    ax.set_title("l={:d};N={:d}; beta={:3.1f}".format(l,N,beta))
    plt.savefig("../imag/Ave_tau_J_hidden_l{:d}_L{:d}_N{:d}_beta{:2.0f}.eps".format(l,L,N,beta),format='eps')
    plt.savefig("../imag/Ave_tau_J_hidden_l{:d}_L{:d}_N{:d}_beta{:2.0f}.png".format(l,L,N,beta),format='png')

def plot_ggrand_tau_S_tw_ave_over_init(tau,l,L,N,beta,alpha_list,tw_list):
    """In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations."""
    ex = 0
    shape = tau.shape
    print("shape of ggrand_tau_S:")
    print(shape)
    #l_list = [1,2,3,4,5,6,7,8]
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(alpha_list[ex:],tau[x+1][l],color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(alpha_list[ex:],tau[x+2][l],color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(alpha_list[ex:],tau[x+3][l],color_list[2],marker='o',label='tw={}'.format(tw_list[2]))
    ax.plot(alpha_list[ex:],tau[x+4][l],color_list[3],marker='v',label='tw={}'.format(tw_list[3]))
    ax.plot(alpha_list[ex:],tau[x+5][l],color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(alpha_list[ex:],tau[x+6][l],color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\tau_S$")
    ax.set_title("l={:d};N={:d}; beta={:3.1f}".format(l,N,beta))
    plt.savefig("../imag/Ave_grand_tau_S_l{:d}_L{:d}_N{:d}_beta{:2.0f}.eps".format(l,L,N,beta),format='eps')
    plt.savefig("../imag/Ave_grand_tau_S_l{:d}_L{:d}_N{:d}_beta{:2.0f}.png".format(l,L,N,beta),format='png')
def plot_ggrand_tau_S_tw_ave_over_init_fixed_tw(tau,selected_l_list,L,N,beta,alpha_list,tw_list,tw):
    """In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations."""
    ex = 0
    shape = tau.shape
    print("shape of ggrand_tau_S:")
    print(shape)
    #l_list = [1,2,3,4,5,6,7,8]
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    index_tw = tw_list.index(tw)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(alpha_list[ex:],tau[index_tw][selected_l_list[0]],color_list[0],marker='.',label="l={}".format(selected_l_list[0]+1))
    ax.plot(alpha_list[ex:],tau[index_tw][selected_l_list[1]],color_list[1],marker='*',label="l={}".format(selected_l_list[1]+1))
    ax.plot(alpha_list[ex:],tau[index_tw][selected_l_list[2]],color_list[2],marker='o',label='l={}'.format(selected_l_list[2]+1))
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\tau_S$")
    ax.set_title("tw={:d};N={:d}; beta={:3.1f}".format(tw,N,beta))
    plt.savefig("../imag/Ave_grand_tau_S_tw{:d}_L{:d}_N{:d}_beta{:2.0f}.eps".format(tw,L,N,beta),format='eps')
    plt.savefig("../imag/Ave_grand_tau_S_tw{:d}_L{:d}_N{:d}_beta{:2.0f}.png".format(tw,L,N,beta),format='png')

def plot_ggrand_tau_J_tw_X_ave_over_init(tau,l,L,N,beta,alpha_list,tw_list):
    """In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations.
       we plot the relation between tau(alpha), where tau are the averaged tau over different initial configurations.
    """
    ex = 0
    shape = tau.shape
    print("shape of ggrand_tau_J:")
    print(shape)
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 6 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(alpha_list[ex:],tau[x+1][l],color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(alpha_list[ex:],tau[x+2][l],color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(alpha_list[ex:],tau[x+3][l],color_list[2],marker='o',label='tw={}'.format(tw_list[2]))
    ax.plot(alpha_list[ex:],tau[x+4][l],color_list[3],marker='v',label='tw={}'.format(tw_list[3]))
    ax.plot(alpha_list[ex:],tau[x+5][l],color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(alpha_list[ex:],tau[x+6][l],color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\tau_J$")
    ax.set_title("l={:d};N={:d}; beta={:3.1f}".format(l,N,beta))
    plt.savefig("../imag/Ave_tau_J_hidden_l{:d}_L{:d}_N{:d}_beta{:2.0f}.eps".format(l,L,N,beta),format='eps')
    plt.savefig("../imag/Ave_tau_J_hidden_l{:d}_L{:d}_N{:d}_beta{:2.0f}.png".format(l,L,N,beta),format='png')

def plot_ggrand_tau_S_tw_X_ave_over_init(tau,l,L,N,beta,alpha_list,tw_list):
    """In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations."""
    ex = 0
    shape = tau.shape
    print("shape of ggrand_tau_S:")
    print(shape)
    #l_list = [1,2,3,4,5,6,7,8]
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 6 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(alpha_list[ex:],tau[x+1][l],color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(alpha_list[ex:],tau[x+2][l],color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(alpha_list[ex:],tau[x+3][l],color_list[2],marker='o',label='tw={}'.format(tw_list[2]))
    ax.plot(alpha_list[ex:],tau[x+4][l],color_list[3],marker='v',label='tw={}'.format(tw_list[3]))
    ax.plot(alpha_list[ex:],tau[x+5][l],color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(alpha_list[ex:],tau[x+6][l],color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\tau_S$ ($10^4$)")
    ax.set_title("l={:d};N={:d}; beta={:3.1f}".format(l,N,beta))
    plt.savefig("../imag/Ave_grand_tau_S_l{:d}_L{:d}_N{:d}_beta{:2.0f}.eps".format(l,L,N,beta),format='eps')
    plt.savefig("../imag/Ave_grand_tau_S_l{:d}_L{:d}_N{:d}_beta{:2.0f}.png".format(l,L,N,beta),format='png')

def plot_ggrand_tau_J_tw_X_l_ave_over_init(tau,l_list,L,N,beta,alpha_list,tw_list,tw_index):
    """1. In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations.
       we plot the relation between tau(alpha), where tau are the averaged tau over different initial configurations.
       2. tau includes all the data.
       3. tw_index = 0,1,2,3,4,5,...
    """
    ex = 0
    shape = tau.shape
    l = l_list
    print("shape of ggrand_tau_J:")
    print(shape)
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(alpha_list[ex:],tau[tw_index][l[0]],color_list[0],marker='.',label="l=2")
    ax.plot(alpha_list[ex:],tau[tw_index][l[1]],color_list[3],marker='*',label="l=5")
    ax.plot(alpha_list[ex:],tau[tw_index][l[2]],color_list[6],marker='o',label="l=8")
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\tau_J$")
    ax.set_title("tw={:d}; l={:d},{:d},{:d}; N={:d}; beta={:3.1f}".format(tw_list[tw_index],l[0]+1,l[1]+1,l[2]+1,N,beta))
    plt.savefig("../imag/Ave_tau_tw{:d}_J_hidden_l{:d}{:d}{:d}_L{:d}_N{:d}_beta{:2.0f}.eps".format(tw_list[tw_index],l[0],l[1],l[2],L,N,beta),format='eps')
    plt.savefig("../imag/Ave_tau_tw{:d}_J_hidden_l{:d}{:d}{:d}_L{:d}_N{:d}_beta{:2.0f}.png".format(tw_list[tw_index],l[0],l[1],l[2],L,N,beta),format='png')

def plot_ggrand_tau_S_tw_X_l_ave_over_init(tau,l_list,L,N,beta,alpha_list,tw_llist,tw_index):
    """1. In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations.
       2. tau includes all the data.
       3. tw_index = 0,1,2,3,4,5,...
    """
    ex = 0
    shape = tau.shape
    l = l_list
    print("shape of ggrand_tau_S:")
    print(shape)
    #l_list = [1,2,3,4,5,6,7,8]
    fig = plt.figure()
    plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(alpha_list[ex:],tau[tw_index][l[0]],color_list[0],marker='.',label="tw=8192;l=2")
    ax.plot(alpha_list[ex:],tau[tw_index][l[1]],color_list[3],marker='*',label="tw=8192;l=5")
    ax.plot(alpha_list[ex:],tau[tw_index][l[2]],color_list[6],marker='o',label="tw=8192;l=8")
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\tau_S$")
    ax.set_title("tw={:d}; l={:d},{:d},{:d}; N={:d}; beta={:3.1f}".format(tw_list[tw_index],l[0]+1,l[1]+1,l[2]+1,N,beta))
    plt.savefig("../imag/Ave_tau_tw{:d}_S_hidden_l{:d}{:d}{:d}_L{:d}_N{:d}_beta{:2.0f}.eps".format(tw_list[tw_index],l[0],l[1],l[2],L,N,beta),format='eps')
    plt.savefig("../imag/Ave_tau_tw{:d}_S_hidden_l{:d}{:d}{:d}_L{:d}_N{:d}_beta{:2.0f}.png".format(tw_list[tw_index],l[0],l[1],l[2],L,N,beta),format='png')

def plot_tau_J_l_ave_over_init(grand_tau,index,L,M,N,beta,tot_steps,tw_list):
    """1. In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations.
       2. When l is fixed."""
    ex = 1
    shape = grand_tau.shape
    ls = tw_list
    fig = plt.figure()
    plt.xscale('log')
    #plt.yscale('log')
    
    plt.ylim((0.1,1000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau.T
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(ls[ex:],grand_tau[x+1][ex:],color_list[0],marker='.',label="l=1")
    ax.plot(ls[ex:],grand_tau[x+2][ex:],color_list[1],marker='*',label="l=2")
    ax.plot(ls[ex:],grand_tau[x+3][ex:],color_list[2],marker='o',label="l=3")
    ax.plot(ls[ex:],grand_tau[x+4][ex:],color_list[3],marker='v',label="l=4")
    ax.plot(ls[ex:],grand_tau[x+5][ex:],color_list[4],marker='^',label='l=5')
    ax.plot(ls[ex:],grand_tau[x+6][ex:],color_list[5],marker='<',label='l=6')
    ax.plot(ls[ex:],grand_tau[x+7][ex:],color_list[6],marker='>',label='l=7')
    ax.plot(ls[ex:],grand_tau[x+8][ex:],color_list[6],marker='1',label='l=8')
    plt.legend(loc="lower left")
    plt.xlabel("$t_w$")
    plt.ylabel(r"$\tau_J$")
    ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/Ave_tau_J_l_hidden_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_tau_J_l_hidden_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.png".format(index,L,M,N,beta,tot_steps),format='png')

def plot_tau_S_l_ave_over_init(grand_tau,index,L,M,N,beta,tot_steps, tw_list):
    """1. In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations.
       2. When l is fixed."""
    ex = 1
    shape = grand_tau.shape
    ls = tw_list
    fig = plt.figure()
    plt.xscale('log')
    #plt.yscale('log')
    
    plt.ylim((0.1,10000.0))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau.T
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(ls[ex:],grand_tau[x+1][ex:],color_list[0],marker='.',label="l=1")
    ax.plot(ls[ex:],grand_tau[x+2][ex:],color_list[1],marker='*',label="l=2")
    ax.plot(ls[ex:],grand_tau[x+3][ex:],color_list[2],marker='o',label="l=3")
    ax.plot(ls[ex:],grand_tau[x+4][ex:],color_list[3],marker='v',label="l=4")
    ax.plot(ls[ex:],grand_tau[x+5][ex:],color_list[4],marker='^',label="l=5")
    ax.plot(ls[ex:],grand_tau[x+6][ex:],color_list[5],marker='<',label="l=6")
    ax.plot(ls[ex:],grand_tau[x+7][ex:],color_list[6],marker='>',label='l=7')
    ax.plot(ls[ex:],grand_tau[x+8][ex:],color_list[7],marker='1',label='l=8')
    plt.legend(loc="lower left")
    plt.xlabel("$t_w$")
    plt.ylabel(r"$\tau_S$")
    ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/Ave_tau_S_l_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_S_l_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.png".format(index,L,M,N,beta,tot_steps),format='png')

def plot_overlap_J(Q,index,init,tw,L,M,N,beta,tot_steps_,tot_steps):
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],Q[x+1][ex:],color_list[0],marker='.',label="l=1")
    ax.plot(time_arr[ex:],Q[x+2][ex:],color_list[1],marker='*',label="l=2")
    ax.plot(time_arr[ex:],Q[x+3][ex:],color_list[2],marker='o',label="l=3")
    ax.plot(time_arr[ex:],Q[x+4][ex:],color_list[3],marker='v',label="l=4")
    ax.plot(time_arr[ex:],Q[x+5][ex:],color_list[4],marker='^',label='l=5')
    ax.plot(time_arr[ex:],Q[x+6][ex:],color_list[5],marker='<',label='l=6')
    ax.plot(time_arr[ex:],Q[x+7][ex:],color_list[6],marker='>',label='l=7')
    ax.plot(time_arr[ex:],Q[x+8][ex:],color_list[7],marker='1',label="l=8")
    ax.plot(time_arr[ex:],Q[x+9][ex:],color_list[8],marker='2',label="l=9")
    ax.plot(time_arr[ex:],Q[x+10][ex:],color_list[9],marker='3',label="l=10")
    #plt.legend(loc="upper right")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,l)$")
    ax.set_title("init={:d};tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(init,tw,L,M,N,beta))
    plt.savefig("../imag/Overlap_J_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,init,tw,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Overlap_J_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(index,init,tw,L,N,beta,tot_steps),format='png')

def plot_overlap_J_hidden(Q,index,init,tw,L,M,N,beta,tot_steps_,tot_steps):
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],Q[x+1][ex:],color_list[0],marker='.',label="l=1")
    ax.plot(time_arr[ex:],Q[x+2][ex:],color_list[1],marker='*',label="l=2")
    ax.plot(time_arr[ex:],Q[x+3][ex:],color_list[2],marker='o',label="l=3")
    ax.plot(time_arr[ex:],Q[x+4][ex:],color_list[3],marker='v',label="l=4")
    ax.plot(time_arr[ex:],Q[x+5][ex:],color_list[4],marker='^',label='l=5')
    ax.plot(time_arr[ex:],Q[x+6][ex:],color_list[5],marker='<',label='l=6')
    ax.plot(time_arr[ex:],Q[x+7][ex:],color_list[6],marker='>',label='l=7')
    ax.plot(time_arr[ex:],Q[x+8][ex:],color_list[7],marker='1',label="l=8")
    #plt.legend(loc="upper right")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q_h(t,l)$")
    ax.set_title("init={:d};tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(init,tw,L,M,N,beta))
    plt.savefig("../imag/Overlap_J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,init,tw,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Overlap_J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(index,init,tw,L,N,beta,tot_steps),format='png')

def plot_ave_overlap_J(Q,index,tw,L,M,N,beta,tot_steps_,tot_steps):
    """The argument overlap_J is the average one over different init's."""
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((0.0,1.1))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],Q[x+1][ex:],color_list[1],marker='.',label="l=1")
    ax.plot(time_arr[ex:],Q[x+2][ex:],color_list[2],marker='*',label="l=2")
    ax.plot(time_arr[ex:],Q[x+3][ex:],color_list[3],marker='o',label="l=3")
    ax.plot(time_arr[ex:],Q[x+4][ex:],color_list[4],marker='v',label="l=4")
    ax.plot(time_arr[ex:],Q[x+5][ex:],color_list[5],marker='^',label='l=5')
    ax.plot(time_arr[ex:],Q[x+6][ex:],color_list[6],marker='<',label='l=6')
    ax.plot(time_arr[ex:],Q[x+7][ex:],color_list[7],marker='>',label='l=7')
    ax.plot(time_arr[ex:],Q[x+8][ex:],color_list[8],marker='1',label="l=8")
    #plt.legend(loc="upper right")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,l)$")
    ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,tw,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(index,tw,L,N,beta,tot_steps),format='png')

def plot_ave_overlap_J_yoshino_setting1(Q,index,tw,L,M,N,beta,tot_steps_,tot_steps):
    """The argument overlap_J is the average one over different init's."""
    ex = 0
    num = M*L*N + L*N*N
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((0.0,1.1))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],Q[x+1][ex:],color_list[x+1],marker='.',label="l=1")
    ax.plot(time_arr[ex:],Q[x+2][ex:],color_list[x+2],marker='*',label="l=2")
    ax.plot(time_arr[ex:],Q[x+3][ex:],color_list[x+3],marker='o',label="l=3")
    ax.plot(time_arr[ex:],Q[x+4][ex:],color_list[x+4],marker='v',label="l=4")
    ax.plot(time_arr[ex:],Q[x+5][ex:],color_list[x+5],marker='^',label='l=5')
    ax.plot(time_arr[ex:],Q[x+6][ex:],color_list[x+6],marker='<',label='l=6')
    ax.plot(time_arr[ex:],Q[x+7][ex:],color_list[x+7],marker='>',label='l=7')
    ax.plot(time_arr[ex:],Q[x+8][ex:],color_list[x+8],marker='1',label="l=8")
    ax.plot(time_arr[ex:],Q[x+9][ex:],color_list[x+9],marker='2',label="l=9")
    ax.plot(time_arr[ex:],Q[x+10][ex:],color_list[x+10],marker='3',label="l=10")
    #plt.legend(loc="upper right")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,l)$")
    ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,tw,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(index,tw,L,N,beta,tot_steps),format='png')

