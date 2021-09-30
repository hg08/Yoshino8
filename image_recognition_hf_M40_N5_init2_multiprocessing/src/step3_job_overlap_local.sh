#!/bin/bash 

#Metadata
OUTPUT_LOG='std.log'

#Fixed parameters
B=66.7
I=784
J=10
L=10
N=5
R=8 # Fixed parameter 
S=256
#R: the number of replicas

#Changable parameters
C=2
M=40

# the first sleep time should be larger, here set it as w second.
python3 main_overlap_log_tw.py -L $L -M $M -N $N -S ${S[0]} -B $B -I $I -J $J -C $C -R $R 
wait
