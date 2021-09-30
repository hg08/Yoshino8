#!/bin/bash
#SBATCH -p amd_256 
#SBATCH -N 1
#SBATCH -n  1
#SBATCH -c  64
#SBATCH -J clean_tw0 
module load /public1/home/sc91981/anaconda3

B=$1 # invers temperature
C=$2 # init index. Differernt inex will use different pictures in the database.
D=$3 #Waiting time
I=$4 # number of bits for input
J=$5 # number of bits for output
L=$6 # number of layers
M=$7 # number of samples to be stored
N=$8 # num of node at each layer
S=$9 # total steps
#U=${10} # waiting time index. Only used for guest.py. 

waiting0=105
waiting=10

OUTPUT_LOG=$D'_tw.log'
date >> $OUTPUT_LOG
#guest.py need a parater J as index in each running, to find the location of the initial configurations of S and J, etc.
# To obtain a reasonable short-time relaxation of Overlaps (Q(l,t) and q(l,t)), the results from this two function will be averaged.
declare -i #j=0 # define an integer
srun -n 1 --cpus-per-task=1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/init_clean.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C >> $OUTPUT_LOG & 
sleep $waiting0
srun -n 1 --cpus-per-task=1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/host.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C >> $OUTPUT_LOG &
sleep $waiting0 
srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C  &
sleep $waiting
srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C  &
sleep $waiting
srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C  &
sleep $waiting
srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting 
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C & 
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C & 
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting 
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C & 
#sleep $waiting 
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C  &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C  &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C & 
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C & 
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C  &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C  &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C & 
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C & 
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
#srun -n 1 python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}/src/guest.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -D $D -C $C &
#sleep $waiting
wait
