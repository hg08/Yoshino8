#!/bin/bash 
#SBATCH -p amd_256 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64 
#SBATCH -J DigitOverlap

OUTPUT_LOG='std_overlap_ave_over_init.log'
# This script calcuate the average overlap for S and J, over different initial configurations (i.e., different init's). 
R=8
M=40
N=5

#declare -a tw=(0 8192 65536 524288 4194304 33554432)
B=66.7
#R: the number of replicas
python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init2_multiprocessing/src/main_overlap_log_tw_ave_over_init.py -L $L -M $M -N $N -B $B -I $I -J $J -R $R 
#wait

# To submit a job, use `sbatch job.sh` or `u job.sh`
