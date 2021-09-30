#!/bin/bash 
#SBATCH -p amd_256 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -J DigitOverlap

M=40
N=5
C=2
S=2048

OUTPUT_LOG='std_tau_corr.log'
# the first sleep time should be larger, here set it as 20 s.
#srun -n 1 python /public1/home/sc91981/image_recognition/src/overlap.py & 
python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}_multiprocessing/src/main_tau_of_corr.py -M $M -S $S & 
wait

# To submit a job, use `sbatch job.sh` or `u job.sh`
