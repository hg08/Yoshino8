#!/bin/bash 
#SBATCH -p amd_256 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -J DigitOverlap

M=40
N=5
C=2

OUTPUT_LOG='std_overlap_more_ave_over_init.log'
srun -n 1 python /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}_multiprocessing/src/main_overlap_more_ave_over_init.py & 
#sleep 20
wait

# To submit a job, use `sbatch job.sh` or `u job.sh`
