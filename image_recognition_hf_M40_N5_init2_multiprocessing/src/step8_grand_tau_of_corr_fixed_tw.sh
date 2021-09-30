#!/bin/bash 
#SBATCH -p amd_256 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -J OL_clean

M=40
N=5
C=2
OUTPUT_LOG='std_tau_corr.log'
# the first sleep time should be larger, here set it as 20 s.
#srun -n 1 python /public1/home/sc91981/yoshino_setting1/src/overlap.py & 
#sleep 20
python3 /public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}_multiprocessing/src/grand_tau_of_corr_fixed_tw.py -C $C & 
#sleep 20
wait

# To submit a job, use `sbatch job.sh` or `u job.sh`
