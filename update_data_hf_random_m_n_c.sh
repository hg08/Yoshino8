M=$1
N=$2
C=$3
mkdir -p /home/gang/Github/Yoshino8/image_recognition_hf_random_M${M}_N${N}_init$C/data1
papp_cloud rsync sc91981@bscc-a3:/public1/home/sc91981/image_recognition_hf_random_M${M}_N${N}_init$C/data1/*/grand_tau*npy /home/gang/Github/Yoshino8/image_recognition_hf_random_M${M}_N${N}_init$C/data1 
papp_cloud rsync sc91981@bscc-a3:/public1/home/sc91981/image_recognition_hf_random_M${M}_N${N}_init$C/data1/*/overlap_*npy /home/gang/Github/Yoshino8/image_recognition_hf_random_M${M}_N${N}_init$C/data1
papp_cloud rsync sc91981@bscc-a3:/public1/home/sc91981/image_recognition_hf_random_M${M}_N${N}_init$C/data1/*/para_*npy /home/gang/Github/Yoshino8/image_recognition_hf_random_M${M}_N${N}_init$C/data1
