M=$1
N=$2
C=$3
mkdir -p /home/gang/Github/Yoshino8/image_recognition_hf_M${M}_N${N}_init${C}_multiprocessing_reverse/src
papp_cloud rsync sc91981@bscc-a3:/public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}_multiprocessing_reverse/src/step* /home/gang/Github/Yoshino8/image_recognition_hf_M${M}_N${N}_init${C}_multiprocessing_reverse/src 
papp_cloud rsync sc91981@bscc-a3:/public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init${C}_multiprocessing_reverse/src/*py /home/gang/Github/Yoshino8/image_recognition_hf_M${M}_N${N}_init${C}_multiprocessing_reverse/src 
