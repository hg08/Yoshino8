#M=$1
#N=$2
#C=$3
mkdir -p /home/gang/Github/Yoshino_6a/imag
#papp_cloud rsync sc91981@bscc-a3:/public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init$C/imag/*png /home/gang/Github/Yoshino8/image_recognition_hf_M${M}_N${N}_init$C/imag 
#papp_cloud rsync sc91981@bscc-a3:/public1/home/sc91981/image_recognition_hf_M${M}_N${N}_init$C/imag/*eps /home/gang/Github/Yoshino8/image_recognition_hf_M${M}_N${N}_init$C/imag 
papp_cloud rsync blsc784@gz:/PARA2/blsc784/Yoshino_6a/imag/*png /home/gang/Github/Yoshino_6a/imag 
papp_cloud rsync blsc784@gz:/PARA2/blsc784/Yoshino_6a/imag/*eps /home/gang/Github/Yoshino_6a/imag 
