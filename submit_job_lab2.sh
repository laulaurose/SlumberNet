#!/bin/sh 
#BSUB -q gpua40
#BSUB -J lab_2_fix   
#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -W 24:00
##BSUB -u laurarose@sund.ku.dk 
#BSUB -N 
#BSUB -o jobfiles/Output_%J.out 
#BSUB -e jobfiles/Output_%J.err 
source /zhome/dd/4/109414/miniconda3/etc/profile.d/conda.sh
conda activate spindle_javier
module load gcc/12.3.0-binutils-2.40
nvidia-smi
# Load the cuda module
module load cuda/11.6
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery
python /zhome/dd/4/109414/Validationstudy/slumbernet/train_slumbernet_v2.py --basedir /zhome/dd/4/109414/Validationstudy/slumbernet/config/slumbernet_labs.yml --test_lab "Antoine" --out_dir  /work3/laurose/Overall_model_evaluation/SlumberNet/models/fixed_n/run3/lab2/

