#!/bin/sh 
#BSUB -q gpua100
#BSUB -J slumbernet_training_1 
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -W 48:00
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

#python /zhome/dd/4/109414/Validationstudy/spindle/testing_CNN_spindle.py --experiment standard_config_multiple_labs --test_lab "" --save_dir /zhome/dd/4/109414/Validationstudy/spindle/results/spindle_test/ --model_dir /zhome/dd/4/109414/Validationstudy/spindle/results/spindle_test/spindle_ssf10.9741511.h5
python /zhome/dd/4/109414/Validationstudy/slumbernet/train_slumbernet_v2.py --basedir /zhome/dd/4/109414/Validationstudy/slumbernet/config/slumbernet_config.yml --test_lab "" --out_dir  /work3/laurose/SlumberNet/models/orig/run1/
#python /zhome/dd/4/109414/Validationstudy/spindle/training_CNN_fixed_n.py --experiment standard_config_multiple_labs --test_lab  "Sebastian" --save_dir /work3/laurose/spindle/models/all_n
