#!/bin/sh 
#BSUB -q hpc
#BSUB -J test_lab1 
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -W 24:00
#BSUB -u laurarose@sund.ku.dk 
#BSUB -N 
#BSUB -o jobfiles/Output_%J.out 
#BSUB -e jobfiles/Output_%J.err 

source /zhome/dd/4/109414/miniconda3/etc/profile.d/conda.sh
conda activate spindle_javier

#module load cuda/11.6
#/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

python /zhome/dd/4/109414/Validationstudy/slumbernet/test_slumbernet.py --test_lab "mus" --model_dir "/work3/laurose/SlumberNet/models/orig/run2/epoch45.h5" --color [187/255,212/255,233/255]

