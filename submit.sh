#!/bin/sh
#BSUB -J yolov8_digit
#BSUB -o yolov8_digit%J.out
#BSUB -e yolov8_digit%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=8G]"
#BSUB -W 10:00
#BSUB -N 4
# end of BSUB options

# load a module
# replace VERSION 
module load python3/3.10.12

# load CUDA (for GPU support)
# load the correct CUDA for the pytorch version you have installed
module load cuda/11.8
module load matplotlib/3.8.3-numpy-1.26.4-python-3.10.13

# activate the virtual environment
# NOTE: needs to have been built with the same numpy / SciPy  version as above!
source ~/Documents/strawml/venv/bin/activate

python3 strawml/train_digit_detect.py
