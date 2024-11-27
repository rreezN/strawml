#!/bin/sh
#BSUB -J straw_classifier
#BSUB -o straw_classifier%J.out
#BSUB -e straw_classifier%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8G]"
#BSUB -W 24:00
#BSUB -N 4
# end of BSUB options

# load a module
# replace VERSION 
module load python3/3.10.13

# load CUDA (for GPU support)
# load the correct CUDA for the pytorch version you have installed
module load cuda/11.8
module load matplotlib/3.8.3-numpy-1.26.4-python-3.10.13

# activate the virtual environment
# NOTE: needs to have been built with the same numpy / SciPy  version as above!
source ~/strawml/.venv/bin/activate

python3 strawml/train_straw_model.py --hpc --data_path train.hdf5 --lr 0.00001 --epochs 100 --cont --model convnextv2 --id initial --image_size 224 224 --data_subsample 1.0
