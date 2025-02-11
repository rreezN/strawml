#!/bin/sh
#BSUB -J continue_training
#BSUB -o continue_training%J.out
#BSUB -e continue_training%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16G]"
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

python3 strawml/train_straw_model.py --model convnext --load_model models/convnext --seed 0 --batch_size 4 --lr 0.00001120659857537586 --image_size 672 208 --id best_convnext_:apriltag_seed0 --data_subsample 1.0 --optim adam --augment_probability 0.0 --cont --use_wce --hpc --epochs 150 --pretrained --data_path train.hdf5 --cutout_type apriltag