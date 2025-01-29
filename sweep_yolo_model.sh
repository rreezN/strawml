#!/bin/sh
#BSUB -J yolo_sweeper_chute
#BSUB -o yolo_sweeper_chute%J.out
#BSUB -e yolo_sweeper_chute%J.err
#BSUB -q gpua40
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
source venv/bin/activate

# python strawml/cross_validate_yolo.py --path /work3/s194247/yolo_format_bbox_straw_whole_5fold --id 4ccyh2d4
# python strawml/cross_validate_yolo.py --path /work3/s194247/yolo_format_bbox_straw_5fold --id 872t9e6k
python strawml/cross_validate_yolo.py --path /work3/s194247/yolo_format_bbox_chute_5fold --id xkh5pghf