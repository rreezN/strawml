# strawml (Machine Learning for Continuous Straw Level Measurement)

[<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">]()
[<img src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white">]()
[<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue">]()

This repository contains code developed for the Master's project _Machine Learning for Continuous Straw Level Measurement_ for the Fall 2024 semester. The project is done by [David Ari Ostenfeldt (194237)](https://www.linkedin.com/in/david-ostenfeldt/) and [Dennis Chenxi Zhuang (194247)](https://www.linkedin.com/in/dennis-chenxi-zhuang/) at [The Technical University of Denmark](https://www.dtu.dk/) in collaboration with [Meliora Bio](https://meliora-bio.com/) and [Helix Lab](https://helixlab.dk/).
# Quick Guide


# Code Explanation
`make_dataset.py`   
The file contains code that takes a folder of videos to split them into sequential frames and then merges the videos into one data file, .hdf5. Each frame receives a unique `id`; since merging them removes the ability to distinguish between the different recordings, we have added an attribute to each frame that states which video it originally belonged to. Additionally, because of the size of the videos and when converged into numpy arrays in Python the size increases, we encode each frame to a python `.jpg` binary. This allows for quicker storage and smaller data size, and decoding takes little to no time.


# To-Do Data-Augmentation
- [ ] Rotate
- [ ] Zoom and crop
- [ ] translate cute
- [ ] Contrast
- [ ] Gamma
- [ ] Brightness
- [ ] Noise
- [ ] Add artifacts that resemble obstructed view of chute
