# TODO

## Optimal Model Settings
### ConvNeXtv2
- Best mean sensor accuracy: Large images, no sig, ADAM, 1 hidden layer, 512 neurons, no manual features 
- Overall best sensor accuracy: Large images, sig, ADAM, 1 hidden layer, 512 neurons, no manual features
  
### Vision Transformer
- Best mean sensor accuracy: Large images, sig, ADAMW, 1 hidden layer, 512 neurons, no manual features
- Overall best sensor accuracy: Large images, sig, ADAMW, 1 hidden layer, 512 neurons, no manual features

## Model Experiments (in order)
### Architecture experiments
These experiments should be done as barebones as possible. No additional fancy features. No data augmentation. No edges, heatmaps. No LR schedulers, bare minimum LR tuning, ADAM optimizer. The goal is to figure out which overall architecture will best satisfy our needs within both accuracy and inference time.

Also figure out what is the model param size we are aiming for?

MODEL - top1 acc - img_size - param_count
- [X] ConvNeXt - 86.2% - 224 - 50.22
- [X] ConvNeXt - 384
- [X] ViT - 86.6% - 384 - 60.60


#### Regressor vs Classifier
- [X] Regressor
- [X] Classifier
- [X] Classifier w. weighted CE

#### Only Train Head
- [X] Head only

### Image size experiments
Some models (ConvNeXt for example) are fully convolutional, and thus don't depend on image size. Others can manually specify input sizes for. Might be a good idea to test, especially with non-square images.

- [X] [ConvNeXt](https://github.com/huggingface/pytorch-image-models/discussions/2269) (should be able to use any size image)
- [X] [ViT](https://github.com/huggingface/pytorch-image-models/discussions/2104) (can use non-square images, but must be multiple of patch size) (training time does not scale well with larger images - might need to look into compiling torch with flash attention)

Other models should also be possible..

### Minor Architecture Experiments
The goal of these experiments is to test adjustments in architecture after selecting the overall architecture.

#### Sigmoid vs Clamp
- [X] Nothing
- [X] Sigmoid

#### TIMM regressor architecture (feature_regressor)
- [X] Number of layers
- [X] Number of neurons
- [ ] Try a CNN instead of a FCN as the feature regressor
- [ ] BatchNorm, etc..
- [ ] Average pool instead of FCN

#### Manual Features
- [X] RGB (standard images)
- [X] Greyscale only
- [X] Include edges
- [X] Include heatmaps
- [ ] Only edges
- [ ] Only heatmaps
- [ ] Fourier transform
- [ ] Scale-space transform
- [ ] Wavelet transform

### Training Experiments
These experiments will alter the training by introducing different optimisers, learning rate schedulers and so on.

#### Optimisers
- [ ] Research SOTA optimisers
- [X] Adam
- [X] AdamW
- [X] SGD

THESE ARE NEW AND SOTA (and it might be possible to combine them, but that would probably constitute a research paper of its own):
- [ ] [SOAP](https://arxiv.org/abs/2409.11321)
- [ ] [ADOPT](https://x.com/ishohei220/status/1854051859385978979)

#### Learning Rate Schedulers
- [ ] Research SOTA LR schedulers
- [ ] Step Decay
- [ ] Exponential Decay
- [ ] Cosine Annealing
- [ ] Cyclic
- [ ] MultiStep
- [ ] Plateau
- [ ] Linear
- [ ] OneCycle
- [ ] CosineAnnealingWarmRestarts

### Regularisation Experiments
After having selected a model and training procedure, we will try to experiment with data augmentation and manual features. Each step will have to be tested isolated from each other, to see its impact on the model.

#### Model Changes
- [ ] Dropout
- [ ] AveragePool instead of FCN at the end (feature_regressor)
- [X] Image size
- [ ] Smaller batch size sometimes increases regularisation
- [ ] Try a larger model (sometimes can be good even for validation)


#### Data Augmentation
- [X] 25%
- [X] 50%
- [X] 75%
- [X] 100% 


### Hyperparameter Tuning
Finally, once the desired augmentations and features have been selected, we will tune the hyperparameters of the model using Bayesian Optimisation, in order to achieve as good results as possible.

#### Parameters and their ranges:
- Learning Rate: 0.01 - 0.0000001
- Batch size: 4 - 128
- ....

### Squeezing out the juice (if time)
- [ ] Ensembles - usually 2% gain
  - [ ] Would probably need to look into some sort of distilling to hit inference time targets https://arxiv.org/abs/1503.02531
- [ ] Leave it training (for very very long - multiple days)

#### Dataset Size
- [ ] 25% of training data
- [ ] 50% of training data
- [ ] 75% of training data
- [ ] 100% of training data
