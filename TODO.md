# TODO

- [ ] Add testing best model on sensor test set to cross_validation at end + log to wandb
- [ ] Investigate if there are any bugs in sensor dataset
  - [ ] Check distributions etc..

## Model Experiments (in order)
### Architecture experiments
These experiments should be done as barebones as possible. No additional fancy features. No data augmentation. No edges, heatmaps. No LR schedulers, bare minimum LR tuning, ADAM optimizer. The goal is to figure out which overall architecture will best satisfy our needs within both accuracy and inference time.

Also figure out what is the model param size we are aiming for?
#### No pre-training 
MODEL - top1 acc - img_size - param_count
- [ ] CNN (own) - ?? - ANY - ??
- [ ] ConvNeXt - 86.2% - 384 - 50.22
- [ ] Eva02 - 88.7% - 448 - 87.12
- [ ] CAFormer - 87.5% - 384 - 56.20
- [ ] ViT - 86.6% - 384 - 60.60
#### Pre-trained
- [ ] ConvNeXt
- [ ] Eva02
- [ ] CAFormer
- [ ] ViT

### Image size experiments
Some models (ConvNeXt for example) are fully convolutional, and thus don't depend on image size. Others can manually specify input sizes for. Might be a good idea to test, especially with non-square images.

- [ ] [ConvNeXt](https://github.com/huggingface/pytorch-image-models/discussions/2269) (should be able to use any size image)
- [ ] [ViT](https://github.com/huggingface/pytorch-image-models/discussions/2104) (can use non-square images, but must be multiple of patch size) (training time does not scale well with larger images - might need to look into compiling torch with flash attention)

Other models should also be possible..

### Minor Architecture Experiments
The goal of these experiments is to test adjustments in architecture after selecting the overall architecture.

#### Sigmoid vs Clamp
- [ ] Nothing
- [ ] Sigmoid
- [ ] Clamp
  
#### TIMM regressor architecture (feature_regressor)
- [ ] Number of layers
- [ ] Number of neurons
- [ ] BatchNorm, etc..
- [ ] Average pool instead of FCN

#### Regressor vs Classifier
- [ ] Regressor
- [ ] Classifier
- [ ] Classifier w. weighted CE

#### Class amount
- [ ] 5%
- [ ] 10%

### Training Experiments
These experiments will alter the training by introducing different optimisers, learning rate schedulers and so on.

#### Optimisers
- [ ] Research SOTA optimisers
- [ ] Adam
- [ ] AdamW
- [ ] SGD

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
- [ ] Image size
- [ ] Smaller batch size sometimes increases regularisation
- [ ] Try a larger model (sometimes can be good even for validation)

#### Dataset Size
- [ ] 25% of training data
- [ ] 50% of training data
- [ ] 75% of training data
- [ ] 100% of training data

#### Manual Features
- [ ] RGB (standard images)
- [ ] Greyscale only
- [ ] Include edges
- [ ] Include heatmaps
- [ ] Only edges
- [ ] Only heatmaps
- [ ] Fourier transform
- [ ] Scale-space transform
- [ ] Wavelet transform

#### Data Augmentation
- [ ] RGBShift
- [ ] HueSaturationValue
- [ ] ChannelShuffle
- [ ] CLAHE
- [ ] RandomContrast
- [ ] RandomGamma
- [ ] RandomBrightness
- [ ] Blur
- [ ] MedianBlur
- [ ] JpegCompression
- [ ] Noise ? - careful can cause some features to go kaboom


### Hyperparameter Tuning
Finally, once the desired augmentations and features have been selected, we will tune the hyperparameters of the model using Bayesian Optimisation, in order to achieve as good results as possible.

#### Parameters and their ranges:
- Learning Rate: 0.01 - 0.0000001
- ...
- ....

### Squeezing out the juice (if time)
- [ ] Ensembles - usually 2% gain
  - [ ] Would probably need to look into some sort of distilling to hit inference time targets https://arxiv.org/abs/1503.02531
- [ ] Leave it training (for very very long - multiple days)


## Model implementations
- [X] Feature extraction from pretrained to use in continuous version
- [X] Setup predict_model to work with pretrained regressions
- [X] Cross-validation
- [ ] More features ?


## BIG ONE
- [ ] Go through http://karpathy.github.io/2019/04/25/recipe/
