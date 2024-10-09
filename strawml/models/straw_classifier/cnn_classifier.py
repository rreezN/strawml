from __init__ import *

import numpy as np
import torch
import argparse
import cv2

import strawml.models.straw_classifier.utils as utils


class CNNClassifier(torch.nn.Module):
    """ Basic CNN classifier class.
    """
    def __init__(self, image_size=(448, 448), img_mean=[0, 0, 0], img_std=[1, 1, 1], input_channels=3, output_size=21) -> None:
        super(CNNClassifier, self).__init__()

        self.image_size = image_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.Tensor(img_mean).to(self.device)
        self.std = torch.Tensor(img_std).to(self.device)
        
        self.conv1 = torch.nn.Conv2d(input_channels, 32, 3, dtype=torch.float)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.r = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 64, 3, dtype=torch.float)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, dtype=torch.float)
        neurons = self.get_linear_layer_neurons()
        self.fc1 = torch.nn.Linear(neurons, 512, dtype=torch.float)
        self.fc2 = torch.nn.Linear(512, output_size, dtype=torch.float)
        self.norm = utils.LayerNorm(32, eps=1e-6, data_format="channels_first")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,1,self.image_size[0],self.image_size[1]]
            
        Returns:
            Output tensor with shape [N,20]
        
        """  
        x = self.pool(self.r(self.conv1(x)))
        x = self.pool(self.r(self.conv2(x)))
        x = self.pool(self.r(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.r(self.fc1(x))
        x = self.fc2(x)
        x = torch.nn.functional.softmax(x, dim=0)
        return x
    
    def get_linear_layer_neurons(self) -> int:
        convs = []
        maxpool = self.pool
        for layer in self.children():
            if isinstance(layer, torch.nn.Conv2d):
                convs.append(layer)
        
        # Get the number of neurons in the linear layer
        image_size = self.image_size
        for conv in convs:
            image_h = (image_size[0] - conv.kernel_size[0] + 2*conv.padding[0]) // conv.stride[0] + 1
            image_w = (image_size[1] - conv.kernel_size[1] + 2*conv.padding[1]) // conv.stride[1] + 1
            image_size = (image_h, image_w)
  
            image_h = (image_size[0] - maxpool.kernel_size) // maxpool.stride + 1
            image_w = (image_size[1] - maxpool.kernel_size) // maxpool.stride + 1
            image_size = (image_h, image_w)
            
        return image_size[0] * image_size[1] * convs[-1].out_channels

    
def get_args() -> argparse.Namespace:
    args = argparse.ArgumentParser(description='Test the CNN classifier model.')
    args.add_argument('--image_size', type=tuple, default=(1370, 204), help='Size of the input image.')
    return args.parse_args()
    

if __name__ == '__main__':
    from torchinfo import summary
    
    args = get_args()
    model = CNNClassifier(image_size=args.image_size)
    
    # Test the model with a random input and print sizes of each layer
    print(summary(model, input_size=(1, 3, args.image_size[0], args.image_size[1]), device='cpu'))
    print(f'Neurons in first linear layer: {model.get_linear_layer_neurons()}')