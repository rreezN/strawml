from __init__ import *

import numpy as np
import torch
import argparse
import cv2

import strawml.models.straw_classifier.utils as utils


class FeatureRegressor(torch.nn.Module):
    """ Basic CNN classifier class.
    """
    def __init__(self, image_size=(448, 448), input_size=1024, output_size=1, use_sigmoid=False, num_hidden_layers=0, num_neurons=512) -> None:
        super(FeatureRegressor, self).__init__()

        self.image_size = image_size
        self.output_size = output_size
        self.use_sigmoid = use_sigmoid
        self.num_hidden_layers = num_hidden_layers
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.r = torch.nn.ReLU()
        self.input_layer = torch.nn.Linear(input_size, num_neurons, dtype=torch.float)
        
        # Hidden layers
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(torch.nn.Linear(num_neurons, num_neurons, dtype=torch.float))
        
        # Output layer
        self.out = torch.nn.Linear(num_neurons, self.output_size, dtype=torch.float)
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,1,self.image_size[0],self.image_size[1]]
            
        Returns:
            Output tensor with shape [N,1]
        
        """
        x = torch.flatten(x, 1)
        x = self.r(self.input_layer(x))
        
        if self.num_hidden_layers > 0:
            for i in range(self.num_hidden_layers):
                x = self.r(self.hidden_layers[i](x))
        
        x = self.out(x)
        x = torch.flatten(x)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x
        

    
def get_args() -> argparse.Namespace:
    args = argparse.ArgumentParser(description='Test the CNN classifier model.')
    args.add_argument('--image_size', type=tuple, default=(1370, 204), help='Size of the input image.')
    return args.parse_args()
    

if __name__ == '__main__':
    from torchinfo import summary
    
    args = get_args()
    model = FeatureRegressor(image_size=args.image_size, input_size=1024)
    
    # Test the model with a random input and print sizes of each layer
    print(summary(model, input_size=1024, device='cpu'))