from __init__ import *

import numpy as np
import torch
import argparse
import cv2

import strawml.models.straw_classifier.utils as utils


class FeatureRegressor(torch.nn.Module):
    """ Basic CNN classifier class.
    """
    def __init__(self, image_size=(448, 448), input_size=1024, output_size=1, use_sigmoid=False) -> None:
        super(FeatureRegressor, self).__init__()

        self.image_size = image_size
        self.output_size = output_size
        self.use_sigmoid = use_sigmoid
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.r = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(input_size, 512, dtype=torch.float)
        # self.fc2 = torch.nn.Linear(512, 512, dtype=torch.float)
        # self.fc3 = torch.nn.Linear(512, 512, dtype=torch.float)
        # self.fc4 = torch.nn.Linear(512, 512, dtype=torch.float)
        
        # Output layer
        self.out = torch.nn.Linear(512, self.output_size, dtype=torch.float)
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,1,self.image_size[0],self.image_size[1]]
            
        Returns:
            Output tensor with shape [N,1]
        
        """
        x = torch.flatten(x, 1)
        x = self.r(self.fc1(x))
        # x = self.r(self.fc2(x))
        # x = self.r(self.fc3(x))
        # x = self.r(self.fc4(x))
        
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