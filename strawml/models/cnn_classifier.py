import torch

class CNNClassifier(torch.nn.Module):
    """ Basic CNN classifier class.
    """
    def __init__(self, image_size=[1440, 2440]) -> None:
        super(CNNClassifier, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, 3, dtype=torch.double)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.r = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 64, 3, dtype=torch.double)
        self.fc1 = torch.nn.Linear(2880, 128, dtype=torch.double)
        self.fc2 = torch.nn.Linear(128, 20, dtype=torch.double)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,1,1440,2440]
            
        Returns:
            Output tensor with shape [N,20]
        
        """
        
        x = self.pool(self.r(self.conv1(x)))
        x = self.pool(self.r(self.conv2(x)))
        # Temporary solution to reduce the size of the tensor
        x = self.pool(self.pool(self.pool(self.pool(self.pool(self.pool(x))))))
        x = x.flatten()
        x = self.r(self.fc1(x))
        x = self.fc2(x)
        x = torch.nn.functional.softmax(x, dim=0)
        return x