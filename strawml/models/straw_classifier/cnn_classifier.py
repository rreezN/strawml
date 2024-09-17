import torch

class CNNClassifier(torch.nn.Module):
    """ Basic CNN classifier class.
    """
    def __init__(self, image_size=[1440, 2440]) -> None:
        super(CNNClassifier, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, 3, dtype=torch.float)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.r = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 64, 3, dtype=torch.float)
        self.fc1 = torch.nn.Linear(186624, 128, dtype=torch.float)
        self.fc2 = torch.nn.Linear(128, 20, dtype=torch.float)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,1,1440,2440]
            
        Returns:
            Output tensor with shape [N,20]
        
        """
        
        x = self.pool(self.r(self.conv1(x)))
        x = self.pool(self.r(self.conv2(x)))
        x = x.flatten()
        x = self.r(self.fc1(x))
        x = self.fc2(x)
        x = torch.nn.functional.softmax(x, dim=0)
        return x