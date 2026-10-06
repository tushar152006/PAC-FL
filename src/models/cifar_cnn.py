import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Lightweight Convolutional Neural Network designed for CIFAR-10 edge FL simulation.
    Fast to train across 20-50 clients on standard CPUs.
    """
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 32x32 -> Pool -> 16x16 -> Pool -> 8x8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1: (3, 32, 32) -> (32, 16, 16)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2: (32, 16, 16) -> (64, 8, 8)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Block 3: (64, 8, 8) -> (64, 8, 8)
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_model(num_classes: int = 10) -> nn.Module:
    """Helper factory function to instantiate model."""
    return SimpleCNN(num_classes=num_classes)
