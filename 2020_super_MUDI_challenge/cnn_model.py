import torch.nn as nn
import torch.nn.functional as F


class DiffusionSRCNN(nn.Module):

    def __init__(self, configuration):
        super(DiffusionSRCNN, self).__init__()

        # network configuration
        self.configuration = configuration

        # n1 filters of size 1 x kernel_size1 x kernel_size1
        # input: 1 channel
        # output: n1 channels
        self.conv1 = nn.Conv2d(self.configuration['channels'], self.configuration['n1'],
                               kernel_size=self.configuration['kernel_size1'],
                               padding=self.configuration['padding1'])

        # n2 filters of size n1 x 1 x 1
        # input: n1 channel
        # output: n2 channels
        self.conv2 = nn.Conv2d(self.configuration['n1'], self.configuration['n2'],
                               kernel_size=self.configuration['kernel_size2'],
                               padding=self.configuration['padding2'])

        # one filter of size n2 x kernel_size3 x kernel_size3
        # input: n2 channels
        # output: 1 channel
        self.conv3 = nn.Conv2d(self.configuration['n2'], self.configuration['channels'],
                               kernel_size=self.configuration['kernel_size3'],
                               padding=self.configuration['padding3'])

    def forward(self, x):
        # Conv2D + ReLU
        x = self.conv1(x)
        x = F.relu(x)

        # Conv2D + ReLU
        x = self.conv2(x)
        x = F.relu(x)

        # Conv2D
        x = self.conv3(x)

        return x
