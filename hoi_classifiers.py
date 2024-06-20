import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)


class ConvolutionFusionModel(FeedForwardNetwork):
    def __init__(self, embedding_size,nr_of_input_channels, nr_of_kernels, hidden_size, output_size):
        super(ConvolutionFusionModel, self).__init__((embedding_size * nr_of_kernels), hidden_size, output_size)
        self.conv1 = nn.Conv1d(nr_of_input_channels,nr_of_kernels,1)
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.layers(x)

        return x


class ElementwiseAvgFusionModel(FeedForwardNetwork):
    def __init__(self, input_size, hidden_size, output_size):
        super(ElementwiseAvgFusionModel, self).__init__(input_size, hidden_size, output_size)
    
    def forward(self, x):
        x = x.mean(dim=1)
        x = self.layers(x)

        return x

class ElementwiseMaxFusionModel(FeedForwardNetwork):
    def __init__(self, input_size, hidden_size, output_size):
        super(ElementwiseMaxFusionModel, self).__init__(input_size, hidden_size, output_size)
    
    def forward(self, x):
        x = x.max(dim=1).values
        x = self.layers(x)

        return x






