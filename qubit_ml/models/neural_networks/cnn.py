import torch.nn as nn
import torch

class ConvNeuralNetwork(nn.Module):
    def __init__(self, qubit_count, time) -> None:
        super().__init__()
        
        # Initialziing Conv layer
        
        self.convFeature = nn.Sequential(
            nn.Conv1d(2, 16, 128),    #1
            nn.ReLU(),                #2
            nn.Conv1d(16, 32, 5),     #3
            nn.ReLU(),                #4
            nn.MaxPool1d(3),          #5
        )
        
               
        #Calculating size for linear layer
        with torch.no_grad():
            x = torch.zeros(1, 2, time) # Time series data with 1 value, with 2 channels
            flat_dim = self.convFeature(x).flatten(1).size(1)
        
        
        self.linearFeature = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flat_dim, flat_dim//2),
            nn.ReLU(),
            nn.Linear(flat_dim//2, qubit_count)
        )
        
        self.apply(self._init_weights)
        
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias) # Precent static type check error
            
    
    def forward(self, x):
        # x: (batch, 2, time)
        x = self.convFeature(x)
        x = torch.flatten(x, 1)   # flatten from channel dimension onward
        x = self.linearFeature(x)
        return x
        