import torch.nn as nn

class DeepNeuralNetwork(nn.Module):
    def __init__(self, qubit_count, time) -> None:
        super().__init__()
        
        self.linearFeature = nn.Sequential(
            nn.Linear(time, time//2),
            nn.SELU(),
            nn.Linear(time//2, time//4),
            nn.SELU(),
            nn.Linear(time//4, 2**qubit_count)
        )
           
    def forward(self, x):
        x = self.linearFeature(x)
        return x
        