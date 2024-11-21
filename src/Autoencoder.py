import torch
import torch.nn as nn



class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        self.encoder = nn.Sequential(
            torch.nn.Linear(2, 9),
            torch.nn.ReLU(),
            # torch.nn.Linear(128, 10),
            # torch.nn.ReLU(),
            # torch.nn.Linear(64, 36),
            # torch.nn.ReLU(),
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            # torch.nn.Linear(18, 9),
            # torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )
         

        self.decoder = nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9),
            torch.nn.ReLU(),
            # torch.nn.Linear(18, 36),
            # torch.nn.ReLU(),
            # torch.nn.Linear(36, 64),
            # torch.nn.ReLU(),
            # torch.nn.Linear(64, 128),
            # torch.nn.ReLU(),
            torch.nn.Linear(9, 2)#,
            #torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x