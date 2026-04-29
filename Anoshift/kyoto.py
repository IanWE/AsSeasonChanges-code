"""
Replace AnoShift/baselines_OOD_setup/baseline_deep_svdd/networks/kyoto.py with this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


IN_SIZE = 571
REP_DIM = 50

class Kyoto_Net(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = REP_DIM
        
        self.encoder = nn.Sequential(
            torch.nn.Linear(IN_SIZE, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
#             torch.nn.ReLU(),
#             torch.nn.Linear(364, 400),
#             torch.nn.ReLU(),
#             torch.nn.Linear(400, 500)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class Kyoto_Net_Autoencoder(BaseNet):
    def __init__(self):
        super().__init__()
        self.rep_dim = REP_DIM
        self.encoder = nn.Sequential(
            torch.nn.Linear(IN_SIZE, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
        )
          
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = nn.Sequential(
            torch.nn.Linear(50, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, IN_SIZE),
            torch.nn.Sigmoid()
        )


    def forward(self, x):
        print("x", x.shape)
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)

        return x

class Kyoto_MOE(BaseNet):
    def __init__(self):
        super().__init__()
        self.num_experts = 16
        self.centers = nn.Parameter(torch.linspace(0, 1, self.num_experts))  #use a fixed start point, not good
        #self.centers = nn.Parameter(torch.randn(self.num_experts))
        init = torch.full((self.num_experts,), 1.0)#1/2*num_experts)
        self.log_sigma = nn.Parameter(torch.log(init))
        self.experts = nn.ModuleList([
            nn.Sequential(
                torch.nn.Linear(570, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 50),
            ) for _ in range(self.num_experts)
        ])
        self.rep_dim = REP_DIM

    def forward(self, x):
        timestamp = x[:, -1].unsqueeze(1) # (B,1)
        centers = self.centers.unsqueeze(0)#torch.sigmoid(self.centers.unsqueeze(0))  # (1,N)
        B = x.size(0)       
        #    w_ij = exp(- (t_j - c_i)^2 / (2σ^2) ) 
        sigma = torch.exp(self.log_sigma).unsqueeze(0) #(1,N)
        numerator = timestamp - centers
        route = torch.exp(- numerator.pow(2) / (2 * sigma.pow(2))) #/(sigma*torch.sqrt(2*torch.pi))  # (B, N)
        expert_outputs = torch.stack([expert(x[:,:-1]) for expert in self.experts], dim=1)
        output = torch.einsum('be,beo->bo', route, expert_outputs)
        return output

