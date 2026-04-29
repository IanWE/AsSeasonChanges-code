import torch
import torch.nn as nn
import torch.nn.functional as F

class MOENet(nn.Module):
    def __init__(self, input_dim, hidden, num_experts, output_dim = 2):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden
        self.centers = nn.Parameter(torch.linspace(0, 1, num_experts))  #Do not use a fixed start point
        init = torch.full((num_experts,), 1.0)
        self.log_sigma = nn.Parameter(torch.log(init))
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim-1, hidden),#removing time feature from expert is better
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                torch.nn.Dropout(0.5),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                torch.nn.Dropout(0.5),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                torch.nn.Dropout(0.5),
                nn.Linear(hidden, output_dim),
            ) for _ in range(num_experts)
        ])

    def forward(self, x, return_gate=False):
        timestamp = x[:, -1].unsqueeze(1) # (B,1)
        centers = self.centers.unsqueeze(0)  # (1,N)
        B = x.size(0)       
        #    w_ij = exp(- (t_j - c_i)^2 / (2σ^2) ) 
        sigma = torch.exp(self.log_sigma).unsqueeze(0) #(1,N)
        numerator = timestamp - centers
        logits = torch.exp(- numerator.pow(2) / (2 * sigma.pow(2)))
        if logits.max() == 0.0:
            raise ValueError("The time is out of the scope of MoE")

        expert_outputs = torch.stack([expert(x[:,:-1]) for expert in self.experts], dim=1)
        output = torch.einsum('be,beo->bo', logits, expert_outputs)
        return output
