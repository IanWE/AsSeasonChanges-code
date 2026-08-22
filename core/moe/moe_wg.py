import torch
import torch.nn as nn
import torch.nn.functional as F

class MOENet_wg(nn.Module):
    def __init__(self, input_dim, hidden, num_experts, output_dim = 2):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden
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
        # Gate
        self.gate = nn.Sequential(
            nn.Linear(1, hidden), 
            nn.ReLU(),
            nn.Linear(hidden, num_experts), 
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            torch.nn.Dropout(0.5),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        timestamp = x[:, -1].unsqueeze(1) # (B,1)
        route = self.gate(timestamp)
        route = F.softmax(route, dim=1)
        expert_outputs = torch.stack([expert(x[:,:-1]) for expert in self.experts], dim=1)
        output = torch.einsum('be,beo->bo', route, expert_outputs)
        return output
