import torch
import torch.nn as nn

class EnsembleRewardMLP(nn.Module):
    def __init__(self, ensemble_size, hidden_layers, act, output_act):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.models = nn.ModuleList()

        for _ in range(ensemble_size):
            layers = []
            input_dim = 256  # adjust based on input shape
            for h in hidden_layers:
                layers.append(nn.Linear(input_dim, h))
                layers.append(getattr(nn.functional, act))
                input_dim = h
            layers.append(nn.Linear(input_dim, 1))
            self.models.append(nn.Sequential(*layers))

    def forward(self, x):
        rewards = torch.stack([model(x) for model in self.models], dim=0)
        return rewards.mean(dim=0)  # or however the ensemble is averaged