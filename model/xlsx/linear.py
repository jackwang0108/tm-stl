# Torch Library
import torch
import torch.nn as nn


class LinearNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.in_features = in_features
        self.out_features = out_features

        self.in_layer = nn.Linear(in_features, hidden_dim)

        self.hidden_layers = nn.Sequential()
        for i in range(num_layers):
            self.hidden_layers.add_module(
                f"linear{i}", nn.Linear(hidden_dim, hidden_dim)
            )
            self.hidden_layers.add_module(f"relu{i}", nn.ReLU())

        self.out_layer = nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # sourcery skip: inline-immediately-returned-variable
        features = self.in_layer(x)
        features = self.hidden_layers(features)
        logits = self.out_layer(features)
        return logits


if __name__ == "__main__":

    data = torch.randn(32, 16)
    net = LinearNetwork(16, 1, num_layers=2, hidden_dim=32)

    print(net)
    print(f"{data.shape=}, {net(data).shape=}")