# Torch Library
import torch
import torch.nn as nn


class LinearResidualAttentionNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.in_features = in_features
        self.out_features = out_features

        self.in_layer = nn.Linear(in_features, hidden_dim)

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_dim))
            self.hidden_layers.append(nn.ReLU())

        self.out_layer = nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # sourcery skip: inline-immediately-returned-variable

        features: torch.FloatTensor
        identity: torch.FloatTensor

        features = self.in_layer(x)

        for i in range(self.num_layers):
            identity = features.clone().detach()

            attention = self.hidden_layers[3 * i](features)
            features = features * attention

            features = self.hidden_layers[3 * i + 1](features)
            features = self.hidden_layers[3 * i + 2](features)
            features = self.hidden_layers[3 * i + 3](features)
            features = features + identity

        logits = self.out_layer(features)
        return logits


if __name__ == "__main__":

    torch.autograd.set_detect_anomaly = True

    data = torch.randn(32, 24)
    net = LinearResidualAttentionNetwork(24, 1, num_layers=2, hidden_dim=32)

    loss = net(data).sum()
    loss.backward()

    print(net)
    print(f"{data.shape=}, {net(data).shape=}")
