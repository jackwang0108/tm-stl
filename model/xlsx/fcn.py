import torch
import torch.nn as nn


class FullyConvolutionNetwork(nn.Module):
    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super(FullyConvolutionNetwork, self).__init__()

        # 卷积层
        self.in_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.hidden_layer = nn.Sequential()
        for i in range(num_layers):
            self.hidden_layer.add_module(
                f"conv {i}",
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    stride=1,
                ),
            )
            self.hidden_layer.add_module(f"bn {i}", nn.BatchNorm1d(hidden_dim))
            self.hidden_layer.add_module(f"relu {i}", nn.ReLU())

        self.out_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=out_feature,
                kernel_size=1,
                stride=1,
            )
        )

        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=4, stride=3)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = self.in_layer(x)
        x = self.hidden_layer(x)
        x = self.out_layer(x)
        return x


if __name__ == "__main__":
    net = FullyConvolutionNetwork(24, 1, 8, 3)

    x = torch.randn(300, 24)
    print(f"{net(x).shape=}")
