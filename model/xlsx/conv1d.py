import torch
import torch.nn as nn


class Conv1DRegression(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(Conv1DRegression, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1)

        # 激活层
        self.relu = nn.ReLU()

        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=4, stride=3)

        # 全连接层
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, output_size)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # [batch_size, 1, input_size]
        x = self.conv1(x)
        # [batch_size, 16, input_size]
        x = self.relu(x)
        # [batch_size, 16, input_size // 2]
        x = self.pool(x)

        x = self.conv2(x)
        # [batch_size, 32, input_size // 2]
        x = self.relu(x)
        # (batch_size, 32, 1]
        x = self.pool(x)

        # [batch_size, 32]
        x = x.flatten(start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    net = Conv1DRegression(24, 1)

    x = torch.randn(300, 1, 24)
    print(f"{net(x).shape=}")
