import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        channel = 6 if normal_channel else 3

        self.feat = PointNetEncoder(
            global_feat=True, feature_transform=True, channel=channel
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3_1 = nn.Linear(256, k)
        self.fc3_2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x_cls = self.fc3_1(x)
        x_reg = self.fc3_2(x)
        x_cls = F.log_softmax(x_cls, dim=1)
        return x_cls, x_reg, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(
        self,
        pred: torch.FloatTensor,
        target: torch.LongTensor,
        trans_feat: torch.FloatTensor,
    ):
        # sourcery skip: inline-immediately-returned-variable
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
