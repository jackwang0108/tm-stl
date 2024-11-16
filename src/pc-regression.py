# Standard Library
import copy
import datetime
from pathlib import Path

# Third-Party Library
import numpy as np

# Torch Library
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# My Library
import utils.provider as provider
from utils import (
    get_device,
    get_logger,
    set_random_seed,
)
from utils.common import (
    make_tupleList,
    make_hierarchyDict,
)
from utils.pc_helper import (
    collate_fn,
    calculate_acc,
    random_split_data,
    get_pointcloud_files,
    PointCloudDataset,
)
from model.pc.pointnet.pointnet_cls import (
    get_loss,
    get_model,
)

set_random_seed(42)

device = get_device()
base_dir = Path(__file__).resolve().parent.parent
pc_path = base_dir / "data/new-pc"
xlsx_path = pc_path / "分类-强度数据集.xlsx"
log_dir = base_dir / "log"
logger_path = log_dir / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
logger = get_logger(logger_path)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def augment_pointcloud(pc: torch.FloatTensor) -> torch.FloatTensor:

    pc = pc.data.numpy()
    pc = provider.random_point_dropout(pc)
    pc[:, :, :3] = provider.random_scale_point_cloud(pc[:, :, :3])
    pc[:, :, :3] = provider.shift_point_cloud(pc[:, :, :3])
    augmented_pc = torch.Tensor(pc)
    return augmented_pc.transpose(2, 1)


def train_epoch(
    model: get_model,
    train_loader: DataLoader,
    loss_func: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> torch.FloatTensor:

    model.train(True)

    mse_loss = nn.MSELoss()

    accs, mses, losses = [], [], []

    mse: torch.FloatTensor
    loss: torch.FloatTensor
    points: torch.FloatTensor
    cls_label: torch.LongTensor
    reg_label: torch.LongTensor
    reg_preds: torch.FloatTensor
    cls_preds: torch.FloatTensor
    for i, (points, cls_label, reg_label) in enumerate(train_loader):
        optimizer.zero_grad()

        cls_label, reg_label, augmented_points = (
            cls_label.to(device=device, dtype=torch.int64),
            reg_label.to(device=device, dtype=torch.float32),
            augment_pointcloud(points).to(device),
        )

        cls_preds, reg_preds, trans_feature = model(augmented_points)

        loss = loss_func(cls_preds, cls_label, trans_feature) + (
            mse := mse_loss(reg_preds.squeeze(), reg_label)
        )

        loss.backward()
        optimizer.step()

        pred_choice = cls_preds.data.max(1)[1]

        accs.append(calculate_acc(cls_label, pred_choice))
        mses.append(mse.item())
        losses.append(loss.item())

        digits = len(f"{len(train_loader)}")
        logger.info(f"\t\t batch [{i+1:{digits}d}/{len(train_loader)}], {loss=:.4f}")

    return sum(losses) / len(losses), sum(accs) / len(accs), sum(mses) / len(mses)


@torch.no_grad()
def test_epoch(model: get_model, loader: DataLoader, num_class: int = 4):

    model.eval()

    mean_correct = []
    class_acc = np.zeros((num_class, 3))

    meses = []
    mse_loss = nn.MSELoss()

    cls_label: torch.FloatTensor
    reg_label: torch.FloatTensor
    points: torch.FloatTensor

    # sourcery skip: no-loop-in-tests
    for points, cls_label, reg_label in loader:
        cls_label, reg_label, points = (
            cls_label.to(device=device, dtype=torch.int64),
            reg_label.to(device=device, dtype=torch.float32),
            points.to(dtype=torch.float32, device=device),
        )

        points = points.transpose(2, 1)
        cls_pred, reg_pred, _ = model(points)
        pred_choice = cls_pred.data.max(1)[1]

        meses.append(mse_loss(reg_label, reg_pred.squeeze()))

        for cat in np.unique(cls_label.cpu()):
            classacc = (
                pred_choice[cls_label == cat]
                .eq(cls_label[cls_label == cat].long().data)
                .cpu()
                .sum()
            )
            class_acc[cat, 0] += classacc.item() / float(
                points[cls_label == cat].size()[0]
            )
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(cls_label.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc, sum(meses) / len(meses)


def train(
    model: get_model,
    loss_func: nn.Module,
    train_data: dict[str, list[Path]],
    val_data: dict[str, list[Path]],
    num_epoch: int,
):

    valset, trainset = (
        PointCloudDataset(val_data, 1024, "random", xlsx_path),
        PointCloudDataset(train_data, 1024, "random", xlsx_path),
    )

    val_loader = DataLoader(
        valset,
        batch_size=12,
        shuffle=False,
        num_workers=2,
        drop_last=True,
        collate_fn=collate_fn,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=12,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    best_instance_acc = 0.0
    digits = len(f"{num_epoch}")
    for epoch in range(num_epoch):
        model = model.train()

        train_loss, train_acc, train_mse = train_epoch(
            model, train_loader, loss_func, optimizer
        )

        instance_acc, class_acc, val_mse = test_epoch(model, val_loader, num_class=4)

        scheduler.step()

        if best_instance_acc < instance_acc:
            best_instance_acc = instance_acc
            best_model = copy.deepcopy(model)

        logger.info(
            f"Epoch: [{epoch:{digits}d}/{num_epoch}], "
            f"train loss: {train_loss:.4f}, "
            f"train acc: {train_acc * 100:.2f} %, "
            f"train mse: {train_mse :.4f}, "
            f"validation class Acc: {class_acc * 100:.2f} %, "
            f"validation instance Acc: {instance_acc * 100:.2f} %, "
            f"validation  MSE: {val_mse:.4f}, "
            f"best validation Acc: {best_instance_acc * 100:.2f} %"
        )

    return best_model


def main():

    file_lists = get_pointcloud_files(pc_path)

    splitted_data = {
        type: random_split_data(data=filelist, ratio=[7, 1, 2])
        for type, filelist in file_lists.items()
    }

    data = {
        split: {key: v[i] for key, v in splitted_data.items()}
        for i, split in enumerate(["train", "val", "test"])
    }

    # PointNet V1, 30个Epoch, 98 %
    model: nn.Module = get_model(4, normal_channel=False)
    model.apply(inplace_relu)
    model = model.to(device)

    loss_func: nn.Module = get_loss().to(device)

    train(model, loss_func, data["train"], data["val"], 200)


if __name__ == "__main__":
    main()
