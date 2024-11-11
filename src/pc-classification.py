# Standard Library
import sys
import copy
import datetime
import importlib
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
pc_path = base_dir / "data/pc"
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
    model: nn.Module,
    train_loader: DataLoader,
    loss_func: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> torch.FloatTensor:

    model.train(True)

    acc, losses = [], []

    logits: torch.FloatTensor
    loss: torch.FloatTensor
    label: torch.LongTensor
    points: torch.FloatTensor
    for i, (points, label) in enumerate(train_loader):
        optimizer.zero_grad()

        label, augmented_points = (
            label.to(device),
            augment_pointcloud(points).to(device),
        )

        logits, trans_feature = model(augmented_points)
        loss = loss_func(logits, label, trans_feature)

        loss.backward()
        optimizer.step()

        pred_choice = logits.data.max(1)[1]

        acc.append(calculate_acc(label, pred_choice))
        losses.append(loss.item())

        digits = len(f"{len(train_loader)}")
        logger.info(f"\t\t batch [{i:{digits}d}/{len(train_loader)}], {loss=:.4f}")

    return sum(losses) / len(losses), sum(acc) / len(acc)


@torch.no_grad()
def test_epoch(model: nn.Module, loader: DataLoader, num_class: int = 4):

    model.eval()

    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    label: torch.FloatTensor
    points: torch.FloatTensor

    # sourcery skip: no-loop-in-tests
    for points, label in loader:
        points, label = points.to(dtype=torch.float32, device=device), label.to(device)

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(label.cpu()):
            classacc = (
                pred_choice[label == cat]
                .eq(label[label == cat].long().data)
                .cpu()
                .sum()
            )
            class_acc[cat, 0] += classacc.item() / float(points[label == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(label.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def train(
    model: nn.Module,
    loss_func: nn.Module,
    train_data: dict[str, list[Path]],
    val_data: dict[str, list[Path]],
    num_epoch: int,
):

    valset, trainset = (
        PointCloudDataset(val_data, 1024, "random"),
        PointCloudDataset(train_data, 1024, "random"),
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

        train_loss, train_acc = train_epoch(model, train_loader, loss_func, optimizer)

        instance_acc, class_acc = test_epoch(model, val_loader, num_class=4)

        scheduler.step()

        if best_instance_acc < instance_acc:
            best_instance_acc = instance_acc
            best_model = copy.deepcopy(model)

        logger.info(
            f"Epoch: [{epoch:{digits}d}/{num_epoch}], "
            f"train loss: {train_loss:.4f}, "
            f"train acc: {train_acc * 100:.2f} %, "
            f"validation class Acc: {class_acc * 100:.2f} %, "
            f"validation instance Acc: {instance_acc * 100:.2f} %, "
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

    model: nn.Module = get_model(4, normal_channel=False)
    model.apply(inplace_relu)
    model = model.to(device)

    loss_func: nn.Module = get_loss().to(device)

    train(model, loss_func, data["train"], data["val"], 200)


if __name__ == "__main__":
    main()
