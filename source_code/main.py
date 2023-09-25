import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from VTFNet import VTFNet
from dataset import LeatherDataset
from deeplabv3plus import DeepLabV3Plus


def mean_iou(y_true, y_pred):
    cm = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1))
    intersection = np.diag(cm)  # 交集
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)  # 并集
    IoU = intersection / union  # 交并比，即IoU
    MIoU = np.mean(IoU)  # 计算MIoU
    return MIoU


def mean_pixel_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1))
    pa = np.diag(cm) / np.sum(cm, axis=0)
    mpa = np.nanmean(pa)
    return mpa


def train_one_epoch(model, criterion, optimizer, data_loader, device, scaler=None):
    model.train()
    losses = []
    mious = []
    pbar = tqdm(data_loader, desc="Training...")
    for data in pbar:
        mask = data["mask"].to(device)
        visual = data["visual"].to(device)
        tactile = data["tactile"].to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # preds = model(visual, tactile)
            preds = model(visual)
            # preds = model(torch.cat((visual, tactile), dim=1))
            loss = criterion(preds, mask.long())

            preds = torch.softmax(preds, dim=1)
            _, preds = torch.max(preds, dim=1)
            miou = mean_iou(mask.detach().cpu().long().numpy(), preds.detach().cpu().long().numpy())
            mious.append(miou)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        pbar.set_postfix(**{'train_loss': loss.item(), 'miou': miou})

        losses.append(loss.item())

    return sum(losses) / len(losses), sum(mious) / len(mious)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_set = LeatherDataset(
        visual_dir="./data/datasets/train",
        tactile_dir="./data/datasets/tactile_images",
        masks_dir="./data/datasets/ground_truth"
    )
    train_iter = DataLoader(dataset=train_set, batch_size=8, shuffle=True, num_workers=4, drop_last=True)

    # model = VTFNet()
    model = DeepLabV3Plus(in_channel=3)
    # model = DeepLabV3Plus(in_channel=4)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()),
                                lr=0.01, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()),
    #                   lr=0.0001, betas=(0.9, 0.999), weight_decay=0.001)
    scaler = torch.cuda.amp.GradScaler()

    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        print('\n' + '=' * 100 + '\n' + f'Epoch {epoch} / {num_epochs}' + '\n')
        loss, miou = train_one_epoch(model, criterion, optimizer, train_iter, device, scaler)
        experiment.log({
            'train_loss': loss,
            'miou': miou,
            'epoch': epoch
        })
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/deeplab_epoch={epoch}.pth')


if __name__ == '__main__':
    os.environ["WANDB_API_KEY"] = "c8ddccb46bd291d16653fdaf18a8de222c8ed9af"
    os.environ["WANDB_MODE"] = "dryrun"
    experiment = wandb.init(project='visualTactileFusionNet', resume='allow', anonymous='must')
    main()
