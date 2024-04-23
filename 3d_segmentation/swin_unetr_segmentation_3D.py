import os
import json
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
    SpatialCrop,
    SpatialPad, Lambda
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data
from monai.data import decollate_batch, pad_list_data_collate, DataLoader
from functools import partial

from uterus import UterUS

from torch.cuda.amp import autocast, GradScaler

import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)
        print("update " + str(self.val) + " " + str(self.sum) + " " + str(self.count) + " " + str(self.avg))


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def save_checkpoint(model, epoch, filename="model.pt", best_acc=0):
    global ROOT
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(ROOT, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def debug_transform(data):
    print("Current image shape:", data["image"].shape)
    print("Current label shape:", data["label"].shape)
    return data


def get_loader(batch_size, data_dir, fold, roi):
    train_transform = transforms.Compose([
        # transforms.Lambda(print_shape),
        # , keys=["image", "label"], roi_size=roi, random_size=False),
        # transforms.SpatialPadd(keys=["image", "label"], spatial_size=roi, method='symmetric'),  # Adjust size as needed
        transforms.ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=roi),
        transforms.NormalizeIntensityd(keys=["image", "label"], nonzero=True, channel_wise=True),
        transforms.EnsureType()
    ])

    # Example transforms for validation
    val_transform = transforms.Compose([
        # transforms.SpatialPadd(keys=["image", "label"], spatial_size=roi, method='symmetric'),  # Adjust size as needed
        transforms.ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=roi),
        transforms.EnsureType()
    ])

    # Initialize your custom dataset for training and validation
    train_dataset = UterUS(base_dir=data_dir, split='train', transform=train_transform)
    val_dataset = UterUS(base_dir=data_dir, split='test', transform=val_transform)

    # Create PyTorch DataLoaders from your datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_list_data_collate,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Typically validation batch size is set to 1 for medical images
        shuffle=False,
        num_workers=0,
        collate_fn=pad_list_data_collate,
        pin_memory=False
    )
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, epoch, loss_func):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=batch_size)
        print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()
    return run_loss.avg


def val_epoch(
        model,
        loader,
        epoch,
        acc_func,
        model_inferer=None,
        post_sigmoid=None,
        post_pred=None,
):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_tc = run_acc.avg
            # dice_wt = run_acc.avg[1]
            # dice_et = run_acc.avg[2]
            print(
                "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                ", dice_tc:",
                dice_tc,
                # ", dice_wt:",
                # dice_wt,
                # ", dice_et:",
                # dice_et,
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

    return run_acc.avg


def trainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        acc_func,
        scheduler,
        model_inferer=None,
        start_epoch=0,
        post_sigmoid=None,
        post_pred=None,
):
    val_acc_max = 0.0
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_func=loss_func,
        )
        print(
            "Final training  {}/{}".format(epoch, max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            dice_tc = val_acc[0]
            # dice_wt = val_acc[1]
            # dice_et = val_acc[2]
            val_avg_acc = np.mean(val_acc)
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                ", dice_tc:",
                dice_tc,
                # ", dice_wt:",
                # dice_wt,
                # ", dice_et:",
                # dice_et,
                ", Dice_Avg:",
                val_avg_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            dices_tc.append(dice_tc)
            # dices_wt.append(dice_wt)
            # dices_et.append(dice_et)
            dices_avg.append(val_avg_acc)
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                save_checkpoint(
                    model,
                    epoch,
                    best_acc=val_acc_max,
                )
            scheduler.step()
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )


roi = (128, 128, 128)
batch_size = 2
sw_batch_size = 2
fold = 1
infer_overlap = 0.5
max_epochs = 100
val_every = 1
learning_rate = 1e-6

def printParams():
    print("Roi: ", roi)
    print("Batch size: ", batch_size)
    print("Fold: ", fold)
    print("Infer overlap: ", infer_overlap)
    print("Max epochs: ", max_epochs)
    print("Val every: ", val_every)
    print("Learning rate: ", learning_rate)

def main():
    global max_epochs, device, batch_size, val_every
    print_config()

    # data_dir = "C:/Users/Eva/Documents/UterUS/dataset"
    data_dir = "/home/bonese/UterUS/dataset"
    # json_list = "C:/Users/Eva/Documents/UterUS/dataset/train.json"
    train_loader, val_loader = get_loader(batch_size, data_dir, fold, roi)

    printParams()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinUNETR(
        img_size=roi,
        in_channels=1,
        out_channels=1,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    )
    
    weight = torch.load("/home/bonese/tutorials/model_swinvit.pt")
    model.load_from(weights=weight)
    model.to(device)

    torch.backends.cudnn.benchmark = True
    dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True, num_classes=2)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    start_epoch = 0

    (   val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    ) = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        scheduler=scheduler,
        model_inferer=model_inferer,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
    )

    print(f"train completed, best average dice: {val_acc_max:.4f} ")


# ROOT = "C:/Users/Eva/Documents/MONAI-tutorials/3d_segmentation/results"
ROOT = os.environ.get('MONAI_DATA_DIRECTORY')

if __name__ == "__main__":
    main()
