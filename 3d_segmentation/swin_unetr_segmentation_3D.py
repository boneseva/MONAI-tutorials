import os
import json
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from monai.losses import DiceLoss, DiceCELoss
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

from tqdm import tqdm

from uterus import UterUS

from torch.cuda.amp import autocast, GradScaler

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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

def calculate_roi_size():
    image_size = data["image"].shape
    # the ROI size should be divisible by 2^5 and only padded, not cropped
    return tuple((np.array(image_size) // 32 + 1) * 32)

def save_checkpoint(model, epoch, filename="model.pt", best_acc=0):
    global ROOT
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(ROOT, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def get_loader(batch_size, data_dir, roi, roi_validation):
    # Define transformations for the training dataset
    train_transform = transforms.Compose([
        transforms.RandSpatialCropd(keys=["image", "label"], roi_size=roi, random_size=False),
        transforms.ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=roi),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        transforms.EnsureTyped(keys=["image", "label"], track_meta=True),
    ])

    # Define transformations for the validation dataset
    val_transform = transforms.Compose([
        transforms.RandSpatialCropd(keys=["image", "label"], roi_size=roi_validation, random_size=False),
        transforms.ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=roi_validation),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        transforms.EnsureTyped(keys=["image", "label"], track_meta=True),
    ])

    # Initialize the custom UterUS dataset class for training and validation
    train_dataset = UterUS(base_dir=data_dir, split='train', transform=train_transform)
    val_dataset = UterUS(base_dir=data_dir, split='test', transform=val_transform)

    # Create DataLoader for the training dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_list_data_collate,
        pin_memory=False
    )

    # Create DataLoader for the validation dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Often, validation batch size is set to 1 for medical images
        shuffle=False,
        num_workers=0,
        collate_fn=pad_list_data_collate,
        pin_memory=False
    )

    return train_loader, val_loader


def save_volume(data, filename):
    # Save the volume that is metatensor to a file
    # check if data has 5 channels, if so, remove the first channel
    if len(data[0].shape) > 3:
        array = data[0].cpu().detach().numpy()
    if len(array.shape) > 3:
        array = np.squeeze(array[0])
    nib.save(nib.Nifti1Image(array, np.eye(4)), filename + "volume.nii.gz")


def validation(epoch_iterator_val, test=False):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            if test:
                val_inputs, val_labels, val_name = (batch["image"].cuda(), batch["label"].cuda(), batch["name"])

            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, roi_size=None, sw_batch_size=sw_batch_size, predictor=model)

            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = val_labels_list
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            if test:
                directory = os.path.join(ROOT, "test_results")
                save_volume(val_labels_list, os.path.join(directory, val_name[0] + 'labels'))
                save_volume(val_outputs_list, os.path.join(directory, val_name[0] + 'result'))

            dice_metric(y_pred=val_output_convert, y=val_labels_convert)

            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        with torch.cuda.amp.autocast():
            logit_map = model(x)
            loss = loss_function(logit_map, y)
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_iterator.set_description(  # noqa: B038
            f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})"
        )

        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val, True)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            dice_val_best = dice_val
            global_step_best = global_step
            torch.save(model.state_dict(), os.path.join(ROOT, "best_metric_model.pth"))
            if dice_val > dice_val_best:
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


roi = (160, 160, 128)
roi_validation = (160, 160, 128)
batch_size = 2
sw_batch_size = 1
infer_overlap = 0.5
learning_rate = 1e-3
max_iterations = 100000
eval_num = 100


def printParams():
    print("Roi: ", roi)
    print("Batch size: ", batch_size)
    print("Infer overlap: ", infer_overlap)
    print("Max epochs: ", max_iterations)
    print("Val every: ", eval_num)
    print("Learning rate: ", learning_rate)


def main():
    global max_epochs, device, batch_size, val_every, learning_rate, model, optimizer, scheduler, dice_loss, post_sigmoid, post_pred, dice_acc, \
        model_inferer, loss_function, scaler, dice_metric, global_step, dice_val_best, global_step_best, epoch_loss_values, metric_values, data_dir, \
        train_loader, val_loader, roi, sw_batch_size, infer_overlap, max_iterations, eval_num, post_label
    print_config()

    # data_dir = "C:/Users/Eva/Documents/UterUS/dataset"
    data_dir = "/home/bonese/UterUS/dataset"
    train_loader, val_loader = get_loader(batch_size, data_dir, roi, roi_validation)

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

    # weight = torch.load("C:/Users/Eva/Documents/MONAI-tutorials/3d_segmentation/results/best_metric_model.pth")
    # model.load_from(weights=weight)
    # model.load_state_dict(
    #     torch.load(r'C:\Users\Eva\Documents\MONAI-tutorials\3d_segmentation\results\best_metric_model.pth'))
    model.to(device)

    torch.backends.cudnn.benchmark = True
    loss_function = DiceCELoss(include_background=True, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    post_label = AsDiscrete(threshold=0.5)
    post_pred = AsDiscrete(threshold=0.5)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
    model.load_state_dict(torch.load(os.path.join(ROOT, "best_metric_model.pth")))
    print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")


# ROOT = "C:/Users/Eva/Documents/MONAI-tutorials/3d_segmentation/results"
ROOT = os.environ.get('MONAI_DATA_DIRECTORY')

if __name__ == "__main__":
    main()
