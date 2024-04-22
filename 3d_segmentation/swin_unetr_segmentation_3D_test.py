import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json

import numpy as np
import nibabel as nib

from monai.inferers import sliding_window_inference
from monai import transforms

from monai.config import print_config
from monai.networks.nets import SwinUNETR
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

roi = (128, 128, 64)
batch_size = 1
sw_batch_size = 2
fold = 1
infer_overlap = 0.5
max_epochs = 100
val_every = 10

def main():
    global max_epochs, device, batch_size, val_every
    print_config()

    data_dir = "C:/Users/Eva/Documents/UterUS/dataset"
    # json_list = "/home/bonese/UterUS/dataset/train.json"

    # Example transforms for validation
    test_transform = transforms.Compose([
        # transforms.SpatialPadd(keys=["image", "label"], spatial_size=roi, method='symmetric'),  # Adjust size as needed
        transforms.EnsureType()
    ])

    test_dataset = UterUS(base_dir=data_dir, split='test', transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Typically validation batch size is set to 1 for medical images
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

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
    ).to(device)

    torch.backends.cudnn.benchmark = True

    model.load_state_dict(torch.load(os.path.join(ROOT, "model.pt"))["state_dict"])
    model.to(device)
    model.eval()

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=1,
        predictor=model,
        overlap=0.6,
    )

    with torch.no_grad():
        for batch_data in test_loader:
            image = batch_data["image"].cuda()
            prob = torch.sigmoid(model_inferer_test(image))
            seg = prob[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            # seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            # save seg_out as nifti file
            seg_ch = seg.squeeze(0)
            seg_out = nib.Nifti1Image(seg_ch, np.eye(4))
            name = batch_data["name"]
            seg_out.to_filename(os.path.join(ROOT, name[0]+".result.nii.gz"))

ROOT = "C:/Users/Eva/Documents/MONAI-tutorials/3d_segmentation/results"
 
if __name__ == "__main__":

    main()