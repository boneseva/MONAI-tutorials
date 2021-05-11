# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch

from monai.config import print_config
from monai.data import Dataset, DataLoader, create_test_image_3d
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    Resized,
    SaveImaged,
    ScaleIntensityd,
    ToTensord,
)


def main(tempdir):
    print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    print(f"generating synthetic data to {tempdir} (this may take a while)")
    for i in range(5):
        im, _ = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"im{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(tempdir, "im*.nii.gz")))
    files = [{"img": img} for img in images]

    # define pre transforms
    pre_transforms = Compose([
        LoadImaged(keys="img"),
        EnsureChannelFirstd(keys="img"),
        Orientationd(keys="img", axcodes="RAS"),
        Resized(keys="img", spatial_size=(96, 96, 96), mode="trilinear", align_corners=True),
        ScaleIntensityd(keys="img"),
        ToTensord(keys="img"),
    ])
    # define dataset and dataloader
    dataset = Dataset(data=files, transform=pre_transforms)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=4)
    # define post transforms
    post_transforms = Compose([
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold_values=True),
        Invertd(keys="pred", transform=pre_transforms, loader=dataloader, orig_keys="img", nearest_interp=True),
        SaveImaged(keys="pred_inverted", output_dir="./output", output_postfix="seg", resample=False),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    net.load_state_dict(torch.load("best_metric_model_segmentation3d_dict.pth"))

    net.eval()
    with torch.no_grad():
        for d in dataloader:
            images = d["img"].to(device)
            # define sliding window size and batch size for windows inference
            d["pred"] = sliding_window_inference(inputs=images, roi_size=(96, 96, 96), sw_batch_size=4, predictor=net)
            # execute post transforms to invert spatial transforms and save to NIfTI files
            post_transforms(d)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)
