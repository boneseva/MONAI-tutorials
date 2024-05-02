import nibabel as nib
import numpy as np
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

import torch
from monai.networks.nets import SwinUNETR
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def to_tensor(data):
    # add channel dimension
    data = np.expand_dims(data, axis=0)
    data = np.expand_dims(data, axis=0)
    return torch.from_numpy(data).to(device)

def inference(model, input, gt):
    loss_function = DiceLoss(include_background=True, sigmoid=True)
    input = to_tensor(input)
    gt = to_tensor(gt)

    with torch.cuda.amp.autocast():
        logit_map = model(input)
        loss = loss_function(logit_map, gt)
        print(loss)

    return logit_map


def calculate_dice(gt, pred):
    pred = pred.cpu().detach().numpy()
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    gt = gt.cpu().detach().numpy()

    intersection = np.sum(gt * pred)

    ground_o = np.sum(gt)
    pred_o = np.sum(pred)

    denominator = ground_o + pred_o

    return 2.0 * intersection / denominator


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filename = "04_01.nii.gz"
    input = nib.load("C:/Users/Eva/Documents/MONAI-tutorials/3d_segmentation/test_volume/04_01.nii.gz").get_fdata()
    groundtruth = nib.load("C:/Users/Eva/Documents/MONAI-tutorials/3d_segmentation/test_label/04_01.nii.gz").get_fdata()

    test1 = [-10, -10, -10, 100, 100, 100, 100, -10, -10, -10]
    test2 = [1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    test3 = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    test4 = [0, 0, 1, 1, 0, 1, 1, 0, 0, 0]

    test1 = to_tensor(test1)
    test2 = to_tensor(test2)
    test3 = to_tensor(test3)
    test4 = to_tensor(test4)

    loss_function = DiceLoss(include_background=True, sigmoid=True)

    loss2 = loss_function(test1, test3)
    loss3 = loss_function(test1, test4)

    roi = [64, 64, 64]

    torch.backends.cudnn.benchmark = True
    model = SwinUNETR(img_size=roi, in_channels=1, out_channels=1, feature_size=48)
    model.load_state_dict(
        torch.load('C:/Users/Eva/Documents/MONAI-tutorials/3d_segmentation/results/loss.pth'))

    model.to(device)
    model.eval()

    # cut out the center part of the volume
    # input = input[87:151, 62:126, 27:91]
    # gt = groundtruth[87:151, 62:126, 27:91]
    # nib.save(nib.Nifti1Image(input, np.eye(4)), os.path.join(os.path.curdir, filename + "input.nii.gz"))
    # nib.save(nib.Nifti1Image(gt, np.eye(4)), os.path.join(os.path.curdir, filename + "gt.nii.gz"))

    # convert input to float32
    input = input.astype(np.float32)
    out_logit = inference(model, input, groundtruth)

    loss1 = loss_function(out_logit, to_tensor(groundtruth))
    print(loss1)

    nib.save(nib.Nifti1Image(out_logit.cpu().detach().numpy()[0,0,:,:,:].astype(np.float32), np.eye(4)), os.path.join(os.path.curdir, filename + "output.nii.gz"))

    res = torch.sigmoid(out_logit)

    dice = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    calc_dice = calculate_dice(to_tensor(groundtruth), res)
    print(calc_dice)
    nib.save(nib.Nifti1Image(res.cpu().detach().numpy()[0,0,:,:,:].astype(np.float32), np.eye(4)), os.path.join(os.path.curdir, filename + "res.nii.gz"))

