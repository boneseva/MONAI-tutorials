import nibabel as nib
import numpy as np
import os

filenames = ["04_01.nii.gz", "04_02.nii.gz", "04_03.nii.gz", "04_04.nii.gz"]
for filename in filenames:
    input_or = nib.load("C:/Users/Eva/Documents/UterUS/dataset/annotated_volumes/" + filename).get_fdata()
    groundtruth_or = nib.load("C:/Users/Eva/Documents/UterUS/dataset/annotations/" + filename).get_fdata()

    size = [64,64,64]
    centerx, centery, centerz = [input_or.shape[0]//2, input_or.shape[1]//2, input_or.shape[2]//2]

    input = input_or[centerx-size[0]//2:centerx+size[0]//2, centery-size[1]//2:centery+size[1]//2, centerz-size[2]//2:centerz+size[2]//2]
    gt = groundtruth_or[centerx-size[0]//2:centerx+size[0]//2, centery-size[1]//2:centery+size[1]//2, centerz-size[2]//2:centerz+size[2]//2]
    nib.save(nib.Nifti1Image(input, np.eye(4)), os.path.join("C:/Users/Eva/Documents/MONAI-tutorials/3d_segmentation/test_label", filename))
    nib.save(nib.Nifti1Image(gt, np.eye(4)), os.path.join("C:/Users/Eva/Documents/MONAI-tutorials/3d_segmentation/test_volume", filename))
