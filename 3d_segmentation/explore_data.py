# read the data in a directory and find the largest shape in each dimension
import os
import nibabel as nib
import numpy as np

directory = r"C:\Users\Eva\Documents\UterUS\dataset\annotated_volumes"
shapes = []
for filename in os.listdir(directory):
    if filename.endswith(".nii.gz"):
        image = nib.load(os.path.join(directory,filename))
        shapes.append(image.shape)
print("Max shape in each dimension: ", np.max(shapes, axis=0))
print("Min shape in each dimension: ", np.min(shapes, axis=0))
