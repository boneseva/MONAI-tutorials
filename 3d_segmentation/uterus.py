import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import nibabel as nib

from monai.data.image_reader import NibabelReader


class UterUS(Dataset):
    """ UterUS Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        print("Init dataset loader "+split)
        
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        if split == 'testing':
            self.split = 'testing'
            print("Just loading one volume")
            files = os.listdir("C:/Users/Eva/Documents/MONAI-tutorials/3d_segmentation/test_volume")
            self.image_list = files

        train_path = self._base_dir+'/train.txt'
        val_path = self._base_dir+'/val.txt'
        test_path = self._base_dir+'/test.txt'

        if split == 'train':
            self.split = 'train'
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'val':
            self.split = 'val'
            with open(val_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            self.split = 'test'
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        
        image_name = self.image_list[idx]

        if self.split == 'testing':
            reader = NibabelReader(channel_dim=None)
            image_data = reader.read("C:/Users/Eva/Documents/MONAI-tutorials/3d_segmentation/test_volume/{}".format(image_name))
            image, metadata = reader.get_data(image_data)

            label_data = reader.read("C:/Users/Eva/Documents/MONAI-tutorials/3d_segmentation/test_label/{}".format(image_name))
            label, metadata = reader.get_data(label_data)
        else:
            reader = NibabelReader(channel_dim=None)
            image_data = reader.read(self._base_dir + "/annotated_volumes/{}.nii.gz".format(image_name))
            image, metadata = reader.get_data(image_data)

            label_data = reader.read(self._base_dir + "/annotations/{}.nii.gz".format(image_name))
            label, metadata = reader.get_data(label_data)
        
        image_t = torch.from_numpy(image)
        label_t = torch.from_numpy(label)
        
        # Ensure that the tensors are cloned if they have been transformed
        image = image_t.clone().detach().unsqueeze(0)
        label = label_t.clone().detach().unsqueeze(0)
        
        image = image.float()
        # image.to(torch.float16)
    
        label = label.float()
        # label.to(torch.float16)
    
        sample = {'image': image, 'label': label, 'name': image_name}

        try:
            if self.transform:
                sample = self.transform(sample)
            return sample


        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            # Return an empty dict or handle the case accordingly
            return sample

def save_volume(data, filename, randint):
    results_folder = "C:/Users/Eva/Documents/MONAI-tutorials/3d_segmentation/augmentation_results/"
    # Save the volume that is metatensor to a file
    # check if data has 5 channels, if so, remove the first channel
    if len(data[0].shape) > 3:
        data = data[0].cpu().detach().numpy()
    if len(data.shape) > 3:
        data = np.squeeze(data[0])
    nib.save(nib.Nifti1Image(data, np.eye(4)), os.path.join(results_folder, filename + randint + "volume.nii.gz"))

class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(
            image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(
            1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)