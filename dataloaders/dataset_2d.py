import os
import pdb
import copy
import h5py
import torch
import random
import itertools
import numpy as np
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from scipy.ndimage.interpolation import zoom
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
import torchvision.transforms.functional as TF

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "Synapse" in dataset:
        ref_dict = {"1": 93, "2": 256, "4": 522, "9": 1069, "18": 2211}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def compute_pixel_avg(label_l, label, label_r):
    average = np.empty_like(label)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            values = [label[i, j], label_l[i, j], label_r[i, j]]
            if len(set(values)) < 3:
                average[i, j] = next(x for x in values if values.count(x) > 1)
            else:
                average[i, j] = random.choice(values)
    return average


class Synapse(Dataset):
    def __init__(self, base_dir, file_name='train_2d', num=None, selected_idxs=None, transform=None, strong_transform=None):
        self.transform = transform
        self.strong_transform = strong_transform
        self._base_dir = base_dir
        self.file_name = file_name

        file_path = self._base_dir + '/../' + file_name + '.txt'
        print(file_path)
        with open(file_path, 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]

        if selected_idxs is not None:
            total = list(range(len(self.image_list)))
            excluded_idxs = [x for x in total if x not in selected_idxs]
        else:
            excluded_idxs = []
        for exclude_id in reversed(sorted(excluded_idxs)):
            self.image_list.pop(exclude_id)

        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx].strip('\n')
        if self.file_name == "train":
            data = np.load(os.path.join(self._base_dir, image_name + '.npz'))
            image, label = data['image'], data['label']
        elif self.file_name == "test_vol":
            filepath = self._base_dir + "/{}.npy.h5".format(image_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        if self.strong_transform:
            raw_img = torch.from_numpy(sample['image'].astype(np.float32)).unsqueeze(0)
            sample_strong = self.strong_transform(sample)
            sample['strong_aug'] = sample_strong['image']
            sample['image'] = raw_img
            sample['label'] = torch.from_numpy(sample['label']).long()
        sample['name'] = image_name
        return sample

class NPY_datasets(Dataset):
    def __init__(self, base_dir, train=True, num=None, transform=None, strong_transform=None):
        super(NPY_datasets, self)
        self.transform = transform
        self.strong_transform = strong_transform
        if train:
            images_list = sorted(os.listdir(base_dir + 'train/images/'))
            masks_list = sorted(os.listdir(base_dir + 'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = base_dir + 'train/images/' + images_list[i]
                mask_path = base_dir + 'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            if num is not None:
                self.data = self.data[:num]
        else:
            images_list = sorted(os.listdir(base_dir + 'val/images/'))
            masks_list = sorted(os.listdir(base_dir + 'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = base_dir + 'val/images/' + images_list[i]
                mask_path = base_dir + 'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))     # (256, 256, 3)
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255     # (256, 256, 1)
        img, msk = self.transform((img.astype(np.float32), msk))
        sample = {'image': img, 'label': msk.squeeze()}
        if self.strong_transform:
            raw_img = sample['image']
            sample['image'] = sample['image'].numpy().astype(np.float32)
            sample['label'] = sample['label'].numpy().astype(np.float32)
            sample_strong = self.strong_transform(sample)
            sample['strong_aug'] = sample_strong['image'].squeeze()
            sample['image'] = raw_img
            sample['label'] = torch.from_numpy(sample['label']).long()
        sample['name'] = img_path.split('.')[-2].split('/')[-1]
        return sample

class Polyp_datasets(Dataset):
    def __init__(self, base_dir, train=True, test_dataset='CVC-300', num=None, transform=None, strong_transform=None):
        super(Polyp_datasets, self)
        self.transform = transform
        self.strong_transform = strong_transform
        if train:
            images_list = sorted(os.listdir(base_dir + 'TrainDataset/image/'))
            masks_list = sorted(os.listdir(base_dir + 'TrainDataset/mask/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = base_dir + 'TrainDataset/image/' + images_list[i]
                mask_path = base_dir + 'TrainDataset/mask/' + masks_list[i]
                self.data.append([img_path, mask_path])
            if num is not None:
                self.data = self.data[:num]
        else:
            images_list = sorted(os.listdir(base_dir + 'TestDataset/' + test_dataset + '/images/'))
            masks_list = sorted(os.listdir(base_dir + 'TestDataset/' + test_dataset + '/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = base_dir + 'TestDataset/' + test_dataset + '/images/' + images_list[i]
                mask_path = base_dir + 'TestDataset/' + test_dataset + '/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, msk_path = self.data[index]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255  # (256, 256, 1)

        img, msk = self.transform((img.astype(np.float32), msk))
        sample = {'image': img, 'label': msk.squeeze()}
        if self.strong_transform:
            raw_img = sample['image']
            sample['image'] = sample['image'].numpy().astype(np.float32)
            sample['label'] = sample['label'].numpy().astype(np.float32)
            sample_strong = self.strong_transform(sample)
            sample['strong_aug'] = sample_strong['image'].squeeze()
            sample['image'] = raw_img
            sample['label'] = torch.from_numpy(sample['label']).long()
        sample['name'] = img_path.split('.')[-2].split('/')[-1]
        return sample

class Kvasir(Dataset):
    def __init__(self, base_dir, train=True, num=None, transform=None, strong_transform=None):
        super(Kvasir, self)
        self.transform = transform
        self.strong_transform = strong_transform
        self.base_dir = base_dir
        self.train = train
        if self.train:
            file_path = base_dir + '/train.txt'
        else:
            file_path = base_dir + '/test.txt'

        print(file_path)
        with open(file_path, 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace("\n", "") for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        if self.train:
            img = np.array(Image.open(self.base_dir + "/images/{}".format(image_name)).convert('RGB'))
            msk = np.expand_dims(np.array(Image.open(self.base_dir + "/masks/{}".format(image_name)).convert('L')), axis=2) / 255  # (256, 256, 1)
        else:
            img = np.array(Image.open(self.base_dir + "/images/{}".format(image_name)).convert('RGB'))
            msk = np.expand_dims(np.array(Image.open(self.base_dir + "/masks/{}".format(image_name)).convert('L')), axis=2) / 255  # (256, 256, 1)

        img, msk = self.transform((img.astype(np.float32), msk))
        sample = {'image': img, 'label': msk.squeeze()}
        if self.strong_transform:
            raw_img = sample['image']
            sample['image'] = sample['image'].numpy().astype(np.float32)
            sample['label'] = sample['label'].numpy().astype(np.float32)
            sample_strong = self.strong_transform(sample)
            sample['strong_aug'] = sample_strong['image'].squeeze()
            sample['image'] = raw_img
            sample['label'] = torch.from_numpy(sample['label']).long()
        sample['name'] = image_name
        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        sample = {"image": image, "label": label}
        return sample


class myToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1)

class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        else:
            return image, mask

class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        else:
            return image, mask

class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.rotate(image, self.angle), TF.rotate(mask, self.angle)
        else:
            return image, mask

class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'polyp':
            if train:
                self.mean = 86.17
                self.std = 69.08
            else:
                self.mean = 86.17
                self.std = 69.08
        elif data_name == 'kvasir':
            if train:
                self.mean = 86.17
                self.std = 69.08
            else:
                self.mean = 86.17
                self.std = 69.08

    def __call__(self, data):
        img, msk = data
        img_normalized = (img - self.mean) / self.std
        img_normalized = ((img_normalized - np.min(img_normalized))
                          / (np.max(img_normalized) - np.min(img_normalized))) * 255.
        return img_normalized, msk


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
