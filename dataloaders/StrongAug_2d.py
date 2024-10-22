import pdb
import copy
import random
import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose, AbstractTransform
from batchgenerators.transforms.utility_transforms import RenameTransform, NumpyToTensor
from batchgenerators.transforms.noise_transforms import GaussianBlurTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform

class RandomSelect(Compose):
    def __init__(self, transforms, sample_num=1):
        super(RandomSelect, self).__init__(transforms)
        self.transforms = transforms
        self.sample_num = sample_num

    def update_list(self, list):
        self.list = list

    def __call__(self, data_dict):

        tr_transforms = random.sample(self.transforms, k=self.sample_num)
        list = copy.deepcopy(self.list)
        if tr_transforms is not None:
            for i in range(len(tr_transforms)):
                list.insert(3, tr_transforms[i])
        # print(list)
        # print(data_dict)
        for t in list:
            # print(t)
            data_dict = t(**data_dict)
        del tr_transforms
        del list

        for key in data_dict.keys():
            if key == "image":
                data_dict[key] = data_dict[key].squeeze(0)
            elif key == "label":
                data_dict[key] = data_dict[key].squeeze(0).squeeze(0)

        return data_dict

class begin(AbstractTransform):
    def __call__(self, **data_dict):
        image, label = data_dict['image'], data_dict['label']
        label = label[np.newaxis, np.newaxis, ...]
        image = image[np.newaxis, np.newaxis, ...]
        return {'image': image, 'label': label}

def get_StrongAug_pixel(patch_size, sample_num, p_per_sample=0.3):
    tr_transforms = []
    tr_transforms_select = []
    tr_transforms.append(begin())
    tr_transforms.append(RenameTransform('image', 'data', True))
    tr_transforms.append(RenameTransform('label', 'seg', True))

    # ========== Pixel-level Transforms =================
    tr_transforms_select.append(GaussianBlurTransform((0.7, 1.3), p_per_sample=p_per_sample))
    tr_transforms_select.append(BrightnessMultiplicativeTransform(p_per_sample=p_per_sample))
    tr_transforms_select.append(ContrastAugmentationTransform(contrast_range=(0.5, 1.5), p_per_sample=p_per_sample))
    tr_transforms_select.append(GammaTransform(invert_image=False, per_channel=True, retain_stats=True, p_per_sample=p_per_sample))  # inverted gamma

    tr_transforms.append(RenameTransform('seg', 'label', True))
    tr_transforms.append(RenameTransform('data', 'image', True))
    tr_transforms.append(NumpyToTensor(['image', 'label'], 'float'))
    trivialAug = RandomSelect(tr_transforms_select, sample_num)
    trivialAug.update_list(tr_transforms)
    return trivialAug

