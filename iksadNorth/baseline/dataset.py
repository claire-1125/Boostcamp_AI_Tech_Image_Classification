from cProfile import label
import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split, WeightedRandomSampler
# from torchvision import transforms
# from torchvision.transforms import *
from albumentations import *
from albumentations.pytorch import ToTensorV2

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            CenterCrop(320, 256, p=1.),
            Resize(resize[0], resize[1], Image.BILINEAR, p=1.),
            Normalize(mean=mean, std=std),
            ToTensorV2(p=1.),
        ], p=1.)

    def __call__(self, image):
        return self.transform(image=image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.resize = resize
        self.mean = mean
        self.std = std
        self.transform =  Compose([
            CenterCrop(320, 256, p=1.),
            Resize(resize[0], resize[1], Image.BILINEAR, p=1.),
            ShiftScaleRotate(shift_limit=0.05, rotate_limit=20, p=.7),
            RandomBrightnessContrast(p=.7),
            OneOf([
                FancyPCA(p=.5),
                GaussNoise(p=.2),
            ], p=1.),
            Normalize(mean=mean, std=std),
            ToTensorV2(p=1.0),
        ], p=1.)

    def __call__(self, image):
        return self.transform(image=image)


class RandAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.resize = resize
        self.mean = mean
        self.std = std
        self.transform =  Compose([
            ShiftScaleRotate(shift_limit=0.05, rotate_limit=20, p=.7),
            RandomBrightnessContrast(p=.7),
            OneOf([
                FancyPCA(p=1.),
                GaussNoise(p=.5),
            ], p=1.),
            Normalize(mean=mean, std=std),
            ToTensorV2(p=1.0),
        ], p=1.)

    def __call__(self, image):
        return self.transform(image=image)
    
class CustomAug_without_oneof:
    def __init__(self, resize, mean, std, **args):
        self.resize = resize
        self.mean = mean
        self.std = std
        self.transform =  Compose([
            CenterCrop(320, 256, p=1.),
            Resize(resize[0], resize[1], Image.BILINEAR, p=1.),
            ShiftScaleRotate(shift_limit=0.05, rotate_limit=20, p=.7),
            RandomBrightnessContrast(p=.7),
            # OneOf([
            #     FancyPCA(p=.5),
            #     GaussNoise(p=.2),
            # ], p=1.),
            Normalize(mean=mean, std=std),
            ToTensorV2(p=1.0),
        ], p=1.)

    def __call__(self, image):
        return self.transform(image=image)
    

class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }
    
    image_paths = []
    profiles = []
    
    mask_labels = []
    gender_labels = []
    age_labels = []
    multi_labels = []  
    
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.downsample = True
        
        self.setup()
    
    def __getitem__(self, index):
        return self.read_image(index), self.multi_labels[index]
        
    def __len__(self):
        return len(self.image_paths)
    
    def setup(self):
        self.profiles = [profile for profile in os.listdir(self.data_dir) if not profile.startswith('.')]
        for profile in self.profiles:
            _, gender, _, age = profile.split("_")
            gender_label = GenderLabels.from_str(gender)
            age_label = AgeLabels.from_number(age)
            
            include_mask = True
            img_folder = os.path.join(self.data_dir, profile)
            lst_files = os.listdir(img_folder)
            random.shuffle(lst_files) # in-place operation (add randomness in selected images with mask label)
            for file_name in lst_files:
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue
                # if ext != '.jpg': # contains some png files
                #     continue
                if self.downsample and file_name.startswith('mask'):
                    if not include_mask:
                        continue
                    include_mask = False # include only 1 mask image per profile
                
                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
                self.multi_labels.append(self.encode_multi_class(mask_label, gender_label, age_label))

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def read_image(self, index):
        # read an image from directory and return it as a numpy array
        image_path = self.image_paths[index]
        return np.array(Image.open(image_path))


class MaskSplitByProfileDataset(MaskBaseDataset):
    # calc_statistics self.image_paths[:3000]:
    #   [0.5573112  0.52429302 0.50174594] [0.61373778 0.58633636 0.56743769]
    def __init__(self, data_dir, label='multi', n_fold:int=2, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super().__init__(data_dir, mean, std)
        
        self.label = label
        self.target_label = []
        self.set_target_label()
        
        self.n_fold = n_fold
        self.kfold_indices = []
        self.stratified_kfold()
        self.indices = self.kfold_indices[0]
        
        self.class_weights = self.compute_class_weight()
        
    def set_target_label(self):
        if self.label == 'multi':
            self.num_classes = 3 * 2 * 3
            self.target_label = self.multi_labels
        elif self.label == 'mask':
            self.num_classes = 3
            self.target_label = self.mask_labels
        elif self.label == 'gender':
            self.num_classes = 2
            self.target_label = self.gender_labels
        elif self.label == 'age':
            self.num_classes = 3
            self.target_label = self.age_labels
        else:
            raise ValueError(f"label must be 'multi', 'mask', 'gender', or 'age', {self.label}")

    def __getitem__(self, index):
        return self.read_image(index), self.target_label[index]

    def stratified_kfold(self):
        from sklearn.model_selection import StratifiedKFold
        
        profile_labels = []
        for profile in self.profiles:
            _, gender, _, age = profile.split("_")
            gender_label = GenderLabels.from_str(gender)
            age_label = AgeLabels.from_number(age)
            profile_label = self.encode_multi_class(0, gender_label, age_label)
            profile_labels.append(profile_label)
        
        skf = StratifiedKFold(n_splits=self.n_fold)
        for train_profiles, val_profiles in skf.split(self.profiles, profile_labels):
            train_index, val_index = [], []
            for profile_idx in train_profiles:
                train_index.extend(range(profile_idx*3, profile_idx*3+3))
            for profile_idx in val_profiles:
                val_index.extend(range(profile_idx*3, profile_idx*3+3))
            self.kfold_indices.append({
                'train': train_index,
                'val': val_index
            })

    def set_indices(self, idx:int=0):
        # set train/val indices of specified kfold indices
        self.indices = self.kfold_indices[idx]

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]
    
    def get_train_labels(self, label):
        # returns train data of the input label
        train_index = self.indices['train']
        return [label[idx] for idx in train_index]
    
    def get_classweight_label(self, label) -> torch.tensor:
        # returns class weight of a label within train dataset
        train_labels = self.get_train_labels(label)
        _, n_samples = np.unique(train_labels, return_counts=True)
        weights = 1. / torch.tensor(n_samples, dtype=torch.float)
        return weights
    
    def normalize_weight(self, weights):
        norm_weights = [1 - (weight / sum(weights)) for weight in weights]
        return torch.tensor(norm_weights, dtype=torch.float)

    ##################### need refactoring ##################### 
    def weight0(self):
        pass
        # v0: weights on target label
        # train_index = self.indices['train'] # indices of train dataset
        # train_labels = [self.target_label[idx] for idx in train_index] # target_label of train dataset
        # class_counts = np.array([len(np.where(train_labels==t)[0]) for t in np.unique(train_labels)]) # get counts of each class 
        # weights = 1. / torch.tensor(class_counts, dtype=torch.float) # get weights (more class count == less weight(frequent) it will be sampled)
        # samples_weights = weights[train_labels] # map weights for each train dataset, len(samples_weights) == len(train dataset)
        # return WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    
    def weight1(self):
        pass
        # # v1: normalized weights on target label (better than v0)
        # sample_weight = [self.class_weights[self.target_label[idx]] for idx in self.indices['train']]
        # return WeightedRandomSampler(weights=sample_weight, num_samples=len(sample_weight), replacement=True)

    def weight2(self):
        pass
        # # v2: normalized weights on of specific ratio ``age=.9 : gender=.1``
        # age_weight = self.get_classweight_label(self.age_labels)
        # gender_weight = self.get_classweight_label(self.gender_labels)
        # weights = [age_weight[self.age_labels[idx]]*.9 + gender_weight[self.gender_labels[idx]]*.1 for idx in self.indices['train']]
        # return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    def weight3(self):
        # v3: normalized weights on multi label
        multi_weight = self.get_classweight_label(self.multi_labels)
        multi_weight = self.normalize_weight(multi_weight)
        sample_weight = [multi_weight[self.multi_labels[idx]] for idx in self.indices['train']]
        return WeightedRandomSampler(weights=sample_weight, num_samples=len(sample_weight), replacement=True)

    def weight4(self):
        # v4: weights on multi label
        multi_weight = self.get_classweight_label(self.multi_labels)
        sample_weight = [multi_weight[self.multi_labels[idx]] for idx in self.indices['train']]
        return WeightedRandomSampler(weights=sample_weight, num_samples=len(sample_weight), replacement=True)

    def get_weighted_sampler(self, ver: int=0) -> WeightedRandomSampler:  
        """
        returns WeightedRandomSampler based on the distribution of the train label
        used to prevent overfitting due to unbalanced dataset
        """
        if ver==0: return self.weight0()
        elif ver==1: return self.weight1()
        elif ver==2: return self.weight2()
        elif ver==3: return self.weight3()
        elif ver==4: return self.weight4()
        else: raise ValueError(f'invalid version of {ver}')

    def compute_class_weight(self) -> torch.tensor:
        """
        estimate class weights for unbalanced dataset
        `` 1 - n_sample / sum(n_samples) ````
        used for loss function: weighted_cross_entropy
        """
        train_index = self.indices['train']
        train_labels = [self.target_label[idx] for idx in train_index]
        _, n_samples = np.unique(train_labels, return_counts=True)
        norm_weights = [1 - (sample / sum(n_samples)) for sample in n_samples]
        return torch.tensor(norm_weights, dtype=torch.float).to(device='cuda')


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.transform = None
    
    def __getitem__(self, idx):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"
        
        np_img, label = self.dataset[self.indices[idx]]
        image_transform = self.transform(np_img)['image']
        return image_transform, label

    def __len__(self):
        return len(self.indices)
    
    def set_transform(self, transform):
        self.transform = transform


class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = Compose([
            CenterCrop(320, 256, p=1.),
            Resize(resize[0], resize[1], Image.BILINEAR),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], p=1)

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image_np = np.array(image)
            trans_image = self.transform(image=image_np)['image']
        return trans_image

    def __len__(self):
        return len(self.img_paths)

import torch
from torchvision import transforms

import numpy as np
import pandas as pd
from pandas import DataFrame

from itertools import cycle
from copy import deepcopy
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from dataset import MaskBaseDataset, Subset
from typing import Tuple, List
from torch.utils.data import Dataset
from einops import rearrange

from dataset import MaskSplitByProfileDataset

class SamplingDataset(MaskSplitByProfileDataset):    
    image_paths = []
    
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=None, std=None, val_ratio=0.2, label='multi', n_fold:int=5):
        # super().__init__(data_dir, label=label, n_fold=n_fold, mean=mean, std=std, val_ratio=val_ratio)
        
        # self.class_weights = self.compute_class_weight()
        
        tmp = label
        label_to_num = {'multi':3*2*3, 'age':3, 'gender':2, 'mask':3}
        
        self.label_name = tmp.lower()
        self.num_classes = label_to_num[self.label_name]
        
        self.data_dir = data_dir
        self.dataframe = pd.read_csv(Path(data_dir) / 'info.csv')
        
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio # validation ratio 
        
        self.transform = None
        self.setup()
        self.calc_statistics()

        self.n_fold = n_fold
        self.kfold_indices = []
        self.stratified_kfold()
        self.indices = self.kfold_indices[0]
        
    def setup(self):
        for _, row in tqdm(self.dataframe.iterrows()):
            img_path = row['new_loc']
            mask_label, gender_label, age_label = self.decode_multi_class(row['label'])
            
            self.image_paths.append(img_path)
            self.mask_labels.append(mask_label)
            self.gender_labels.append(gender_label)
            self.age_labels.append(age_label)

    def __getitem__(self, index):
        print(self.transform)
        # assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        if self.label_name == 'multi':
            mask_label = self.get_mask_label(index)
            gender_label = self.get_gender_label(index)
            age_label = self.get_age_label(index)
            final_label = self.encode_multi_class(mask_label, gender_label, age_label)
        else:
            label_fn = getattr(self, f'get_{self.label_name}_label')
            final_label = label_fn(index) 

        image_np = self.read_image(index)
        image_transform = self.transform(image_np)['image']
        
        return image_transform, final_label

    def read_image(self, index):
        # read an image from directory and return it as a numpy array
        image_path = self.image_paths[index]
        return np.load(image_path)
    
    
    def split_dataset(self) -> Tuple[Subset, Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]
        
    
    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                # image = np.array(Image.open(image_path)).astype(np.int32)
                #### 새로 추가한 부분 #########
                image = np.load(image_path).astype(np.int32)
                image = rearrange(image, 'c h w -> h w c')
                ##############################
                sums.append(image.mean(axis=(0, 1)))
                
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255
            print(self.mean, self.std)
    
    
    def stratified_kfold(self):
        TOLERANCE = 10
        NUM_SUB = 1
        NUM_TRIAL = 10  
        
        dataframe_arranged = self.dataframe.reset_index()
        indices = dataframe_arranged.drop_duplicates(['id'], inplace=False).id.tolist()
        
        print("stratified_kfold outside")
        N = self.n_fold
        print()
        for _ in range(N):
            print("stratified_kfold inside")
            
            n_val = int(len(indices) * self.val_ratio)
            n_train = len(indices) - n_val
            
            indices_train = np.random.choice(indices, size=n_train, replace=False).tolist()
            indices_val = [idx for idx in indices if not idx in indices_train]
            
            for _ in range(NUM_TRIAL):
                print("stratified_kfold trial")
                
                cond_None = pd.isna(dataframe_arranged['Augment'])
                cond_id = dataframe_arranged['id'].isin(indices_val)
                    
                val_set = dataframe_arranged[cond_None & cond_id]
                
                low_bound, high_bound = n_val - TOLERANCE // 2, n_val + TOLERANCE // 2
                if low_bound <= val_set.__len__() <= high_bound:
                    break
                
                indices_val += np.random.choice(indices_train, size=NUM_SUB, replace=False).tolist()
            
            train_set = dataframe_arranged.index.difference(val_set.index)
            val_set = val_set.index
            
            train_set, val_set = train_set.tolist(), val_set.tolist()
            
            self.kfold_indices.append({
                'train':train_set, 
                'val':val_set
                })
        
    def compute_class_weight(self) -> torch.tensor:
        return None