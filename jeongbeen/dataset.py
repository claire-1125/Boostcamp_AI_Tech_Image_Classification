import os
import random
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
# from torchvision import transforms
# from torchvision.transforms import *
from albumentations import *
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold


IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

"""
모든 augmentation은 torchvision이 아닌 albumentation 기반이다.
"""

## 기본 preprocessing
######## CenterCrop과 Resize 파트 고민해보기
class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            CenterCrop(320, 256, p=1.),
            Resize(resize[0], resize[1], Image.BILINEAR, p=1.),
            Normalize(mean=mean, std=std),
            ToTensorV2(p=1.)
        ], p=1.)

    def __call__(self, image):
        return self.transform(image=image)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


## (real) augmentation
######## CenterCrop과 Resize 파트 고민해보기
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



## (real) augmentation
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
    

class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2  # not wear


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()  # male or female
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
    ##### 각 카테고리에 대해 분류기를 만들 경우 이를 변경해야 한다!! 
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

    # 각 카테고리에 대한 이미지의 클래스 리스트
    mask_labels, gender_labels, age_labels, multi_labels = [], [], [], []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean, self.std = mean, std
        # self.val_ratio = val_ratio
        self.downsample = True
        
        # self.transform = None
        self.setup()
        # self.calc_statistics()

    
    def __getitem__(self, index):
        # assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        # mask_label = self.get_mask_label(index)
        # gender_label = self.get_gender_label(index)
        # age_label = self.get_age_label(index)
        # multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        # image = self.read_image(index)
        # image_np = np.array(image)
        # image_transform = self.transform(image_np)['image']
        # return image_transform, multi_class_label

        return self.read_image(index), self.multi_labels[index]


    def __len__(self):
        return len(self.image_paths)


    
    ## 이미지 절대경로, 마스크 착용상태, 성별, 나이대를 멤버변수로 설정
    def setup(self):
        # ["000001_female_Asian_45","000002_female_Asian_52",...]
        # profiles = os.listdir(self.data_dir) # 지정한 디렉토리("/opt/ml/input/data/train/images") 내 모든 파일과 디렉토리의 리스트 
        
        self.profiles = [profile for profile in os.listdir(self.data_dir) if not profile.startswith('.')]

        # 한 사람씩 진행
        for profile in self.profiles:
            # if profile.startswith("."):  # "." 로 시작하는 디렉토리 무시
            #     continue

            _, gender, _, age = profile.split("_")
            gender_label = GenderLabels.from_str(gender)
            age_label = AgeLabels.from_number(age)
            
            include_mask = True


            # /opt/ml/input/data/train/images/000001_female_Asian_45
            img_folder = os.path.join(self.data_dir, profile)
            
            lst_files = os.listdir(img_folder)
            random.shuffle(lst_files)

            # 특정 사람의 각 마스크 상태에 대해 반복
            for file_name in lst_files:  # ['incorrect_mask.jpg', 'mask1.jpg',...,'normal.jpg']
                _file_name, ext = os.path.splitext(file_name)   # ('incorrect_mask', '.jpg')
                if _file_name not in self._file_names:  # "." 로 시작하는 파일(임시파일) 및 invalid 한 파일들은 무시
                    continue

                if self.downsample and file_name.startswith('mask'):
                    if not include_mask:
                        continue
                    include_mask = False


                 # /opt/ml/input/data/train/images/000001_female_Asian_45/incorrect_mask.jpg
                img_path = os.path.join(self.data_dir, profile, file_name)  # 특정 한 장에 대한 절대경로
                mask_label = self._file_names[_file_name]  # mask:0, incorrect:1, normal:2

                # _, gender, _, age = profile.split("_")  # 000001, female, Asian, 45
                # gender_label = GenderLabels.from_str(gender)  # male:0, female:1
                # age_label = AgeLabels.from_number(age)  # 30세 미만:0, 60세 미만:1, 60세 이상:2

                self.image_paths.append(img_path)  # "opt/ml/input/data/train/images/000001_female_Asian_45/incorrect_mask.jpg"
                self.mask_labels.append(mask_label)  # 1
                self.gender_labels.append(gender_label)  # 1
                self.age_labels.append(age_label)  # 1
                self.multi_labels.append(self.encode_multi_class(mask_label, gender_label, age_label))


    # def calc_statistics(self):
    #     has_statistics = self.mean is not None and self.std is not None  # self.mean, self.std 값 있는지 여부 체크
    #     if not has_statistics:  # self.mean, self.std 값이 없다면 직접 계산
    #         print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
    #         sums = []
    #         squared = []
    #         for image_path in self.image_paths[:3000]:  # 각 사진의 절대경로들. 앞의 3000장에 대해서만 계산 진행
    #             image = np.array(Image.open(image_path)).astype(np.int32)  # 각 이미지를 PIL → np.array로 형변환
    #             sums.append(image.mean(axis=(0, 1)))
    #             squared.append((image ** 2).mean(axis=(0, 1)))

    #         self.mean = np.mean(sums, axis=0) / 255  # 255로 나누는 이유? normalize!! (0 ~ 1 사이의 값으로 나오도록)
    #         self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255
    #         # print(self.mean, self.std)


    # def set_transform(self, transform):
    #     self.transform = transform


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
        image_path = self.image_paths[index]
        return np.array(Image.open(image_path))


    # def split_dataset(self) -> Tuple[Subset, Subset]:
    #     """
    #     데이터셋을 train 과 val 로 나눕니다,
    #     pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
    #     torch.utils.data.Subset 클래스 둘로 나눕니다.
    #     구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
    #     """
    #     n_val = int(len(self) * self.val_ratio)
    #     n_train = len(self) - n_val
    #     train_set, val_set = random_split(self, [n_train, n_val])
    #     return train_set, val_set


## 꼭 이걸로 돌려야 한다!!
class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """
    
    def __init__(self, data_dir, label='multi', n_fold:int=2, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        # self.indices = defaultdict(list)  # value의 datatype이 list...!!
        
        # 실질적으로 해당 클래스에서 override한 setup() 메소드 호출
        super().__init__(data_dir, mean, std)


        self.label = label  # 분류기를 하나만 사용할지, 여러개를 사용할지의 tag (multi vs. mask vs. gender vs. age)
        # self.multi_labels = []  # 분류기를 하나(multi)만 사용하는 경우의 각 이미지 클래스 리스트
        # self.downsample = True  # down-sampling할지의 여부
        
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
            self.target_label = self.multi_labels  # setup()에서 세팅한 결과가 넣어진다.
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
        """
        return 형태
            1st : np.array(Image.open(self.image_paths[index]))
            2nd : 클래스값
        """
        return self.read_image(index), self.target_label[index]


    def stratified_kfold(self):
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
        """
        train으로 split된 사진들의 label값을 return한다.
        """
        train_index = self.indices['train']  # {"train":[0,4,8,245,..], "val":[3,884,...]} (사진 기준으로 indexing된다.)
        return [label[idx] for idx in train_index]


    # normalize되지 않은 weight 할당
    def get_classweight_label(self, label) -> torch.tensor:
        """
        train image에 대해 각 label 개수에 반비례하는 weight 설정
        """
        train_labels = self.get_train_labels(label)  # train으로 분류된 사진들의 클래스 label값 리스트
        # np.unique() : unique한 값들만 오름차순으로 정렬, return_counts : 각 원소가 몇 개 있었나 출력
        _, n_samples = np.unique(train_labels, return_counts=True)  # 각 label에 해당하는 사진 장 수 counting한 결과. array([6, 4, 5, 35,...])
        weights = 1. / torch.tensor(n_samples, dtype=torch.float)  # 개수에 반비례하게 weight 설정 [1/6, 1/4, 1/5, 1/35, ...]
        return weights
    

    # normalize된 weight 할당
    def normalize_weight(self, n_samples):
        # n_samples : get_classweight_label이 return한 weight 리스트
        norm_weights = [1 - (sample / sum(n_samples)) for sample in n_samples]
        return torch.tensor(norm_weights, dtype=torch.float)


    ##################### need refactoring ##################### 
    def weight0(self):
        # v0: weights on target label
        train_index = self.indices['train'] # indices of train dataset
        train_labels = [self.target_label[idx] for idx in train_index] # target_label of train dataset
        class_counts = np.array([len(np.where(train_labels==t)[0]) for t in np.unique(train_labels)]) # get counts of each class 
        weights = 1. / torch.tensor(class_counts, dtype=torch.float) # get weights (more class count == less weight(frequent) it will be sampled)
        samples_weights = weights[train_labels] # map weights for each train dataset, len(samples_weights) == len(train dataset)
        return WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    
    def weight1(self):
        # v1: normalized weights on target label (better than v0)
        sample_weight = [self.class_weights[self.target_label[idx]] for idx in self.indices['train']]
        return WeightedRandomSampler(weights=sample_weight, num_samples=len(sample_weight), replacement=True)

    def weight2(self):
        # # v2: normalized weights on of specific ratio ``age=.9 : gender=.1``
        age_weight = self.get_classweight_label(self.age_labels)
        gender_weight = self.get_classweight_label(self.gender_labels)
        weights = [age_weight[self.age_labels[idx]]*.9 + gender_weight[self.gender_labels[idx]]*.1 for idx in self.indices['train']]
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

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


    # 각 label 개수에 반비례하는 정규화된 weight 할당
    def compute_class_weight(self) -> torch.tensor:
        """
        estimate class weights for unbalanced dataset
        `` 1 - n_sample / sum(n_samples) ````
        used for loss function: weighted_cross_entropy
        """
        train_index = self.indices['train']  # train으로 지정된 이미지들의 index 리스트
        train_labels = [self.target_label[idx] for idx in train_index]  # train으로 지정된 이미지들의 label값 리스트
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
    

   
