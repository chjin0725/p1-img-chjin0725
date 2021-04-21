import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import random
import os
from PIL import Image
import numpy as np
from glob import glob

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

class TrainDataset(Dataset):

    def __init__(self, data_root, input_size=(384, 512), transform=None):
        super().__init__()


        self.img_list, self.label_list = self._load_img_list(data_root)
#         self.meta_data = pd.read_csv(os.path.join(meta_root, 'train.csv'))
        self.len = len(self.img_list)
        self.input_size = input_size
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_list[index]
#         print(img_path)
        # Image Loading
        img = Image.open(img_path)


        if self.transform:
            img = self.transform(img)

        # Ground Truth
        label = self.label_list[index]

        return img, label

    def __len__(self):
        return self.len

    def _load_img_list(self, data_root):


        folder_list = glob(data_root + '/*')
        random.shuffle(folder_list)
        img_list = []
        label_list = []
        
        for dir in folder_list:
            images = glob(dir+'/*')
            img_list.extend(images)
            dir_name = os.path.basename(dir)
            _, sex, _, age = dir_name.split('_')
            for img in images: 
                label_list.append(self._get_class_idx_from_img_name(img, sex, int(age)))
                
        return img_list, label_list


    def _get_class_idx_from_img_name(self, img_path, sex, age):
        img_name = os.path.basename(img_path)
        gender_offset = 3 if 'female' == sex else 0
        
        if age < 30:
            age_offset = 0
        elif age < 60:
            age_offset = 1
        else:
            age_offset = 2
        
         
        if 'incorrect' in img_name:
            mask_offset = 6
        elif 'mask' in img_name:
            mask_offset = 0
        elif 'normal' in img_name:
            mask_offset = 12
        else:
            raise ValueError("%s is not a valid filename. Please change the name of %s." % (img_name, img_path))
        
        return gender_offset + age_offset + mask_offset
    

    
class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
    

def img_transform(train = True):
    
    if train:
        transform = transforms.Compose([transforms.CenterCrop((500,250)),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])

                                        ])
       
    else:
        transform = transforms.Compose([transforms.CenterCrop((500,250)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])

                                        ])
    
    return transform

class TrainDatasetAgeAugmentation(Dataset):
    '''age로 label을 나눌 때 60세에서 58세로 기준을 바꿈.'''
    def __init__(self, data_root, input_size=(384, 512), transform=None):
        super().__init__()


        self.img_list, self.label_list = self._load_img_list(data_root)
#         self.meta_data = pd.read_csv(os.path.join(meta_root, 'train.csv'))
        self.len = len(self.img_list)
        self.input_size = input_size
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_list[index]
#         print(img_path)
        # Image Loading
        img = Image.open(img_path)


        if self.transform:
            img = self.transform(img)

        # Ground Truth
        label = self.label_list[index]

        return img, label

    def __len__(self):
        return self.len

    def _load_img_list(self, data_root):


        folder_list = glob(data_root + '/*')
        random.shuffle(folder_list)
        img_list = []
        label_list = []
        
        for dir in folder_list:
            images = glob(dir+'/*')
            img_list.extend(images)
            dir_name = os.path.basename(dir)
            _, sex, _, age = dir_name.split('_')
            for img in images: 
                label_list.append(self._get_class_idx_from_img_name(img, sex, int(age)))

        return img_list, label_list


    def _get_class_idx_from_img_name(self, img_path, sex, age):
        img_name = os.path.basename(img_path)
        gender_offset = 3 if 'female' == sex else 0
        
        if age < 30:
            age_offset = 0
        elif age < 58:
            age_offset = 1
        else:
            age_offset = 2
        
         
        if 'incorrect' in img_name:
            mask_offset = 6
        elif 'mask' in img_name:
            mask_offset = 0
        elif 'normal' in img_name:
            mask_offset = 12
        else:
            raise ValueError("%s is not a valid filename. Please change the name of %s." % (img_name, img_path))
        
        return gender_offset + age_offset + mask_offset
    

    
class TrainDatasetWithAge(Dataset):
    '''return also age infomation in _load_img_list'''
    def __init__(self, data_root, input_size=(384, 512), transform=None):
        super().__init__()


        self.img_list, self.label_list, self.age_list = self._load_img_list(data_root)
#         self.meta_data = pd.read_csv(os.path.join(meta_root, 'train.csv'))
        self.len = len(self.img_list)
        self.input_size = input_size
        self.transform = transform
        
    def __getitem__(self, index):
        img_path = self.img_list[index]
#         print(img_path)
        # Image Loading
        img = Image.open(img_path)


        if self.transform:
            img = self.transform(img)

        # Ground Truth
        label = self.label_list[index]

        return img, label, self.age_list[index]

    def __len__(self):
        return self.len

    def _load_img_list(self, data_root):


        folder_list = glob(data_root + '/*')
        random.shuffle(folder_list)
        img_list = []
        label_list = []
        
        age_list = []
        
        for dir in folder_list:
            images = glob(dir+'/*')
            img_list.extend(images)
            dir_name = os.path.basename(dir)
            _, sex, _, age = dir_name.split('_')
            for img in images: 
                label_list.append(self._get_class_idx_from_img_name(img, sex, int(age)))
                age_list.append(int(age))
        return img_list, label_list, age_list


    def _get_class_idx_from_img_name(self, img_path, sex, age):
        img_name = os.path.basename(img_path)
        gender_offset = 3 if 'female' == sex else 0
        
        if age < 30:
            age_offset = 0
        elif age < 58:
            age_offset = 1
        else:
            age_offset = 2
        
         
        if 'incorrect' in img_name:
            mask_offset = 6
        elif 'mask' in img_name:
            mask_offset = 0
        elif 'normal' in img_name:
            mask_offset = 12
        else:
            raise ValueError("%s is not a valid filename. Please change the name of %s." % (img_name, img_path))
        
        return gender_offset + age_offset + mask_offset