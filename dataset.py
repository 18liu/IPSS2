import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms as transforms
from torch.utils.data import Dataset
from sampling_alter import prior_guided_patch_sampling
import warnings


warnings.filterwarnings("ignore", message="Series.__getitem__ treating keys as positions is deprecated.*")


class MyDataset(Dataset):
    def __init__(self, csv_path, transform=None, seed=100):
        self.seed = seed
        self.data = pd.read_csv(csv_path)
        ref_name = self.data.iloc[:, 0].tolist()
        dis_name = self.data.iloc[:, 1].tolist()
        self.ref_image_name = ref_name
        self.dis_image_name = dis_name
        self.transforms = transform
    
    def __len__(self):
        return len(self.ref_image_name)
    
    def __getitem__(self, idx):
        
        for index, row in self.data.iterrows():
            if row[1] == self.dis_image_name[idx]:
                label = row[2]
            
        ref_image_path = os.path.join('/mnt/10T/rjl/dataset/final_430_ref', self.ref_image_name[idx])
        dis_image_path = os.path.join('/mnt/10T/rjl/dataset/final_10320_dis/10k_dis_10320', self.dis_image_name[idx])
        ref_image = Image.open(ref_image_path)
        dis_image = Image.open(dis_image_path)
            
        dis_image = self.transforms(dis_image)
        ref_image = self.transforms(ref_image)
            
        dis_patchs = prior_guided_patch_sampling(dis_image, patch_num=10, p_h=0.2, p_m=0.6, p_l=0.2, seed=self.seed)
        ref_patchs = prior_guided_patch_sampling(ref_image, patch_num=10, p_h=0.2, p_m=0.6, p_l=0.2, seed=self.seed)
        
        sample = (dis_patchs, ref_patchs, label)
        
        return sample


