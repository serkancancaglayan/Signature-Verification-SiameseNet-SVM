import os
import cv2 as cv
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class SignatureData(Dataset):

    def __init__(self, img_folder, annotation_csv, transform=None, target_transform=None):

        self.img_folder = img_folder
        self.annotation_df = pd.read_csv(annotation_csv)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):

        return len(self.annotation_df)
    

    def __getitem__(self, index):

        img1_abs_path = os.path.join(self.img_folder, self.annotation_df.iat[index, 0])
        img2_abs_path = os.path.join(self.img_folder, self.annotation_df.iat[index, 1])
        label = self.annotation_df.iat[index, 2]

        #img1 = cv.resize(cv.imread(img1_abs_path), (self.img_size, self.img_size))
        #img2 = cv.resize(cv.imread(img2_abs_path), (self.img_size, self.img_size))
        img1 = Image.open(img1_abs_path)
        img2 = Image.open(img2_abs_path)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label

