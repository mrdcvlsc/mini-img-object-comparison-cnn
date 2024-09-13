import os
import glob
from typing import Tuple
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        print('CustomImageDataset.__init__() : CustomImageDataset Initializing')
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        #############################################################################

        self.shapeImgPaths = []

        print("CustomImageDataset.__init__() : reading all the circle image filenames")
        circleImages = glob.glob1(os.path.join(img_dir, "C"), "*.png")
        for circleImage in circleImages:
            self.shapeImgPaths.append(os.path.join(img_dir, "C", circleImage))

        print("CustomImageDataset.__init__() : reading all the square image filenames")
        squareImages = glob.glob1(os.path.join(img_dir, "B"), "*.png")
        for squareImage in squareImages:
            self.shapeImgPaths.append(os.path.join(img_dir, "B", squareImage))

        print("CustomImageDataset.__init__() : reading all the star image filenames")
        starImages = glob.glob1(os.path.join(img_dir, "S"), "*.png")
        for starImage in starImages:
            self.shapeImgPaths.append(os.path.join(img_dir, "S", starImage))

        print("CustomImageDataset.__init__() : reading all the triangle image filenames")
        triangleImages = glob.glob1(os.path.join(img_dir, "T"), "*.png")
        for triangleImage in triangleImages:
            self.shapeImgPaths.append(os.path.join(img_dir, "T", triangleImage))

        # keep only 50% of the dataset (too many)
        self.shapeImgPaths = self.shapeImgPaths[0 : len(self.shapeImgPaths) // 2]

        print('CustomImageDataset.__init__() : expected data set len  =', len(self.shapeImgPaths)**2)
        
        #############################################################################

        print("CustomImageDataset.__init__() : building the dataset paths and labels")
        
        self.datasetPathsAndLabels : list[Tuple[str, str, float]] = []
        i = 0
        while i < len(self.shapeImgPaths):
            if (i % (len(self.shapeImgPaths) // 10) == 0):
                print((i + 1)  / (len(self.shapeImgPaths)) * 100, '%')
            j = 0

            pathPartOfAnchorImg  = os.path.split(self.shapeImgPaths[i])[0]
            
            while j < len(self.shapeImgPaths):
                pathPartOfSubjectImg = os.path.split(self.shapeImgPaths[j])[0]

                self.datasetPathsAndLabels.append((
                    self.shapeImgPaths[i],
                    self.shapeImgPaths[j],
                    1. if pathPartOfAnchorImg == pathPartOfSubjectImg else 0.
                ))
                j += 1
            i += 1

        #############################################################################

        print('CustomImageDataset.__init__() : generated data set len =', len(self.datasetPathsAndLabels))

    def __len__(self):
        return len(self.datasetPathsAndLabels)

    def __getitem__(self, idx):
        anchor_image  = read_image(self.datasetPathsAndLabels[idx][0], mode=ImageReadMode.GRAY)
        subject_image = read_image(self.datasetPathsAndLabels[idx][1], mode=ImageReadMode.GRAY)
        label         = self.datasetPathsAndLabels[idx][2]
        
        if self.transform:
            anchor_image  = self.transform(anchor_image)
            subject_image = self.transform(subject_image)
        if self.target_transform:
            label = self.target_transform(label)
        return anchor_image, subject_image, label

from torch.utils.data import DataLoader

training_data = CustomImageDataset('S')

training_dataloader = DataLoader(training_data, batch_size=2, shuffle=True, num_workers=2)

anchor_img, subject_img, label = next(iter(training_dataloader))

print('training_data_loader anchor_img  =', anchor_img)
print('training_data_loader subject_img =', subject_img)
print('training_data_loader label       =', label)