import os
import csv
import torch
import random
import itertools
import numpy as np
import pandas as pd
from PIL import Image
from copy import deepcopy
from PIL.ImageOps import invert
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset



class SiameseNetworkDataset(Dataset):
    
    def __init__(self,training_dir,training_csv,transform=None,should_invert=True):        
        # used to prepare the labels and images pathes
        self.train_df = pd.read_csv(training_csv)
        self.training_dir = training_dir    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        # getting the image path
        image1_path = self.train_df.iat[index, 0]
        image2_path = self.train_df.iat[index, 1]


        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        
        if self.should_invert:
            img0 = invert(img0)
            img1 = invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            
        return (img0, img1 , torch.from_numpy(
                np.array([int(self.train_df.iat[index, 2])], dtype=np.float)
            ))
    
    def __len__(self):
        return len(self.train_df)

def generate_csv(directory,csv_path,total_number=0):
    #load all images
    print("Data directory: ",directory)
    if os.path.exists(os.path.join(directory,".ipynb_checkpoints")):
        os.rmdir(os.path.join(directory,".ipynb_checkpoints"))
    folder_dataset = ImageFolder(root=directory)
    
    #put the pathes of images with the different classes 
    data_list = []
    temp = []
    current_label = folder_dataset[0][1]
    for img_path,label in folder_dataset.imgs:
        if label==current_label:
            temp.append(img_path)
        else:
            current_label = label
            data_list.append(deepcopy(temp))
            temp = []
            temp.append(img_path)
            
    # Generate all pairs with same and different labels
    pairsT = []
    pairsF = []
    for i in range(len(data_list)):
        for pair in itertools.combinations(data_list[i], 2):
            pairsT.append([pair[0], pair[1], '0'])
        for j in range(i+1,len(data_list)):
             for pair in itertools.product(data_list[i], data_list[j]):
                    pairsF.append([pair[0], pair[1], '1'])
    
    # Select same number of pairs in both cases
    l_min = min(len(pairsT),len(pairsF))
    if 0 < total_number < 2*l_min:
        l_min = round(total_number/2)
        
    pairsT = random.sample(pairsT, l_min)
    pairsF = random.sample(pairsF, l_min)
    
    with open(csv_path,'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerows(pairsT)
        spamwriter.writerows(pairsF)


if __name__ == '__main__':
    import config
    generate_csv(config.training_dir,config.training_csv)
    generate_csv(config.testing_dir,config.testing_csv)
#     sigT = []
#     sigF = []
#     dir_list  = os.listdir(directory)
#     dir_list.sort()
#     for directory in dir_list[0:-1:2]:
#         for root, dirs, files in os.walk(os.path.join(config.training_dir, directory)):
#             sigT = deepcopy(files)
#         for root, dirs, files in os.walk(os.path.join(config.training_dir, directory + "_forg")):
#             sigF = deepcopy(files)
#         for pair in itertools.combinations(sigT, 2):
#             rows.append([os.path.join(directory, pair[0]), os.path.join(directory, pair[1]), '0'])
#         for pair in itertools.product(sigT, sigF):
#             rows.append([os.path.join(directory, pair[0]), os.path.join(directory + "_forg", pair[1]), '1'])
#     if 0 < total_number < len(rows):
#         rows = random.sample(rows, total_number)



