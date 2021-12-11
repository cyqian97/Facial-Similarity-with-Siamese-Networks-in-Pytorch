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
    
    def __init__(self,data_csv,transform=None,should_invert=True):        
        # used to prepare the labels and images pathes
        self.data_csv = pd.read_csv(data_csv)
#         self.directory = directory    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        # getting the image path
        image1_path = self.data_csv.iat[index, 0]
        image2_path = self.data_csv.iat[index, 1]


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
                np.array([int(self.data_csv.iat[index, 2])], dtype=np.float)
            ))
    
    def __len__(self):
        return len(self.data_csv)

class CNNDataset(Dataset):
    
    def __init__(self,data_csv,transform=None,should_invert=True):        
        # used to prepare the labels and images pathes
        self.data_csv = pd.read_csv(data_csv)
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        # getting the image path
        image0_path = self.data_csv.iat[index,0]


        img0 = Image.open(image0_path)
        img0 = img0.convert("L")
        
        
        if self.should_invert:
            img0 = invert(img0)

        if self.transform is not None:
            img0 = self.transform(img0)
            
        return (img0, torch.tensor(int(self.data_csv.iat[index,1])))
    
    def __len__(self):
        return len(self.data_csv)


def generate_csv_siamese(directory,csv_path,total_number=0):
    
    print("Data directory: ",directory)
    
    # Delete .ipynb_checkpoints folder because it confuses ImageFolder function
    if os.path.exists(os.path.join(directory,".ipynb_checkpoints")):
        os.rmdir(os.path.join(directory,".ipynb_checkpoints"))
        
    # load all images
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

    data_list.append(deepcopy(temp))
    
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


def generate_csv_compare(directory,siamese_csv_path,cnn_csv_path,test_csv_path,num_per_class=-1):

    print("Data directory: ",directory)
    
    # Delete .ipynb_checkpoints folder because it confuses ImageFolder function
    if os.path.exists(os.path.join(directory,".ipynb_checkpoints")):
        os.rmdir(os.path.join(directory,".ipynb_checkpoints"))
        
    # load all images
    folder_dataset = ImageFolder(root=directory)
    
    # put the pathes of images with the different classes 
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

    data_list.append(deepcopy(temp))
    
    cnn_data = []
    test_data = []
    pairsT = []
    pairsF = []
    for i in range(len(data_list)):
        # Generate csv for classic cnn training
        for data in data_list[i][1:num_per_class]:
            
            cnn_data.append([data,str(i),folder_dataset.classes[i]])
        
        # Pick the last image in each class to form a testing set
        test_data.append([data_list[i][-1],str(i),folder_dataset.classes[i]])
        
        # Generate all pairs with same and different labels
        for pair in itertools.combinations(data_list[i][1:num_per_class], 2):
            pairsT.append([pair[0], pair[1], '0'])
        for j in range(i+1,len(data_list)):
             for pair in itertools.product(data_list[i][1:num_per_class], data_list[j][1:num_per_class]):
                    pairsF.append([pair[0], pair[1], '1'])
    
    # Select same number of pairs in both cases
    l_min = min(len(pairsT),len(pairsF))
    
    pairsT = random.sample(pairsT, l_min)
    pairsF = random.sample(pairsF, l_min)
    
    with open(test_csv_path,'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerows(test_data)
    
    with open(cnn_csv_path,'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerows(cnn_data)
    
    with open(siamese_csv_path,'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerows(pairsT)
        spamwriter.writerows(pairsF)


if __name__ == '__main__':
    import config
#     generate_csv_siamese(config.training_dir,config.siamese_training_csv)
#     generate_csv_siamese(config.testing_dir,config.siamese_testing_csv) 
    generate_csv_compare(config.training_dir,config.compare_siamese_csv,config.compare_cnn_csv,config.compare_test_csv)


