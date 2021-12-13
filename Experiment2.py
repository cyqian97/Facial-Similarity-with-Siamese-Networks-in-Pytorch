#!/usr/bin/env python3
import torch
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from copy import deepcopy
import torchvision.utils
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,random_split

import config
from utils import imshow
from models import SiameseNetwork
from training import trainSiamese,inferenceSiamese
from datasets import SiameseNetworkDataset
from loss_functions import ContrastiveLoss

# generate_csv(config.training_dir)

import os
if not os.path.exists('state_dict'):
    os.makedirs('state_dict')
print("background process started")
errors = []
margins = np.arange(0.5,6.5,0.5)
for margin in margins:
    e_temp = []
    for i in range(4):
        siamese_dataset = SiameseNetworkDataset(config.siamese_training_csv,
                                                transform=transforms.Compose([
                                                    transforms.Resize((config.img_height,config.img_width)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(0,1)]),
                                                should_invert=False)

        # Split the dataset into train, validation and test sets
        num_train = round(0.9*siamese_dataset.__len__())
        num_validate = siamese_dataset.__len__()-num_train
        siamese_train, siamese_valid = random_split(siamese_dataset, [num_train,num_validate])
        train_dataloader = DataLoader(siamese_train,
                                shuffle=True,
                                num_workers=8,
                                batch_size=config.train_batch_size)
        valid_dataloader = DataLoader(siamese_valid,
                                shuffle=True,
                                num_workers=8,
                                batch_size=1)

        net = SiameseNetwork().cuda()
        criterion = ContrastiveLoss(margin = margin)
        optimizer = optim.Adam(net.parameters(),lr = config.learning_rate )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,config.step_size, config.gamma)

        net, train_loss_history, valid_loss_history,dict_name = trainSiamese(net,criterion,optimizer,scheduler,train_dataloader,
                     valid_dataloader,config.train_number_epochs,do_show=True,do_print=False)

        net = SiameseNetwork().cuda()
        net.load_state_dict(torch.load(os.path.join("state_dict",dict_name)))
        net.eval()

        siamese_test = SiameseNetworkDataset(config.siamese_testing_csv,
                                                transform=transforms.Compose([transforms.Resize((config.img_height,config.img_width)),
                                                                              transforms.ToTensor(),
                                                                              transforms.Normalize(0,1)
                                                                              ])
                                               ,should_invert=False)
        test_dataloader = DataLoader(siamese_test,num_workers=8,batch_size=1,shuffle=True)
        dataiter = iter(test_dataloader)

        test_loss, test_er = inferenceSiamese(net,criterion,test_dataloader,do_print=False)
        print("Test loss: %.4f\t Test error: %.4f"
              %(test_loss, test_er))
        e_temp.append(test_er)
        print(e_temp)
    errors.append(deepcopy(e_temp))
    print(errors)


errors_min = [np.min(t) for t in errors]


fig = plt.figure()
plt.plot(margins,errors_min)
plt.xlabel("Margins in the contrastive loss")
plt.ylabel("Error rate on the testing set")
fig.savefig('destination_path.eps', format='eps', dpi=1200)

# +


margins = np.arange(0.5,6.5,0.5)
errors =[[0.20891364902506965, 0.15041782729805014, 0.18662952646239556, 0.20334261838440112], [0.1894150417827298, 0.19220055710306408, 0.12256267409470752, 0.1532033426183844], [0.1977715877437326, 0.17548746518105848, 0.1894150417827298, 0.23119777158774374], [0.18384401114206128, 0.1532033426183844, 0.15041782729805014, 0.16991643454038996], [0.20334261838440112, 0.17270194986072424, 0.181058495821727, 0.15041782729805014], [0.11977715877437325, 0.16991643454038996, 0.1532033426183844, 0.1615598885793872], [0.3983286908077994, 0.17827298050139276, 0.13649025069637882, 0.1671309192200557], [0.15041782729805014, 0.17827298050139276, 0.11977715877437325, 0.1894150417827298], [0.17548746518105848, 0.1309192200557103, 0.1615598885793872, 0.19220055710306408], [0.15877437325905291, 0.1894150417827298, 0.13370473537604458, 0.14484679665738162], [0.1532033426183844, 0.16434540389972144, 0.181058495821727, 0.14206128133704735], [0.116991643454039, 0.15598885793871867, 0.1532033426183844, 0.18662952646239556]]
errors_mean = [np.mean(t) for t in errors]
errors_std = [np.std(t) for t in errors]
fig = plt.figure()
plt.errorbar(margins,errors_mean,yerr = errors_std, fmt = '-o', capsize = 3)
plt.xlabel("Margins in the contrastive loss")
plt.ylabel("Error rate on the testing set")
fig.savefig('destination_path.eps', format='eps', dpi=1200)

# +
accS_mean = [np.mean(t) for t in accS]
accS_std = [np.std(t) for t in accS]
accC_mean = [np.mean(t) for t in accC]
accC_std = [np.std(t) for t in accC]

fig = plt.figure()
plt.errorbar(nums,accS_mean, yerr=accS_std, fmt='-o',capsize=3)#, color='black',
             #ecolor='lightgray', elinewidth=3, capsize=0);
plt.errorbar(nums,accC_mean, yerr=accC_std, fmt='-o', capsize=3)#, color='black',
            # ecolor='lightgray', elinewidth=3, capsize=0);
plt.xlabel("Number of images in each class")
plt.ylabel("Accuracy on the testing set")
plt.legend(["Siamese + Contrastive loss","Single CNN + Cross-entropy loss"])
fig.savefig('destination_path.eps', format='eps', dpi=1200)
