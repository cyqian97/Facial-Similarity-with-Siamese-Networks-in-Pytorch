# +
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# -

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    """

    def __init__(self, margin=config.margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        print("margin = ",self.margin)

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
    
    def func(self,output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive   

 
