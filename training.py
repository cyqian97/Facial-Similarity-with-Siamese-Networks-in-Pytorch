import math
import numpy as np
from tqdm import tqdm
from torch.nn.functional import pairwise_distance
from utils import show_plots,decision_stub

def trainSiamese(net,criterion,optimizer,train_dataloader,
                 valid_dataloader,number_epochs,do_show=False):
    counter = []
    train_loss_history = [] 
    valid_loss_history = []
    for epoch in range(0,number_epochs):
        print("Epoch ",epoch," training")
        for i, data in enumerate(tqdm(train_dataloader),0):
            img0, img1 , label = data
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
        train_loss_history.append(loss_contrastive.item())
    
        # Empirical error on the validation set
        print("Epoch ",epoch," validating")
        valid_distance = np.zeros((valid_dataloader.__len__(),2))
        valid_loss = 0
        for i, data in enumerate(tqdm(valid_dataloader),0):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            output1, output2 = net(img0, img1)
            valid_distance[i,0] = pairwise_distance(output1,output2).detach().cpu().numpy()
            valid_distance[i,1] = 1-label.detach().cpu().numpy()*2
            valid_loss += criterion.func(output1, output2, label).detach().cpu().numpy()
        valid_loss /= valid_dataloader.__len__()
        valid_loss_history.append(valid_loss)
        
        # Use decision stub to find the best threshhold
        valid_er = decision_stub(valid_distance.tolist(),verbose=True)
        
        print("Epoch-%d\t Train loss: %.4f\t Valid loss: %.4f\t Valid error: %.4f"
              %(epoch,loss_contrastive.item(),valid_loss,valid_er))
    
    if do_show:
        show_plots(train_loss_history,valid_loss_history,legends = ["train_loss","valid_loss"])
        
    

    return net, train_loss_history, valid_loss_history
