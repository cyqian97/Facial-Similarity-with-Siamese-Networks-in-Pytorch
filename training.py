import os
import math
import numpy as np
from tqdm import tqdm
from torch import save
from os.path import join
from torch.nn.functional import pairwise_distance
from utils import show_plots,decision_stub

# +
def trainSiamese(net,criterion,optimizer,scheduler,train_dataloader,
                 valid_dataloader,number_epochs,do_show=False,do_print=True):
    counter = []
    train_loss_history = [] 
    valid_loss_history = []
    
    valid_er_min = math.inf
    valid_loss_min = math.inf
    loss_min = math.inf
    
#     dict_names = []
    dict_name = ''
    
    for epoch in range(0,number_epochs):
        if do_print:print("Epoch ",epoch," training")
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
        scheduler.step()
        
        train_loss_history.append(loss_contrastive.item())
    
        # Empirical error on the validation set
        
        if do_print:print("Epoch ",epoch," validating")
        valid_loss, valid_er = inferenceSiamese(net,criterion,valid_dataloader,do_print = do_print)
        valid_loss_history.append(valid_loss)
        
        
        if do_print:print("Epoch-%d\t Train loss: %.4e\t Valid loss: %.4e\t Valid error: %.4f"
                            %(epoch,loss_contrastive.item(),valid_loss,valid_er))
    
    
        # Save state_dict if there is any improvement
        if epoch>0:
            d = optimizer.state_dict()['param_groups'][0]
            if valid_er < valid_er_min:
                valid_er_min = valid_er
                if dict_name:os.remove(join("state_dict",dict_name))
                dict_name = str(optimizer).split(' ')[0]+" lr-{:.2e} wd-{:.2e} bs-{} train_loss-{:.2e} valid_loss-{:.2e} valid_error-{:.2e}.pth".format(
                    d['lr'],d['weight_decay'],train_dataloader.batch_size, loss_contrastive.item(),valid_loss,valid_er)
                save(net.state_dict(),join("state_dict",dict_name))
#                 dict_names.append(dict_name)
                if do_print:print("new model saved")            
            elif valid_loss < valid_loss_min:
                valid_loss_min = valid_loss
                if dict_name:os.remove(join("state_dict",dict_name))
                dict_name = str(optimizer).split(' ')[0]+" lr-{:.2e} wd-{:.2e} bs-{} train_loss-{:.2e} valid_loss-{:.2e} valid_error-{:.2e}.pth".format(
                    d['lr'],d['weight_decay'],train_dataloader.batch_size, loss_contrastive.item(),valid_loss,valid_er)
                save(net.state_dict(),join("state_dict",dict_name))
#                 dict_names.append(dict_name)
                if do_print:print("new model saved")
                
            # Delete unnecessary checkpoints    
#             if (valid_er <= valid_er_min) and (valid_loss <= valid_loss_min):
#                 for sdict in dict_names[1:-1]:
#                     os.remove(join("state_dict",sdict))
#                 dict_names = []
#                 dict_names.append(dict_name)
        else:
            loss_min = loss_contrastive.item()
            valid_er_min = valid_er
            valid_loss_min = valid_loss
            
    if do_show:
        show_plots(train_loss_history,valid_loss_history,legends = ["train_loss","valid_loss"])
        
#     for sdict in dict_names[1:-1]:
#         os.remove(join("state_dict",sdict))

    return net, train_loss_history, valid_loss_history,dict_name


# +
def trainCNN(net,criterion,optimizer,scheduler,train_dataloader,
                 number_epochs,do_show=False,do_print=True):
    counter = []
    train_loss_history = [] 
    loss_min = math.inf
    
    dict_name = ''
    
    for epoch in range(0,number_epochs):
#         print("Epoch ",epoch," training")
        for i, data in enumerate(train_dataloader,0):
            img0, label = data
            img0, label = img0.cuda(), label.cuda()
            optimizer.zero_grad()
            output1 = net(img0)
            loss_CrossEntropy = criterion(output1,label)
            loss_CrossEntropy.backward()
            optimizer.step()
        scheduler.step()
        
        train_loss_history.append(loss_CrossEntropy.item())
    
#         # Empirical error on the validation set
#         print("Epoch ",epoch," validating")
#         valid_loss = inferenceCNN(net,criterion,valid_dataloader)
#         valid_loss_history.append(valid_loss)
        
        
        if do_print:print("Epoch-%d\t Train loss: %.4e"
                            %(epoch,loss_CrossEntropy.item()))
    
    
        # Save state_dict if there is any improvement
        if epoch>0:
            if loss_CrossEntropy.item() < loss_min:
                if dict_name:
                    os.remove(join("state_dict",dict_name))
                    
                loss_min = loss_CrossEntropy.item()
                d = optimizer.state_dict()['param_groups'][0]
                dict_name = "cnn "+str(optimizer).split(' ')[0]+" lr-{:.2e} wd-{:.2e} bs-{} train_loss-{:.2e}.pth".format(
                    d['lr'],d['weight_decay'],train_dataloader.batch_size, loss_CrossEntropy.item())
                save(net.state_dict(),join("state_dict",dict_name))
                if do_print:print("new model saved")
                
        else:
            loss_min = loss_CrossEntropy.item()
            
    if do_show:
        show_plots(train_loss_history,legends = ["train_loss"])
        
    

    return net, train_loss_history, dict_name


# -

def inferenceSiamese(net,criterion,dataloader,do_print = True):
        data_distance = np.zeros((dataloader.__len__(),2))
        data_loss = 0
        for i, data in enumerate(dataloader,0):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            output1, output2 = net(img0, img1)
            data_distance[i,0] = pairwise_distance(output1,output2).detach().cpu().numpy()
            data_distance[i,1] = 1-label.detach().cpu().numpy()*2
            data_loss += criterion.func(output1, output2, label).detach().cpu().numpy()
        data_loss /= dataloader.__len__()
        
        # Use decision stub to find the best threshhold
        data_er = decision_stub(data_distance.tolist(),verbose=do_print)
        return data_loss, data_er

# +
# def inferenceCNN(net,criterion,dataloader):
#         data_loss = 0
#         for i, data in enumerate(tqdm(dataloader),0):
#             img0, label = data
#             img0, label = img0.cuda(), label.cuda()
#             output1= net(img0)
#             data_loss += criterion.func(output1, output2, label).detach().cpu().numpy()
#         data_loss /= dataloader.__len__()
        
#         return data_loss
