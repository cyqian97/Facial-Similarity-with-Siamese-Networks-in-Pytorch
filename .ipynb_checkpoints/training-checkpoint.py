import math
import numpy as np
from tqdm import tqdm
from torch import save
from os.path import join
from torch.nn.functional import pairwise_distance
from utils import show_plots,decision_stub

# +
def trainSiamese(net,criterion,optimizer,train_dataloader,
                 valid_dataloader,number_epochs,do_show=False):
    counter = []
    train_loss_history = [] 
    valid_loss_history = []
    valid_er_min = math.inf
    valid_loss_min = math.inf
    
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
        valid_loss, valid_er = inferenceSiamese(net,criterion,valid_dataloader)
        valid_loss_history.append(valid_loss)
        
#         valid_distance = np.zeros((valid_dataloader.__len__(),2))
#         valid_loss = 0
#         for i, data in enumerate(tqdm(valid_dataloader),0):
#             img0, img1, label = data
#             img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
#             output1, output2 = net(img0, img1)
#             valid_distance[i,0] = pairwise_distance(output1,output2).detach().cpu().numpy()
#             valid_distance[i,1] = 1-label.detach().cpu().numpy()*2
#             valid_loss += criterion.func(output1, output2, label).detach().cpu().numpy()
#         valid_loss /= valid_dataloader.__len__()
        
        
        print("Epoch-%d\t Train loss: %.4f\t Valid loss: %.4f\t Valid error: %.4f"
              %(epoch,loss_contrastive.item(),valid_loss,valid_er))
    
    
        # Save state_dict if there is any improvement
        if epoch>0:
            if valid_er < valid_er_min:
                valid_er_min = valid_er
                d = optimizer.state_dict()['param_groups'][0]
                dict_name = str(optimizer).split(' ')[0]+" lr-"+str(d['lr']) + " wd-"+str(d['weight_decay'])+ " batch_size-" + str(train_dataloader.batch_size)+ " loss-" + str(loss_contrastive.item())+ " error-" + str(valid_er)+ ".pth"
                save(net.state_dict(),join("state_dict",dict_name))
                print("new model saved")            
            elif valid_loss < valid_loss_min:
                valid_loss_min = valid_loss
                d = optimizer.state_dict()['param_groups'][0]
                dict_name = str(optimizer).split(' ')[0]+" lr-"+str(d['lr']) + " wd-"+str(d['weight_decay'])+ " batch_size-" + str(train_dataloader.batch_size)+ " loss-" + str(loss_contrastive.item())+ " error-" + str(valid_er)+ ".pth"
                save(net.state_dict(),join("state_dict",dict_name))
                print("new model saved")
        else:
            valid_er_min = valid_er
            valid_loss_min = valid_loss
            
    if do_show:
        show_plots(train_loss_history,valid_loss_history,legends = ["train_loss","valid_loss"])
        
    

    return net, train_loss_history, valid_loss_history,dict_name


# -

def inferenceSiamese(net,criterion,dataloader):
        data_distance = np.zeros((dataloader.__len__(),2))
        data_loss = 0
        for i, data in enumerate(tqdm(dataloader),0):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            output1, output2 = net(img0, img1)
            data_distance[i,0] = pairwise_distance(output1,output2).detach().cpu().numpy()
            data_distance[i,1] = 1-label.detach().cpu().numpy()*2
            data_loss += criterion.func(output1, output2, label).detach().cpu().numpy()
        data_loss /= dataloader.__len__()
        
        # Use decision stub to find the best threshhold
        data_er = decision_stub(data_distance.tolist(),verbose=True)
        return data_loss, data_er
