import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from transformers import AdamW
import sys

from sklearn import metrics


batch_size = 32
def read_pt(data_type,use_trans=False):
    # make it easier to load into a batch
    assert data_type == 'train' or 'dev' or 'test'
    if use_trans:
        path = 'hidden_'+data_type+'_trans.pt'
    else:
        path = 'hidden_'+data_type+'.pt'
    data = torch.load(path)
    X=[]
    Y=[]
    A=[]                #A for attribute infor
    D=[]
    for idx,dialog_dict in enumerate(data):     #for a dialog
        input_label = dialog_dict['label']      #list of tensors
        persona_list = dialog_dict['persona']   #list of persona int
        hidden_tensor = dialog_dict['hidden']   #list of tensors for hidden  
        if use_trans:
            utterance_list = dialog_dict['dial']
        for i,d in enumerate(input_label):
            
            X.append(hidden_tensor[i][-1])
            Y.append(input_label[i].squeeze())
            A.append(persona_list[i])
            if use_trans:
                D.append(utterance_list[i])
    if(use_trans):
        return X,Y,A,D
        
    return X,Y,A


class Dataset(Dataset):
    def __init__(self, X,Y,A):
        self.X = X
        self.Y = Y
        self.A = A
       
    def __len__(self):

        return len(self.X)
    
    def __getitem__(self, index):
        
        sample_X = self.X[index]
        sample_Y = self.Y[index]
        sample_A = self.A[index]
        return sample_X, sample_Y, sample_A
        
    def collate(self, unpacked_data):
        return unpacked_data

class model_inv_nn(nn.Module):
    def __init__(self,out_num,in_num=1024):
        super(model_inv_nn, self).__init__()
        self.fc1 = nn.Linear(in_num, out_num)
        #self.act = F.softmax()

    def forward(self, x):
        # x should be of shape (?,1024)
        out = self.fc1(x)
        #out = F.softmax(self.fc1(x),dim=1)

        return out



'''

Model for transformer attackers

'''
class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, targets, mask, label_smoothing=-1, reduce=None):
        """
        reduce: None, "batch", "sentence"
        """
        return sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce)

def sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce):
    # type: (Tensor, Tensor, Tensor, float, bool)-> Tensor
    """
    label_smoothing : ``float``, optional (default = 0.0)
        It should be smaller than 1.
    """
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / float(num_classes)
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
                                       
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(-1, logits.shape[1])
    
    # shape : (batch, sequence_length)
    loss = negative_log_likelihood * mask

    if reduce:
        # shape : (batch,)
        loss = loss.sum(1) / (mask.sum(1) + 1e-13)
        
        if reduce is "batch":
            # shape : scalar
            loss = loss.mean()

    return loss

class Dataset_trans(Dataset):
    def __init__(self, X,Y,A,D):
        self.X = X
        self.Y = Y
        self.A = A
        self.D = D
       
    def __len__(self):

        return len(self.X)
    
    def __getitem__(self, index):
        
        sample_X = self.X[index]
        sample_Y = self.Y[index]
        sample_A = self.A[index]
        sample_D = self.D[index]
        return sample_X, sample_Y, sample_A,sample_D
        
    def collate(self, unpacked_data):
        return unpacked_data


def train_on_batch(batch_X, batch_Y,batch_A,model,optimizer,criterion):
    optimizer.zero_grad()
    output = model(batch_X)
    loss = criterion(output, batch_Y)
    loss.backward()
    optimizer.step()
    print(f'loss: {loss.item()}')

def evaluation(dataloader,model,criterion):
    loss_list = []
    predict = []
    ground_truth = []
    count = 0
    with torch.no_grad():
        for (batch_X, batch_Y,batch_A) in dataloader:
            print(f'count:{count}')
            batch_size = batch_X.size()[0]
            label_size = batch_Y.size()[1]
            # move to gpu
            batch_X = batch_X.cuda()
            batch_Y = batch_Y.cuda()
            batch_A = batch_A.cuda() 
            output = model(batch_X)
            m = nn.Sigmoid()
            batch_out = m(output)
            batch_out[batch_out>=0.5] = 1
            batch_out[batch_out<0.5] = 0
            #eval_metrics(batch_out,batch_Y)   ### what we want 

            loss = criterion(output, batch_Y)
            loss_val = loss.item()
            loss_list.append(loss_val*batch_size)
            predict.extend(batch_out.cpu().detach().numpy())
            ground_truth.extend(batch_Y.cpu().detach().numpy())

            count +=1

            
            
    avg_loss = np.mean(loss_list)
    predict = np.array(predict)
    ground_truth = np.array(ground_truth)
    report_score(ground_truth,predict)
    print(f'avg_loss: {avg_loss}')

def report_score(y_true,y_pred):
    # micro result should be reported
    print("micro precision_score: {:.2f}".format(metrics.precision_score(y_true, y_pred, average='micro')))
    print("macro precision_score: {:.2f} ".format( metrics.precision_score(y_true, y_pred, average='macro')))
    print('=='*20)
    print("micro recall_score: {:.2f}".format(metrics.recall_score(y_true, y_pred, average='micro')))
    print("macro recall_score: {:.2f} ".format( metrics.recall_score(y_true, y_pred, average='macro')))
    print('=='*20)
    print("micro f1_score: {:.2f}".format(metrics.f1_score(y_true, y_pred, average='micro')))
    print("macro f1_score: {:.2f} ".format( metrics.f1_score(y_true, y_pred, average='macro')))




if __name__ == '__main__':
    print('main file')
    data_type = 'dev'
    device = torch.device("cuda")
    X,Y,A = read_pt(data_type)
    train_dataset = Dataset(X,Y,A)
    #batch_size = batch_size
    external_criterion = nn.BCEWithLogitsLoss()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    token_num = len(tokenizer)
    inv_model = model_inv_nn(out_num=token_num)
    inv_model.to(device)
    optimizer = torch.optim.Adam(inv_model.parameters(), 
                  lr=3e-5,
                  eps=1e-06)
    train_dataloader = DataLoader(dataset=train_dataset, 
                              shuffle=True, 
                              batch_size=batch_size)
    #dataloader help covert batch into tensors
    for (batch_X, batch_Y,batch_A) in train_dataloader:
        #print(batch_X.size())      [batch, 1024]
        #print(batch_Y.size())      [batch, 50257]
        #print(batch_A.size())      [batch]
        train_on_batch(batch_X.cuda(), batch_Y.cuda(),batch_A.cuda(),inv_model,optimizer,external_criterion)
