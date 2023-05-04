import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine
sys.path.append('..')
import json
import numpy as np
import pandas as pd
import argparse


from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AdamW,get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from attacker_models import SequenceCrossEntropyLoss
from sentence_transformers import SentenceTransformer
from simcse_persona import get_persona_dict
from attacker_evaluation_gpt import eval_on_batch
from datasets import load_dataset

class linear_projection(nn.Module):
    def __init__(self, in_num, out_num=1024):
        super(linear_projection, self).__init__()
        self.fc1 = nn.Linear(in_num, out_num)

    def forward(self, x, use_final_hidden_only = True):
        # x should be of shape (?,in_num) according to gpt2 output
        out_shape = x.size()[-1]
        assert(x.size()[1] == out_shape)
        out = self.fc1(x)


        return out


class personachat(Dataset):
    def __init__(self, data):
        self.data = data


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        text = self.data[index]

        return  text
        
    def collate(self, unpacked_data):
        return unpacked_data

def process_data(data,batch_size,device,config,need_porj=False):
    dataset = personachat(data)
    dataloader = DataLoader(dataset=dataset, 
                              shuffle=True, 
                              batch_size=batch_size, 
                              collate_fn=dataset.collate)

    print('load data done')

    ### extra projection
    if need_porj:
        projection = linear_projection(in_num=768).to(device)
    ### for attackers
    model_attacker = AutoModelForCausalLM.from_pretrained('dialogpt_qnli')
    tokenizer_attacker = AutoTokenizer.from_pretrained(config['model_dir'])
    criterion = SequenceCrossEntropyLoss()
    model_attacker.to(device)
    model_attacker.eval()
    param_optimizer = list(model_attacker.named_parameters())
    no_decay = ['bias', 'ln', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}

    ]
    num_gradients_accumulation = 1
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    num_train_optimization_steps  = len(dataloader) * num_epochs // num_gradients_accumulation
    optimizer = AdamW(optimizer_grouped_parameters, 
                  lr=3e-5,
                  eps=1e-06)
    if need_porj:
        optimizer.add_param_group({'params': projection.parameters()})
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=100, 
                                            num_training_steps = num_train_optimization_steps)
    
    ### process to obtain the embeddings
    for i in range(num_epochs):
        running_ppl = []
        for idx,batch_text in enumerate(dataloader):

            record_loss, perplexity = train_on_batch(batch_D=batch_text,model=model_attacker,tokenizer=tokenizer_attacker,criterion=criterion,device=device,train=False)

            running_ppl.append(perplexity)

        print(f'Validate ppl: {np.mean(running_ppl)}')



def train_on_batch(batch_D,model,tokenizer,criterion,device,train=True):
    padding_token_id = tokenizer.encode(tokenizer.eos_token)[0]
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(batch_D, return_tensors='pt', padding='max_length', truncation=True, max_length=40)

    input_ids = inputs['input_ids'].to(device) # tensors of input ids
    labels = input_ids.clone()

    past = None

    logits, past = model(input_ids,past_key_values  = past,return_dict=False)
    logits = logits[:, :-1].contiguous()
    target = labels[:, 1:].contiguous()

    target_mask = torch.ones_like(target).float()

    loss = criterion(logits, target, target_mask, label_smoothing=0.02, reduce="batch")   

    record_loss = loss.item()
    perplexity = np.exp(record_loss)
    if train:
        loss.backward()

    return record_loss, perplexity


def read_logs(path):
    with open(path) as f:
        data = json.load(f)
    pred  = data["pred"]
    return pred

def get_val_ppl(path,batch_size,device,config):
    sent_list = read_logs(path)
    process_data(sent_list,batch_size,device,config)

def get_qnli_data(data_type):
    dataset = load_dataset('glue','qnli', cache_dir="/home/hlibt/embed_rev/data", split=data_type)
    sentence_list = []
    for i,d in enumerate(dataset):
        sentence_list.append(d['question'])
        sentence_list.append(d['sentence'])
    return sentence_list
def get_personachat_data(data_type):

    sent_list = get_persona_dict(data_type=data_type)
    return sent_list

def get_sent_list(config):
    dataset = config['dataset']
    data_type = config['data_type']
    if dataset == 'personachat':
        sent_list = get_personachat_data(data_type)
        return sent_list
    elif dataset == 'qnli':
        sent_list = get_qnli_data(data_type)
        return sent_list
    else:
        print('Name of dataset only supports: personachat or qnli')
        sys.exit(-1)
if __name__ == '__main__':
    model_cards ={}
    model_cards['sent_t5'] = 'sentence-t5-large'
    model_cards['mpnet'] = 'all-mpnet-base-v1'
    model_cards['sent_roberta'] = 'all-roberta-large-v1'
    model_cards['simcse_bert'] = 'princeton-nlp/sup-simcse-bert-large-uncased'
    model_cards['simcse_roberta'] = 'princeton-nlp/sup-simcse-roberta-large'

    parser = argparse.ArgumentParser(description='Training external NN as baselines')
    parser.add_argument('--model_dir', type=str, default='microsoft/DialoGPT-medium', help='Dir of your model')
    parser.add_argument('--num_epochs', type=int, default=1, help='Training epoches.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch_size #.')
    parser.add_argument('--dataset', type=str, default='personachat', help='Name of dataset: personachat or qnli')
    #parser.add_argument('--dataset', type=str, default='qnli', help='Name of dataset: personachat or qnli')
    parser.add_argument('--data_type', type=str, default='test', help='train/test')
    #parser.add_argument('--data_type', type=str, default='test', help='train/test')
    parser.add_argument('--embed_model', type=str, default='sent_t5', help='Name of embedding model: mpnet/sent_roberta/simcse_bert/simcse_roberta/sent_t5')
    parser.add_argument('--decode', type=str, default='beam', help='Name of decoding methods: beam/sampling')
    #parser.add_argument('--embed_model', type=str, default='simcse_roberta', help='Name of embedding model: mpnet/sent_roberta/simcse_bert/simcse_roberta/sent_t5')
    args = parser.parse_args()
    config = {}
    config['model_dir'] = args.model_dir
    config['num_epochs'] = args.num_epochs
    config['batch_size'] = args.batch_size
    config['dataset'] = args.dataset
    config['data_type'] = args.data_type
    config['embed_model'] = args.embed_model
    config['decode'] = args.decode
    config['embed_model_path'] = model_cards[config['embed_model']]
    config['device'] = torch.device("cuda")
    config['tokenizer'] = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    config['eos_token'] = config['tokenizer'].eos_token

    
    device = torch.device("cuda")

    batch_size = config['batch_size']


    
    ### qnli with beam search decoding
    sbert_roberta_large_pc_path  =  '../models_random/attacker_gpt2_qnli_sent_roberta.log'
    simcse_roberta_large_pc_path  = '../models_random/attacker_gpt2_qnli_simcse_roberta.log'
    simcse_bert_large_pc_path = '../models_random/attacker_gpt2_qnli_simcse_bert.log' 
    sentence_T5_large_pc_path = '../models_random/attacker_gpt2_qnli_sent_t5.log' 
    mpnet_pc_path = '../models_random/attacker_gpt2_qnli_mpnet.log'
    

    
    print('===mpnet===')
    get_val_ppl(mpnet_pc_path,batch_size,device,config)
    print('===sen_roberta===')
    get_val_ppl(sbert_roberta_large_pc_path,batch_size,device,config)
    print('===st5===')
    get_val_ppl(sentence_T5_large_pc_path,batch_size,device,config)
    print('===simcse_bert===')
    get_val_ppl(simcse_bert_large_pc_path,batch_size,device,config)
    print('===simcse_roberta===')
    get_val_ppl(simcse_roberta_large_pc_path,batch_size,device,config)


