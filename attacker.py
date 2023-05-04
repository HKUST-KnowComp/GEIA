import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine

import json
import numpy as np
import pandas as pd
import argparse
import sys

from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AdamW,get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from attacker_models import SequenceCrossEntropyLoss
from sentence_transformers import SentenceTransformer
from simcse_persona import get_persona_dict
from attacker_evaluation_gpt import eval_on_batch
from datasets import load_dataset
from data_process import get_sent_list

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

def process_data(data,batch_size,device,config,need_porj=True):
    #model = SentenceTransformer('all-roberta-large-v1',device=device)   # dim 1024
    device_1 = torch.device("cuda:0")
    model = SentenceTransformer(config['embed_model_path'],device=device_1)   # dim 768
    dataset = personachat(data)
    dataloader = DataLoader(dataset=dataset, 
                              shuffle=True, 
                              batch_size=batch_size, 
                              collate_fn=dataset.collate)

    print('load data done')
    ### extra projection
    if need_porj:
        projection = linear_projection(in_num=768, out_num=1280).to(device)
    ### for attackers
    model_attacker = AutoModelForCausalLM.from_pretrained(config['model_dir'])
    tokenizer_attacker = AutoTokenizer.from_pretrained(config['model_dir'])
    criterion = SequenceCrossEntropyLoss()
    model_attacker.to(device)
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
        model.eval()
        for idx,batch_text in enumerate(dataloader):
            with torch.no_grad():           
                embeddings = model.encode(batch_text,convert_to_tensor = True).to(device)
                print(f'Embedding dim: {embeddings.size()}')

            ### attacker part, needs training
            if need_porj:
               embeddings = projection(embeddings)

            record_loss, perplexity = train_on_batch(batch_X=embeddings,batch_D=batch_text,model=model_attacker,tokenizer=tokenizer_attacker,criterion=criterion,device=device,train=True)
            optimizer.step()
            scheduler.step()
            # make sure no grad for GPT optimizer
            optimizer.zero_grad()
            print(f'Training: epoch {i} batch {idx} with loss: {record_loss} and PPL {perplexity} with size {embeddings.size()}')
            #sys.exit(-1)
        if need_porj:
            proj_path = 'models/' + 'projection_gpt2_large_' + config['dataset'] + '_' + config['embed_model']
            torch.save(projection.state_dict(), proj_path)
        save_path = 'models/' + 'attacker_gpt2_large_' + config['dataset'] + '_' + config['embed_model']
        model_attacker.save_pretrained(save_path)


### used for testing only
def process_data_test(data,batch_size,device,config,need_proj=True):
    #model = SentenceTransformer('all-roberta-large-v1',device=device)   # dim 1024
    #model = SentenceTransformer(config['embed_model_path'],device=device)   #  dim 768
    device_1 = torch.device("cuda:0")
    model = SentenceTransformer(config['embed_model_path'],device=device_1)   # dim 768
    if(config['decode'] == 'beam'):
        save_path = 'models/' + 'attacker_gpt2_large_' + config['dataset'] + '_' + config['embed_model']+'_beam'+'.log'
    else:
        save_path = 'models/' + 'attacker_gpt2_large_' + config['dataset'] + '_' + config['embed_model']+'.log'
    dataset = personachat(data)
    # no shuffle for testing data
    dataloader = DataLoader(dataset=dataset, 
                              shuffle=False, 
                              batch_size=batch_size, 
                              collate_fn=dataset.collate)

    print('load data done')
    if need_proj:
        proj_path = 'models/' + 'projection_gpt2_large_' + config['dataset'] + '_' + config['embed_model']
        projection = linear_projection(in_num=768, out_num=1280)
        projection.load_state_dict(torch.load(proj_path))
        projection.to(device)
        print('load projection done')
    else:
        print('no projection loaded')
    # setup on config for sentence generation   AutoModelForCausalLM
    attacker_path = 'models/' + 'attacker_gpt2_large_' + config['dataset'] + '_' + config['embed_model']
    config['model'] = AutoModelForCausalLM.from_pretrained(attacker_path).to(device)
    config['tokenizer'] = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')

    sent_dict = {}
    sent_dict['gt'] = []
    sent_dict['pred'] = []
    with torch.no_grad():  
        for idx,batch_text in enumerate(dataloader):

            embeddings = model.encode(batch_text,convert_to_tensor = True).to(device)
  
            if need_proj:
                embeddings = projection(embeddings)

            sent_list, gt_list = eval_on_batch(batch_X=embeddings,batch_D=batch_text,model=config['model'],tokenizer=config['tokenizer'],device=device,config=config)    
            print(f'testing {idx} batch done with {idx*batch_size} samples')
            sent_dict['pred'].extend(sent_list)
            sent_dict['gt'].extend(gt_list)
        
        with open(save_path, 'w') as f:
            json.dump(sent_dict, f,indent=4)

    return 0
        

### used for testing only
def process_data_test_simcse(data,batch_size,device,config,proj_dir=None,need_proj=False):
    tokenizer = AutoTokenizer.from_pretrained(config['embed_model_path'])  # dim 1024
    model = AutoModel.from_pretrained(config['embed_model_path']).to(device)
    #save_path = 'logs/attacker_gpt2_qnli_simcse_bert_large.log'
    if(config['decode'] == 'beam'):
        print('Using beam search decoding')
        save_path = 'logs/' + 'attacker_gpt2_large_' + config['dataset'] + '_' + config['embed_model']+'_beam'+'.log'
    else:
        save_path = 'logs/' + 'attacker_gpt2_large_' + config['dataset'] + '_' + config['embed_model']+'.log'
    dataset = personachat(data)
    # no shuffle for testing data
    dataloader = DataLoader(dataset=dataset, 
                              shuffle=False, 
                              batch_size=batch_size, 
                              collate_fn=dataset.collate)

    print('load data done')
    if need_proj:
        projection = linear_projection(in_num=768)
        projection.load_state_dict(torch.load(proj_dir))
        projection.to(device)
        print('load projection done')
    else:
        print('no projection loaded')
    # setup on config for sentence generation   AutoModelForCausalLM
    attacker_path = 'models/' + 'attacker_gpt2_large_' + config['dataset'] + '_' + config['embed_model']
    config['model'] = AutoModelForCausalLM.from_pretrained(attacker_path).to(device)
    config['tokenizer'] = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')

    sent_dict = {}
    sent_dict['gt'] = []
    sent_dict['pred'] = []
    with torch.no_grad():  
        for idx,batch_text in enumerate(dataloader):
            inputs = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt").to(device)
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output  
            if need_proj:
                embeddings = projection(embeddings)
            #sent_list, gt_list = eval_on_batch(batch_X=embeddings,batch_D=batch_text,model=config['model'],tokenizer=config['tokenizer'],device=device,config=config)    
            sent_list, gt_list = eval_on_batch(batch_X=embeddings,batch_D=batch_text,model=config['model'],tokenizer=config['tokenizer'],device=device,config=config) 
            print(f'testing {idx} batch done with {idx*batch_size} samples')
            sent_dict['pred'].extend(sent_list)
            sent_dict['gt'].extend(gt_list)
        with open(save_path, 'w') as f:
            json.dump(sent_dict, f,indent=4)

    return 0


def train_on_batch(batch_X,batch_D,model,tokenizer,criterion,device,train=True):
    padding_token_id = tokenizer.encode(tokenizer.eos_token)[0]
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(batch_D, return_tensors='pt', padding='max_length', truncation=True, max_length=40)
    #dial_tokens = [tokenizer.encode(item) + turn_ending for item in batch_D]
    #print(inputs)
    input_ids = inputs['input_ids'].to(device) # tensors of input ids
    labels = input_ids.clone()
    #print(input_ids.size())
    # embed the input ids using GPT-2 embedding
    input_emb = model.transformer.wte(input_ids)
    # add extra dim to cat together
    batch_X = batch_X.to(device)
    batch_X_unsqueeze = torch.unsqueeze(batch_X, 1)
    inputs_embeds = torch.cat((batch_X_unsqueeze,input_emb),dim=1)   #[batch,max_length+1,emb_dim (1024)]
    past = None
    # need to move to device later
    inputs_embeds = inputs_embeds

    #logits, past = model(inputs_embeds=inputs_embeds,past = past)
    logits, past = model(inputs_embeds=inputs_embeds,past_key_values  = past,return_dict=False)
    logits = logits[:, :-1].contiguous()
    target = labels.contiguous()
    target_mask = torch.ones_like(target).float()
    loss = criterion(logits, target, target_mask, label_smoothing=0.02, reduce="batch")   

    record_loss = loss.item()
    perplexity = np.exp(record_loss)
    if train:
        loss.backward()

    return record_loss, perplexity






if __name__ == '__main__':
    model_cards ={}
    model_cards['sent_t5_large'] = 'sentence-t5-large'
    model_cards['sent_t5_base'] = 'sentence-t5-base'
    model_cards['sent_t5_xl'] = 'sentence-t5-xl'
    model_cards['sent_t5_xxl'] = 'sentence-t5-xxl'
    model_cards['mpnet'] = 'all-mpnet-base-v1'
    model_cards['sent_roberta'] = 'all-roberta-large-v1'
    model_cards['simcse_bert'] = 'princeton-nlp/sup-simcse-bert-large-uncased'
    model_cards['simcse_roberta'] = 'princeton-nlp/sup-simcse-roberta-large'
    parser = argparse.ArgumentParser(description='Training external NN as baselines')
    parser.add_argument('--model_dir', type=str, default='microsoft/DialoGPT-large', help='Dir of your model')
    parser.add_argument('--num_epochs', type=int, default=10, help='Training epoches.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch_size #.')
    parser.add_argument('--dataset', type=str, default='personachat', help='Name of dataset: personachat or qnli')
    #parser.add_argument('--dataset', type=str, default='qnli', help='Name of dataset: personachat or qnli')
    #parser.add_argument('--data_type', type=str, default='train', help='train/test')
    parser.add_argument('--data_type', type=str, default='test', help='train/test')
    parser.add_argument('--embed_model', type=str, default='sent_t5_base', help='Name of embedding model: mpnet/sent_roberta/simcse_bert/simcse_roberta/sent_t5')
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
    config['tokenizer'] = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
    config['eos_token'] = config['tokenizer'].eos_token

    
    device = torch.device("cuda:0")
    #device = torch.device("cpu")
    batch_size = config['batch_size']

    sent_list = get_sent_list(config)
    
    ##### for training
    if(config['data_type'] == 'train'):
        process_data(sent_list,batch_size,device,config)
    elif(config['data_type'] == 'test'):
        if('simcse' in config['embed_model']):
            process_data_test_simcse(sent_list,batch_size,device,config,proj_dir=None,need_proj=False)
        else:
            process_data_test(sent_list,batch_size,device,config,need_proj=True)

