import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import AdamW
import sys
import argparse
import attacker_models
from attacker_models import read_pt, Dataset,Dataset_trans,SequenceCrossEntropyLoss
#from sentence_revocer_transformer import train_on_batch
import json
from decode_beam_search import beam_decode_sentence
import decode_beam_search_opt
#from bookcorpus_train import str2bool,BookCorpus_Dataset
def get_dataloader(config):
    data_type= config['data_type']
    batch_size = config['batch_size']
    if config['use_trans']:
        if config['p_simcse_flag']:
            ### path to pt checkpoint
            data = torch.load('/data/hlibt/gradient_leakage/pytorch/data/personachat_processed/hidden_test_sbert.pt')
            dataset = BookCorpus_Dataset(data)
        else:
            X,Y,A,D = read_pt(data_type,use_trans=config['use_trans'])
            dataset = Dataset_trans(X,Y,A,D)
    else:
        X,Y,A = read_pt(data_type,use_trans=config['use_trans'])
        dataset = Dataset(X,Y,A)
    dataloader = DataLoader(dataset=dataset, 
                              shuffle=True, 
                              batch_size=batch_size)
    return dataloader

def get_model(config):

    model_dir = config['model_dir']
    model_type = config['model_type']

    if model_type == '1layerNN':
        model = attacker_models.model_inv_nn(out_num=config['token_num'])
        model.load_state_dict(torch.load(model_dir))
        model.to(config['device']) 
        criterion = nn.BCEWithLogitsLoss()


    else:
        print('No proper model loaded')
        sys.exit(-1)

    return model,criterion


def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
    """
    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values, 
                             torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), 
                             logits)
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    
    return logits

def generate_sentence(config,hidden_X):
    temperature = 0.9
    top_k = -1
    top_p = 0.9
    sent = []
    prev_input = None
    past = None
    model = config['model']
    tokenizer =config['tokenizer']
    #eos = [tokenizer.encoder["<|endoftext|>"]]
    eos = tokenizer.encode("<|endoftext|>")
    hidden_X_unsqueeze = torch.unsqueeze(hidden_X, 0)
    hidden_X_unsqueeze = torch.unsqueeze(hidden_X_unsqueeze, 0)  #[1,1,embed_dim]
    logits, past = model(inputs_embeds=hidden_X_unsqueeze,past_key_values  = past,return_dict=False)
    logits = logits[:, -1, :] / temperature
    logits = top_filtering(logits, top_k=top_k, top_p=top_p)

    probs = torch.softmax(logits, dim=-1)

    prev_input = torch.multinomial(probs, num_samples=1)
    prev_word = prev_input.item()
    sent.append(prev_word)

    for i in range(50):
        #logits, past = model(prev_input, past=past)
        logits, past = model(prev_input,past_key_values  = past,return_dict=False)
        logits = logits[:, -1, :] / temperature
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)

        probs = torch.softmax(logits, dim=-1)

        prev_input = torch.multinomial(probs, num_samples=1)
        prev_word = prev_input.item()

        if prev_word == eos[0]:
            break
        sent.append(prev_word)
    
    output = tokenizer.decode(sent)

    return output


def eval(dataloader,config):
    model = config['model']
    tokenizer = config['tokenizer']
    device = config['device']
    model.to(device)
    criterion = SequenceCrossEntropyLoss()
    save_path = config['save_path']
    sent_dict = {}
    sent_dict['gt'] = []
    sent_dict['pred'] = []
    with torch.no_grad():
        for idx,(batch_X,batch_D) in enumerate(dataloader):
            batch_D = list(batch_D)
            sent_list, gt_list = eval_on_batch(batch_X,batch_D,model,tokenizer,device,config)    
            sent_dict['pred'].extend(sent_list)
            sent_dict['gt'].extend(gt_list)

        with open(save_path, 'w') as f:
            json.dump(sent_dict, f,indent=4)


def eval_on_batch(batch_X,batch_D,model,tokenizer,device,config):
    decode_method = config['decode']
    padding_token_id = tokenizer.encode(tokenizer.eos_token)[0]
    if(not config['use_opt']):
        tokenizer.pad_token = tokenizer.eos_token
    batch_X = batch_X.to(device)
    print(f'batch_X:{batch_X.size()}')
    sent_list = []
    gt_list = batch_D
    for i,hidden in enumerate(batch_X):
        inputs_embeds = hidden
        if(decode_method == 'beam'):
            #print('Using beam search decoding')
            if(config['use_opt']):
                sentence = decode_beam_search_opt.beam_decode_sentence(hidden_X=inputs_embeds, config = config,num_generate=1, beam_size = 5)
            else:
                sentence = beam_decode_sentence(hidden_X=inputs_embeds, config = config,num_generate=1, beam_size = 5)

            #print(sentence)
            sentence = sentence[0]
        else:
            sentence = generate_sentence(config,hidden_X=inputs_embeds)
        sent_list.append(sentence)



    return sent_list, gt_list


def main():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--model_dir', type=str, default='models/attacker_gpt2_persona_sbert', help='Dir of your model')
    parser.add_argument('--model_type', type=str, default='gpt-2', help='Type of the attacker model.')
    parser.add_argument('--data_type', type=str, default='test', help='Type of the processed data.')
    parser.add_argument('--save_path', type=str, default='logs/attacker_gpt2_p_sbert.log', help='Type of the processed data.')
    parser.add_argument('--num_epochs', type=int, default=1, help='Type of the processed data.')
    parser.add_argument('--p_simcse_flag', type=str2bool, default=True, help='Type of the processed data.')

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    token_num = len(tokenizer)
    config = {}
    config['model_dir'] = args.model_dir
    config['model_type'] = args.model_type
    config['num_epochs'] = args.num_epochs
    config['save_path'] = args.save_path
    config['p_simcse_flag'] = args.p_simcse_flag
    config['batch_size'] = attacker_models.batch_size
    config['data_type'] = args.data_type
    config['device']  = torch.device("cuda")

    config['use_trans'] = True
    config['model'] = AutoModelForCausalLM.from_pretrained(config['model_dir'])
    config['tokenizer'] = tokenizer
    config['token_num'] = token_num
    print('get_model done')
    dataloader = get_dataloader(config)
    print('get_dataloader done')
    eval(dataloader,config)
if __name__ == '__main__':
    main()