import os
import config
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import datasets
from datasets import load_dataset
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import json
from pprint import pprint
class BookCorpus(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        text = self.data[index]

        return  text
        
    def collate(self, unpacked_data):
        return unpacked_data

def process_data(data,batch_size,device):
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-large-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-large-uncased").to(device)
    dataset = BookCorpus(data, tokenizer)
    dataloader = DataLoader(dataset=dataset, 
                              shuffle=True, 
                              batch_size=batch_size, 
                              collate_fn=dataset.collate)
    data_dict = {}
    data_dict['text'] = []
    data_dict['embedding'] = []
    with torch.no_grad():                          
        for idx,batch_text in enumerate(dataloader):
            inputs = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt").to(device)
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            embeddings = embeddings.detach().cpu()
            data_dict['text'].extend(batch_text)
            data_dict['embedding'].extend(embeddings)
            print(f'{idx} batch done with {idx*batch_size} samples')
    return data_dict

def get_processed_persona(kind,processed_persona_path,require_label = True):
    #processed_persona_path = config.processed_persona
    if(require_label):
        path = processed_persona_path + '/%s_merged_shuffle.txt' % kind
    else:
        path = processed_persona_path + '/%s.txt' % kind
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def process_persona(data):
    '''
    get only list of texts for batch training for 
    '''
    sentence_list = []
    for i,dict_i in enumerate(data):
        conv = dict_i['conv']
        sentence_list.extend(conv)

    return sentence_list


def get_persona_dict(data_type):
    processed_persona_path = config.processed_persona
    data = get_processed_persona(data_type,processed_persona_path)
    processed_data = process_persona(data)
    

    return processed_data
# Import our models. The package will take care of downloading the models automatically
if __name__ == '__main__':
    processed_persona_path = config.processed_persona
    #train_data = get_processed_persona('train')
    #val_data = get_processed_persona('dev')
    test_data = get_processed_persona('test')
    processed_test = process_persona(test_data)

    device = torch.device("cuda")
    batch_size = 32

    val_dict = process_data(processed_test,batch_size,device)
    torch.save(val_dict, '/data/hlibt/gradient_leakage/pytorch/data/personachat_processed/hidden_test_simcse.pt')



