from simcse_persona import get_persona_dict
from datasets import load_dataset
from pprint import pprint
import config
import json
import sys

abcd_path = config.abcd_path

'''
list of supported datasets:
['personachat'.'qnli','mnli','sst2','wmt16','multi_woz','abcd']
'''


def get_sent_list(config):
    dataset = config['dataset']
    data_type = config['data_type']
    if dataset == 'personachat':
        sent_list = get_personachat_data(data_type)
        return sent_list
    elif dataset == 'qnli':
        sent_list = get_qnli_data(data_type)
        return sent_list
    elif dataset == 'mnli':
        sent_list = get_mnli_data(data_type)
        return sent_list
    elif dataset == 'sst2':
        sent_list = get_sst2_data(data_type)
        return sent_list
    elif dataset == 'wmt16':
        sent_list = get_wmt16_data(data_type)
        return sent_list
    elif dataset == 'multi_woz':
        sent_list = get_multi_woz_data(data_type)
        return sent_list
    elif dataset == 'abcd':#abcd
        sent_list = get_abcd_data(data_type)
        return sent_list
    else:
        print('Name of dataset only supports: personachat or qnli')
        sys.exit(-1)


def get_personachat_data(data_type):
    sent_list = get_persona_dict(data_type=data_type)
    return sent_list


def get_qnli_data(data_type):
    if(data_type == 'dev'):
        data_type = 'validation'
    dataset = load_dataset('glue', 'qnli', cache_dir="data/", split=data_type)
    sentence_list = []
    for i, d in enumerate(dataset):
        sentence_list.append(d['question'])
        sentence_list.append(d['sentence'])
    return sentence_list

def get_mnli_data(data_type):
    if(data_type == 'test'):
        data_type = 'test_matched'
    if(data_type == 'dev'):
        data_type = 'validation_matched'
    dataset = load_dataset('glue', 'mnli', cache_dir="data/", split=data_type)
    sentence_list = []
    for i, d in enumerate(dataset):
        sentence_list.append(d['premise'])
        sentence_list.append(d['hypothesis'])
    return sentence_list

def get_sst2_data(data_type):
    if(data_type == 'dev'):
        data_type = 'validation'
    dataset = load_dataset('glue', 'sst2', cache_dir="data/", split=data_type)
    sentence_list = []
    for i, d in enumerate(dataset):
        sentence_list.append(d['sentence'])
    return sentence_list

## translation dataset
def get_wmt16_data(data_type):
    if(data_type == 'dev'):
        data_type = 'validation'
    dataset = load_dataset('wmt16', 'cs-en', cache_dir="data/", split=data_type)
    sentence_list = []
    for i, d in enumerate(dataset):

        #pprint(d)
        sentence_list.append(d['translation']['en'])
    return sentence_list

### multi_woz
## translation dataset
def get_multi_woz_data(data_type):
    if(data_type == 'dev'):
        data_type = 'validation'
    dataset = load_dataset('multi_woz_v22', 'v2.2', cache_dir="data/", split=data_type)
    sentence_list = []
    for i, d in enumerate(dataset):
        s = d['turns']['utterance']
        sentence_list.extend(s)
    return sentence_list
def get_abcd_data(data_type,path = abcd_path):
    
    #abcd_path = config.abcd_path
    with open(path, 'r') as f:
        dataset = json.load(f)
    dataset = dataset[data_type]
    #pprint(data[0])
    sentence_list = []
    for i, d in enumerate(dataset):
        s = d['original']
        for sent in s:
            if(sent[0] != 'action'):
                sentence_list.append(sent[1])
        # print('==='*50)
        # pprint(s)
        # pprint(sentence_list)
        # if i>=2:
        #     sys.exit(-1)
    return sentence_list


if __name__ == '__main__':

    config = {}
    config['dataset'] = 'abcd'
    config['data_type'] = 'test'
    sent_list = get_sent_list(config)
    pprint(sent_list[:10])
