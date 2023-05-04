import sys
sys.path.append('..')
from simcse_persona import get_persona_dict
import baseline_models
from datasets import load_dataset

import argparse

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

def get_personachat_data(data_type):
    sent_list = get_persona_dict(data_type=data_type)
    return sent_list

def get_qnli_data(data_type):
    dataset = load_dataset('glue', 'qnli', cache_dir="/home/hlibt/embed_rev/data", split=data_type)
    sentence_list = []
    for i, d in enumerate(dataset):
        sentence_list.append(d['question'])
        sentence_list.append(d['sentence'])
    return sentence_list

def data_stat():
    sent_list_train = get_personachat_data('train')
    sent_list_dev = get_personachat_data('dev')
    sent_list_test = get_personachat_data('test')
    sent_list = []
    sent_list.extend(sent_list_train)
    sent_list.extend(sent_list_dev)
    sent_list.extend(sent_list_test)
    print('===PersonaChat stat===')
    print('===Train===')
    print_stat(sent_list_train)
    print('===Dev===')
    print_stat(sent_list_dev)
    print('===Test===')
    print_stat(sent_list_test)
    print_stat(sent_list)
    sent_list = get_qnli_data('train')
    sent_list.extend(get_qnli_data('test'))
    print('===QNLI stat===')
    print('===Train===')
    print_stat(get_qnli_data('train'))
    print('===Test===')
    print_stat(get_qnli_data('test'))
    print_stat(sent_list)


def print_stat(sent_list):
    sent_len = len(sent_list)
    sent_len_list = [len(i.split()) for i in sent_list]
    avg_len = sum(sent_len_list) / sent_len
    print(f'Number of sentences: {sent_len}')
    print(f'Avg_length of sentences: {avg_len}')

if __name__ == '__main__':
    data_stat()
