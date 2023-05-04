import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn import metrics
import numpy as np
import argparse
import torch
from transformers import AutoModel, AutoTokenizer
import json
from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import cosine
from simcse import SimCSE
import logging
import logging.handlers
from nltk.tokenize import word_tokenize
import string
import re

logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
f_handler = logging.FileHandler('models_arr_feb/decoder_beam.log')
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s -  %(message)s"))
logger.addHandler(f_handler)

def vectorize(sent_list,tokenizer):
    turn_ending = tokenizer.encode(tokenizer.eos_token)
    token_num = len(tokenizer)
    dial_tokens = [tokenizer.encode(item) + turn_ending for item in sent_list]
    dial_tokens_np = np.array(dial_tokens)
    input_labels = []
    for i in dial_tokens_np:
        temp_i = np.zeros(token_num)
        temp_i[i] = 1
        input_labels.append(temp_i)
    input_labels = np.array(input_labels)


    return input_labels





def report_score(y_true,y_pred):
    # micro result should be reported
    precision = metrics.precision_score(y_true, y_pred, average='micro')
    recall = metrics.recall_score(y_true, y_pred, average='micro')
    f1 = metrics.f1_score(y_true, y_pred, average='micro')
    logger.info(f"micro precision_score on token level: {str(precision)}")
    logger.info(f"micro recall_score on token level: {str(recall)}")
    logger.info(f"micro f1_score on token level: {str(f1)}")



def embed_simcse(y_true,y_pred):
    model = SimCSE("princeton-nlp/sup-simcse-roberta-large",device='cuda')
    similarities = model.similarity(y_true, y_pred) # numpy array of N*N
    pair_scores = similarities.diagonal()
    for i,score in enumerate(pair_scores):
        assert pair_scores[i] == similarities[i][i]
    avg_score = np.mean(pair_scores)
    logger.info(f'Evaluation on simcse-roberta with similarity score {avg_score}')


def embed_sbert(y_true,y_pred):
    model = SentenceTransformer('all-roberta-large-v1',device='cuda')       # has dim 768
    embeddings_true = model.encode(y_true,convert_to_tensor = True)
    embeddings_pred = model.encode(y_pred,convert_to_tensor = True)
    cosine_scores = util.cos_sim(embeddings_true, embeddings_pred)
    pair_scores = torch.diagonal(cosine_scores, 0)
    for i,score in enumerate(pair_scores):
        assert pair_scores[i] == cosine_scores[i][i]
    avg_score = torch.mean(pair_scores)
    logger.info(f'Evaluation on Sentence-bert with similarity score {avg_score}')


    return avg_score


def report_embedding_similarity(y_true,y_pred):
    embed_sbert(y_true,y_pred)
    embed_simcse(y_true,y_pred)


def main(log_path):
    with open(log_path, 'r') as f:
        sent_dict = json.load(f)
    y_true = sent_dict['gt']     # list of sentences
    y_pred = sent_dict['pred']   # list of sentences   
    report_embedding_similarity(y_true,y_pred)


'''
26/02 newly appended functions
'''
# remove punctuation from list of sentences 
def punctuation_remove(sent_list):
    removed_list = []
    for sent in sent_list:
        word_list = []
        for word in sent.split():
            word_strip = word.strip(string.punctuation)
            if word_strip:  # cases for not empty string
                word_list.append(word_strip)
        removed_sent = ' '.join(word_list)
        removed_list.append(removed_sent)
    return removed_list

# remove space before punctuation from list of sentences 
def space_remove(sent_list):
    removed_list = []
    for sent in sent_list:
        sent_remove = re.sub(r'\s([?.!"](?:\s|$))', r'\1', sent)
        removed_list.append(sent_remove)
    return removed_list

def metrics_word_level(token_true,token_pred):
    len_pred = len(token_pred)
    len_ture = len(token_true)
    recover_pred = 0
    recover_true = 0
    for p in token_pred:
        if p in token_true:
            recover_pred += 1
    for t in token_true:
        if t in token_pred:
            recover_true += 1
    ### return for precision recall calculation        
    return len_pred,recover_pred,len_ture,recover_true
            
    
def word_level_metrics(y_true,y_pred):
    assert len(y_true) == len(y_pred)
    recover_pred_all = 0
    recover_true_all = 0
    len_pred_all = 0
    len_ture_all = 0
    for i in range(len(y_true)):
        sent_true = y_true[i]
        sent_pred = y_pred[i]
        token_true = word_tokenize(sent_true)
        token_pred = word_tokenize(sent_pred)
        len_pred,recover_pred,len_ture,recover_true = metrics_word_level(token_true,token_pred)
        len_pred_all += len_pred
        recover_pred_all += recover_pred
        len_ture_all += len_ture
        recover_true_all += recover_true
        
        
    ### precision and recall are based on micro (but not exactly)
    precision = recover_pred_all/len_pred_all
    recall = recover_true_all/len_ture_all
    f1 = 2*precision*recall/(precision+recall)
    return precision,recall,f1

def remove_eos(sent_list):
    for i,s in enumerate(sent_list):
        sent_list[i] = s.replace('<|endoftext|>','')

def metric_token(log_path):
    with open(log_path, 'r') as f:
        sent_dict = json.load(f)
    y_true = sent_dict['gt']     # list of sentences
    y_pred = sent_dict['pred']   # list of sentences   
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    y_true_token = vectorize(y_true,tokenizer)
    y_pred_token = vectorize(y_pred,tokenizer)

    ### token-level metrics are reported
    report_score(y_true_token,y_pred_token)
    remove_eos(y_pred)           # make sure to remove <eos>
    ### scores for word level
    y_true_removed_p = punctuation_remove(y_true)       
    y_pred_removed_p = punctuation_remove(y_pred)  
    y_true_removed_s = space_remove(y_true)       
    y_pred_removed_s = space_remove(y_pred)  
    precision,recall,f1 = word_level_metrics(y_true_removed_s,y_pred_removed_s)
    logger.info(f'word level precision: {str(precision)}')
    logger.info(f'word level recall: {str(recall)}')
    logger.info(f'word level f1: {str(f1)}')
    
    precision,recall,f1 = word_level_metrics(y_true_removed_p,y_pred_removed_p)
    logger.info(f'word level precision without punctuation: {str(precision)}')
    logger.info(f'word level recall without punctuation: {str(recall)}')
    logger.info(f'word level f1 without punctuation: {str(f1)}')


if __name__ == '__main__':
    '''
    ### PC with sampling and randomly initialized GPT-2
    sbert_roberta_large_pc_path  =  'models_random/attacker_gpt2_personachat_sent_roberta.log'
    simcse_roberta_large_pc_path  = 'models_random/attacker_gpt2_personachat_simcse_roberta.log'
    simcse_bert_large_pc_path = 'models_random/attacker_gpt2_personachat_simcse_bert.log' 
    sentence_T5_large_pc_path = 'models_random/attacker_gpt2_personachat_sent_t5.log' 
    mpnet_pc_path = 'models_random/attacker_gpt2_personachat_mpnet.log'
    logger.info(f'====={sbert_roberta_large_pc_path}=====')
    metric_token(sbert_roberta_large_pc_path)
    logger.info(f'====={simcse_roberta_large_pc_path}=====')
    metric_token(simcse_roberta_large_pc_path)
    logger.info(f'====={simcse_bert_large_pc_path}=====')
    metric_token(simcse_bert_large_pc_path)    
    logger.info(f'====={sentence_T5_large_pc_path}=====')
    metric_token(sentence_T5_large_pc_path)
    logger.info(f'====={mpnet_pc_path}=====')
    metric_token(mpnet_pc_path)
    '''

    abcd_path = '/home/hlibt/embed_rev/models_arr_feb/attacker_gpt2_abcd_simcse_bert_beam.log'
    mnli_path = '/home/hlibt/embed_rev/models_arr_feb/attacker_gpt2_mnli_simcse_bert_beam.log'
    woz_path = '/home/hlibt/embed_rev/models_arr_feb/attacker_gpt2_multi_woz_simcse_bert_beam.log'
    sst2_path = '/home/hlibt/embed_rev/models_arr_feb/attacker_gpt2_sst2_simcse_bert_beam.log'
    wmt_path = '/home/hlibt/embed_rev/models_arr_feb/attacker_gpt2_wmt16_simcse_bert_beam.log'

    path_list = [abcd_path,mnli_path,woz_path,sst2_path,wmt_path]
    for p in path_list:
        logger.info(f'====={p}=====')
        metric_token(p)