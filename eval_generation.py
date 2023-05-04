import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#sys.path.append('..')
#from text_eval import punctuation_remove
import evaluate
import json
import nltk
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
from evaluate import load
from ppl import calucate_ppl
import editdistance
import string


model = SentenceTransformer('sentence-t5-xxl')

rouge = evaluate.load('rouge')
device = "cuda"
model = model.to(device)
model.eval()
# model_id = "gpt2-large"




#model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
#tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
perplexity = load("perplexity", module_type="metric")

#self training GPT-2 for PPL evaluation
# ppl_model = GPT2LMHeadModel.from_pretrained("gpt_large_persona")
# device = torch.device("cuda")
# ppl_model = ppl_model.to(device)
# ppl_model.eval()
# model = model.to(device)

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


def read_gpt(path):
    with open(path) as f:
        data = json.load(f)
    return data

def get_ppl(data,gpt_train= True):
    gt = data['gt']
    pred = data["pred"]
    if(gpt_train):
        ppl_gt,var_gt,ppl_pred,var_pred = calucate_ppl(gt,pred,ppl_model)
        print(f"GT: Validation Perplexity: {ppl_gt} Variance: {var_gt}")
        print(f"PRED: Validation Perplexity: {ppl_pred} Variance: {var_pred}")
    else:
        results_pred = perplexity.compute(model_id=model_id,
                                add_start_token=True,
                                predictions=pred)
        results_gt = perplexity.compute(model_id=model_id,
                                add_start_token=True,
                                predictions=gt)
        print(f'results_pred: {results_pred["mean_perplexity"]}')
        print(f'results_gt: {results_gt["mean_perplexity"]}')


def get_rouge(data):
    gt = data["gt"]
    pred = data["pred"]
    results = rouge.compute(predictions=pred,references=gt)
    print(results)

def get_bleu(data):
    gt = data['gt']
    pred = data["pred"]
    cands_list_bleu = [sentence.split() for sentence in pred] 
    refs_list_bleu = [[sentence.split()] for sentence in gt]
    bleu_score = nltk.translate.bleu_score.corpus_bleu(refs_list_bleu, cands_list_bleu) 
    bleu_score_1 = nltk.translate.bleu_score.corpus_bleu(refs_list_bleu, cands_list_bleu,weights=(1, 0, 0, 0)) 
    bleu_score_2 = nltk.translate.bleu_score.corpus_bleu(refs_list_bleu, cands_list_bleu,weights=(0, 1, 0, 0)) 
    print(f'bleu1 : {bleu_score_1}')
    print(f'bleu2 : {bleu_score_2}')
    print(f'bleu : {bleu_score}')


def batch(iterable, n):
    iterable=iter(iterable)
    while True:
        chunk=[]
        for i in range(n):
            try:
                chunk.append(next(iterable))
            except StopIteration:
                yield chunk
                return
        yield chunk


def embed_similarity(data,batch_size=16):
    gt = data['gt']
    pred = data["pred"]
    
    gt_batch = list(batch(gt, batch_size))
    pred_batch = list(batch(pred, batch_size))
    cosine_scores_all = []
    for i in range(len(gt_batch)):

        embeddings1 = model.encode(gt_batch[i], convert_to_tensor=True)
        embeddings2 = model.encode(pred_batch[i], convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        assert cosine_scores.size()[0] == cosine_scores.size()[1]
        score_list = [cosine_scores[k][k].item() for k in range(cosine_scores.size()[0])]
        cosine_scores_all.extend(score_list)
        #print(f'{i}-th: {np.mean(score_list).item()}')
    #cosine_scores_all = torch.stack(cosine_scores_all)
    avg_score = np.mean(cosine_scores_all)
    print(f'Avg embed similarity: {avg_score}')
    #print(f'Avg embed similarity running mean: {np.mean(running_mean)}')
    #sys.exit()

def get_edit_dist(data):
    gt = data['gt']
    pred = data["pred"]
    assert len(gt) == len(pred)
    edit_dist_list = []
    for i,d in enumerate(pred):
        gt_str = gt[i]
        pred_str = pred[i]
        dist = editdistance.distance(gt_str, pred_str)
        edit_dist_list.append(dist)
    ### now we return mean and median
    edit_dist_list = np.array(edit_dist_list)
    edit_median  = np.median(edit_dist_list)
    edit_mean = np.mean(edit_dist_list)
    print(f'edit_mean: {edit_mean}')
    print(f'edit_median: {edit_median}')
    return edit_mean,edit_median

def exact_match(data):
    gt = data['gt']
    pred = data["pred"]

    gt_remove = punctuation_remove(gt)       
    pred_remove = punctuation_remove(pred) 

    assert len(gt) == len(pred)
    count = 0 
    for i,d in enumerate(pred):
        gt_str = gt[i]
        pred_str = pred[i]
        if(gt_str == pred_str):
            count += 1
    ratio = count/len(gt)
    count = 0 
    for i,d in enumerate(pred):
        gt_str = gt_remove[i]
        pred_str = pred_remove[i]
        if(gt_str == pred_str):
            count += 1
    ratio_remove = count/len(gt_remove)
    print(f'exact_match ratio: {ratio}')

    print(f'exact_match ratio after removing punctuation: {ratio_remove}')

    return ratio

def remove_eos(data):
    gt = data['gt']
    pred = data["pred"]
    for i,s in enumerate(pred):
        pred[i] = s.replace('<|endoftext|>','')

def report_metrics(data):
    remove_eos(data)
    get_rouge(data)
    get_bleu(data)
    # get_ppl(data)     ### ppl please refer to ppl.py for ppl calculation
    exact_match(data)
    get_edit_dist(data)
    embed_similarity(data)


if __name__ == '__main__':


    abcd_path = '/home/hlibt/embed_rev/models_arr_feb/attacker_gpt2_abcd_simcse_bert_beam.log'
    mnli_path = '/home/hlibt/embed_rev/models_arr_feb/attacker_gpt2_mnli_simcse_bert_beam.log'
    woz_path = '/home/hlibt/embed_rev/models_arr_feb/attacker_gpt2_multi_woz_simcse_bert_beam.log'
    sst2_path = '/home/hlibt/embed_rev/models_arr_feb/attacker_gpt2_sst2_simcse_bert_beam.log'
    wmt_path = '/home/hlibt/embed_rev/models_arr_feb/attacker_gpt2_wmt16_simcse_bert_beam.log'

    path_list = [abcd_path,mnli_path,woz_path,sst2_path,wmt_path]
    for p in path_list:
        print(f'==={p}===')
        data = read_gpt(p)
        report_metrics(data)



