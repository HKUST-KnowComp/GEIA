import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch, gc
torch.cuda.empty_cache()

import sys
sys.path.append('..')

import json

from nltk.corpus import stopwords


# tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
stopwords_list = stopwords.words('english')
for w in ['!',',','.','?','-s','-ly','</s>','s']:
    stopwords_list.append(w)
stopwords_set  = set(stopwords_list)
file_list = ['All_test_results/NN_wmt16_simcse_bert_threshold_0.70.label',
'All_test_results/NN_sst2_simcse_bert_threshold_0.75.label',
'All_test_results/NN_multi_woz_simcse_bert_threshold_0.50.label',
'All_test_results/NN_mnli_simcse_bert_threshold_0.50.label',
'All_test_results/NN_abcd_simcse_bert_threshold_0.45.label',
'All_test_results/test_RNN_wmt16_simcse_bert.label',
'All_test_results/test_RNN_sst2_simcse_bert.label',
'All_test_results/test_RNN_multi_woz_simcse_bert.label',
'All_test_results/test_RNN_mnli_simcse_bert.label',
'All_test_results/test_RNN_abcd_simcse_bert.label']


for file in file_list:
    
    gt_tokens = []
    pred_tokens = []
    with open(file) as f:
        print(file)  # <class '_io.TextIOWrapper'>  也就是文本IO类型
        result = json.load(f)


        #print(result)
        gt_count = 0
        pred_count = 0
        gt_non_stopword_count = 0
        pred_non_stopword_count = 0
        for i in result:
            # print(i)
            gt_list = [item.strip() for item in i['gt']]
            gt_count += len(gt_list)
            gt_set = set(gt_list)
            gt_non_stopword_count += len(gt_set - stopwords_set)
            # print("gt_tokens")
            # print(gt_tokens)
            pred_list = [item.strip() for item in i['pred']]
            pred_count += len(pred_list)
            pred_set = set(pred_list)
            pred_non_stopword_count += len(pred_set - stopwords_set)
            # print("gt static")
            # print(gt_non_stopword_count/gt_count)
            # print("pred static")
            # print(pred_non_stopword_count/pred_count)
            # print("pred_tokens")
            # print(pred_tokens)

        swr_gt = 1 - gt_non_stopword_count/gt_count
        print("gt_count:", end = '')
        print(gt_count)
        print("swr_gt:", end = '')
        print(swr_gt)

        swr_pred = 1 - pred_non_stopword_count/pred_count
        print("pred_count:", end = '')
        print(pred_count)
        print("swr_pred:", end = '')
        print(swr_pred)


file_list = ['All_test_results/attacker_gpt2_wmt16_simcse_bert_beam.log',
'All_test_results/attacker_gpt2_multi_woz_simcse_bert_beam.log',
'All_test_results/attacker_gpt2_abcd_simcse_bert_beam.log',
'All_test_results/attacker_gpt2_sst2_simcse_bert_beam.log',
'All_test_results/attacker_gpt2_mnli_simcse_bert_beam.log']

for file in file_list:
    gt_count = 0
    pred_count = 0
    gt_non_stopword_count = 0
    pred_non_stopword_count = 0
    gt_sentence_list = []
    pred_sentence_list = []

    with open(file) as f:
        print(file)
        # <class '_io.TextIOWrapper'>  也就是文本IO类型
        result = json.load(f)
        for sentence in result['gt']:
            # gt_sentence_list.append([tokenizer.encode(item) for item in sentence])
            gt_sentence_list = sentence.split()
            gt_count += len(gt_sentence_list)
            gt_set = set(gt_sentence_list)
            gt_non_stopword_count += len(gt_set - stopwords_set)
            # print("gt static")
            # print(gt_non_stopword_count / gt_count)
        for sentence in result['pred']:
            # pred_sentence_list.append([tokenizer.encode(item) for item in sentence])
            pred_sentence_list = sentence.split()
            pred_count += len(pred_sentence_list)
            pred_set = set(pred_sentence_list)
            pred_non_stopword_count += len(pred_set - stopwords_set)
            # print("pred static")
            # print(pred_non_stopword_count / pred_count)

        swr_gt = 1 - gt_non_stopword_count/gt_count

        print("gt_count:", end='')
        print(gt_count)
        print("swr_gt:", end='')
        print(swr_gt)

        swr_pred = 1 - pred_non_stopword_count/pred_count
        print("pred_count:", end = '')
        print(pred_count)
        print("swr_pred:", end = '')
        print(swr_pred)
        print()