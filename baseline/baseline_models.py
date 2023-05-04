import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import pipeline
from transformers import AdamW, get_linear_schedule_with_warmup

'''
eval = False retrun logits for stable  nn.BCEWithLogitsLoss()
eval = True reutrun 0~1 probs
'''


class baseline_NN(nn.Module):
    def __init__(self, out_num, in_num=1024):
        super(baseline_NN, self).__init__()
        self.fc1 = nn.Linear(in_num, out_num)
        self.embed_dim = in_num
        # self.act = F.softmax()

    def forward(self, x, eval=False):
        # x should be of shape (?,1024) according to embedding output
        assert (x.dim() == 2)
        # print(f'x.size(): {x.size()}.    embed_dim:{self.embed_dim}')
        assert (x.size()[1] == self.embed_dim)
        out = self.fc1(x)
        if (eval):
            m = nn.Sigmoid()
            out = m(out)
        return out


hidden_size = 512
num_layers = 1


class baseline_RNN(nn.Module):
    def __init__(self, out_num, in_num=1024):
        super(baseline_RNN, self).__init__()
        self.rnn = nn.GRUCell(
            1024,
            hidden_size,
            # num_layers,
            # batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, out_num)
        self.fc2 = nn.Linear(768, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(2048, 512)
        self.m = nn.Softmax(dim=1)
        '''
        Extract representations from GPT-2's tokens as RNN's input

        '''
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        # self.gpt2 = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')  ### seems huggingface has bug for feature extraction with this one, return 50257 as embedding (wrong)
        self.gpt2 = AutoModel.from_pretrained("microsoft/DialoGPT-medium")
        # If you find errors related to padding token, uncomment these 2 lines
        ####self.gpt2_tokenizer.add_special_tokens({'pad_token': self.gpt2_tokenizer.eos_token})
        ####self.gpt2.resize_token_embeddings(len(self.gpt2_tokenizer))
        self.pipe = pipeline('feature-extraction', model=self.gpt2, tokenizer=self.gpt2_tokenizer, device=0)

    '''
    Obtain token embedding from GPT-2 model we use for IEI. (Make fair comparison with same input embedding)
    '''

    def get_token_embedding(self, token_id):
        token = self.gpt2_tokenizer.decode(token_id)
        embedding = self.pipe(token)  # get embedding, [1,1,1024]
        embedding = torch.tensor(embedding).cuda()
        embedding = embedding[0, 0, :]
        return embedding

    '''
    work flow:
        0. initialize GRU cell(structure and parameters)
        1. embedding is a embedding sentence, check its dimension
        2. for loop process x, in each loop:
            2.1 add x in to GRU cell
            2.2 get output type(?)
            2.3 add fc1 to output change its dimension to 1024 or 768
            2.4 add a softmax and do argmax
            2.4 use its token id and h_x as the next input
            2.5 use cross_entropy loss to to update this model (output cross_entropy)  
    '''

    def forward(self, sentences_embedding, hx, batch_label, eval=False):
        '''
        optimizer = AdamW(self.parameters(),
                          lr=3e-5,
                          eps=1e-06)

        '''
        # zero_tensor = torch.zeros(64, 50257).cuda()

        output = []
        loss_list = []

        if sentences_embedding.size()[1] == 768:
            sentences_embedding = self.fc2(sentences_embedding)

        predicted_token_index_batch_list = []
        out_exclued_predicted = None
        for i in range(10):
            hx = self.rnn(sentences_embedding, hx)
            output.append(hx)
            out = self.fc1(output[i])
            out_clone = out.clone().detach()
            out_clone = self.m(out_clone)
            batch_label_clone = batch_label.clone().detach()
            if (predicted_token_index_batch_list != []):
                predicted_token_index_batch_tensor = torch.stack(predicted_token_index_batch_list, 1).cuda()

                if predicted_token_index_batch_tensor != None:
                    out_exclued_predicted = out_clone.scatter_(1, predicted_token_index_batch_tensor, value=0)

                    batch_label_clone = batch_label_clone.scatter_(1, predicted_token_index_batch_tensor, value=0)
            # torch.Size([])
            if out_exclued_predicted is not None:
                token_index_batch = torch.argmax(out_exclued_predicted, dim=1).cuda()
            else:
                token_index_batch = torch.argmax(out, dim=1).cuda()
            predicted_token_index_batch_list.append(token_index_batch.detach())
            '''
            token_embedding_batch_list = []
            for index in range(64):
                token_embedding_batch_list.append(self.get_token_embedding(token_index_batch[index]))
            '''
            '''
            predicted_embedding = torch.stack(token_embedding_batch_list, 0).cuda()
            predicted_embedding = self.fc3(predicted_embedding)
            prev_embedding = self.fc4(embedding)
            embedding = torch.cat((sentences_embedding, predicted_embedding, prev_embedding), 1)
            '''

            if eval == False:
                # loss = F.cross_entropy(out, batch_label)
                pred_out = out.softmax(dim=-1).log().double()
                loss = F.kl_div(pred_out, batch_label_clone.softmax(dim=-1), reduction='sum')
                target_index = torch.nonzero(batch_label_clone).cpu().numpy().tolist()

                loss_list.append(loss)

        if eval == False:
            batch_final_loss = loss_list[0] + loss_list[1] + loss_list[2] + loss_list[3] + loss_list[4] + \
                               loss_list[5] + loss_list[6] + loss_list[7] + loss_list[8] + loss_list[9]

            print("batch_final_loss:", end='')
            print(batch_final_loss)

            return batch_final_loss
        else:
            predicted_token_index_batch_tensor = torch.stack(predicted_token_index_batch_list, 1).cuda()
            target_index = torch.nonzero(batch_label).cpu().numpy().tolist()
            return predicted_token_index_batch_tensor
