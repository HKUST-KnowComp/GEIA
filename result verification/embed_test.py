import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM,AutoTokenizer,AutoModel
from transformers import pipeline




class baseline_RNN(nn.Module):
    def __init__(self, out_num, num_layers, in_num=1024, hidden_size = 2048):
        super(baseline_RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=in_num,
            hidden_size = 4096,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, out_num)


        '''
        Extract representations from GPT-2's tokens as RNN's input
        
        '''
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        #self.gpt2 = AutoModelForCausalLM.from_pretrained('gpt2')
        self.gpt2 = AutoModel.from_pretrained("microsoft/DialoGPT-medium")
        # If you find errors related to padding token, uncomment these 2 lines
        self.gpt2_tokenizer.add_special_tokens({'pad_token': self.gpt2_tokenizer.eos_token})
        self.gpt2.resize_token_embeddings(len(self.gpt2_tokenizer))
        self.pipe = pipeline('feature-extraction', model=self.gpt2, tokenizer=self.gpt2_tokenizer,device=0)


    '''
    Obtain token embedding from GPT-2 model we use for IEI. (Make fair comparison with same input embedding)
    '''
    def get_token_embedding(self,token_id):
        token = self.gpt2_tokenizer.decode(token_id)
        embedding = self.pipe(token)  # get embedding, [1,1,1024]
        embedding = torch.tensor(embedding).cuda()
        print(embedding.size())
        embedding = embedding[0,0,:]
        print(embedding.size())
        return  embedding


if __name__ == '__main__':
    model = baseline_RNN(out_num = 50257, num_layers= 2).cuda()

    token_id = model.gpt2_tokenizer.encode('this is a test')
    model.get_token_embedding(token_id)