'''
Source code modified from https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
implementation of beam search on GPT-2's logits
'''

import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
import sys


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate: for opt here, reuse past can occur errors, here we just use prev of input embeddings
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __lt__(self, x):
        if(self.eval() < x.eval()):
            return True

        else:
            return False



def beam_decode_sentence(hidden_X, config,num_generate=1, beam_size = 5, batch_size = 1):
    '''
    generate a sentence based on beam search
    :param hidden_X: input embeddings
    :param model: GPT-2 model
    :param tokenizer: GPT-2 tokenizer
    :return: decoded_batch
    '''
    #SOS_token = tokenizer.encode("<|endoftext|>")
    beam_width = beam_size
    topk = num_generate  # how many sentence do you want to generate

    past = None
    model = config['model']
    tokenizer =config['tokenizer']
    #eos = [tokenizer.encoder["<|endoftext|>"]]
    eos = [tokenizer.encode(tokenizer.eos_token)]
    EOS_token = eos
    hidden_X_unsqueeze = torch.unsqueeze(hidden_X, 0)
    hidden_X_unsqueeze = torch.unsqueeze(hidden_X_unsqueeze, 0)  #[1,1,embed_dim] [batch_size, seq_len, emb_dim]

    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(batch_size):

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length              hiddenstate, previousNode, wordId, logProb, length
        #node = BeamSearchNode(past, None, torch.tensor([[220]]).cuda(), 0, 1)                    # 220 refers to single space ' '
        # if(config['use_opt']):
        #     node = BeamSearchNode(past, None, torch.tensor([[2]]).cuda(), 0, 1)                    # 2 refers to '</s>' on opt
        # else:
        #     node = BeamSearchNode(past, None, torch.tensor([[220]]).cuda(), 0, 1)                    # 220 refers to single space ' ' on GPT
        node = BeamSearchNode(hidden_X_unsqueeze, None, torch.tensor([[2]]).cuda(), 0, 1)                    # 2 refers to '</s>' on opt
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        for text_len in range(50):
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            try:
                score, n = nodes.get()
            except:
                print('Cannot get nodes')
                while not nodes.empty():
                    next_item = nodes.get()
                    print(next_item)
            prev_input = n.wordid
            past = n.h

            if n.wordid.item() == EOS_token[0] and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                #print(f'EOS found')
                if len(endnodes) >= number_required:
                    #print('break')
                    break
                    
                else:
                    print('continue')
                    continue

            # decode for one step using decoder
            #decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

            output = model(inputs_embeds=past,past_key_values  = None,return_dict=True)
            logits = output.logits

            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1) 
            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(probs, beam_width)
            nextnodes = []
            
            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()
                #### hiddenstate, previousNode, wordId, logProb, length
                input_emb = model.model.decoder.embed_tokens(decoded_t)
                new_past = torch.cat((past,input_emb),dim=1)
                node = BeamSearchNode(new_past, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                try:
                    nodes.put((score, nn))
                except:
                    print('Cannot put nodes')
                    print(score)
                    print(nn)
                # increase qsize
            qsize += len(nextnodes) - 1
        # for loop ends here
        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        text = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid.item())
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid.item())

            utterance = utterance[::-1]
            utterances.append(utterance)
            decode_process = tokenizer.decode(utterance[1:-1],skip_special_tokens=True)
            text.append(decode_process)
        decoded_batch.append(utterances)

    return text


def greedy_decode(decoder_hidden, encoder_outputs, target_tensor):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    batch_size, seq_len = target_tensor.size()
    decoded_batch = torch.zeros((batch_size, MAX_LENGTH))
    decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)], device=device)

    for t in range(MAX_LENGTH):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

        topv, topi = decoder_output.data.topk(1)  # get candidates
        topi = topi.view(-1)
        decoded_batch[:, t] = topi

        decoder_input = topi.detach().view(-1, 1)

    return decoded_batch