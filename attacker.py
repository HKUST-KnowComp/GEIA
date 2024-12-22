import argparse
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import json
from typing import Optional
import os

from transformers import AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM,GPT2Config,GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from attacker_models import SequenceCrossEntropyLoss
from data_process import get_sent_list

model_cards = {
    'sent_t5_large': 'sentence-t5-large',
    'sent_t5_base': 'sentence-t5-base',
    'sent_t5_xl': 'sentence-t5-xl',
    'sent_t5_xxl': 'sentence-t5-xxl',
    'mpnet': 'all-mpnet-base-v1',
    'sent_roberta': 'all-roberta-large-v1',
    'simcse_bert': 'princeton-nlp/sup-simcse-bert-large-uncased',
    'simcse_roberta': 'princeton-nlp/sup-simcse-roberta-large',
    'gpt2_large': 'microsoft/DialoGPT-large',
    'gpt2_medium': 'microsoft/DialoGPT-medium',
    'llama_3_1B': 'meta-llama/Llama-3.2-1B',
    'llama_3_3B': 'meta-llama/Llama-3.2-3B'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class text_dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):  
        return self.data[index]
        
    def collate(self, unpacked_data):
        return unpacked_data
    

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def setup_optimizer(attack_model):
    param_optimizer = list(attack_model.named_parameters())
    no_decay = ['bias', 'ln', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-06)
    return optimizer


def get_language_model(model_name):
    if 'llama_3' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_cards[model_name]).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_cards[model_name])
    elif 'gpt2' in model_name:
        config = GPT2Config.from_pretrained(model_cards[model_name])
        tokenizer = AutoTokenizer.from_pretrained(model_cards[model_name])
        model = GPT2LMHeadModel(config).to(device)
    else:
        raise ValueError("Model not found")
    return model, tokenizer


def get_embeddings(embed_model, batch_text, embed_tokenizer: Optional[AutoTokenizer] = None):
    if embed_tokenizer:
        inputs = embed_tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt").to(device)
        embeddings = embed_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    else:
        embeddings = embed_model.encode(batch_text, convert_to_tensor=True).to(device)
    return embeddings


def get_dimensions(embed_model, attack_model):
    if isinstance(embed_model, SentenceTransformer):
        embed_dim = embed_model.get_sentence_embedding_dimension()
    else:
        embed_dim = embed_model.config.hidden_size
    attack_dim = attack_model.config.hidden_size
    return embed_dim, attack_dim


def train(config, data):
    if 'simcse' in config['embed_model']:
        embed_model = AutoModel.from_pretrained(model_cards[config['embed_model']]).to(device)
        embed_tokenizer = AutoTokenizer.from_pretrained(model_cards[config['embed_model']])
    else:
        embed_model = SentenceTransformer(model_cards[config['embed_model']], device=device)
        embed_tokenizer = None

    attack_model, tokenizer = get_language_model(config['attack_model'])
    tokenizer.pad_token = tokenizer.eos_token

    dataset = text_dataset(data)
    dataloader = DataLoader(dataset, config['batch_size'], True, collate_fn=dataset.collate)
    optimizer = setup_optimizer(attack_model)
    scheduler = get_linear_schedule_with_warmup(optimizer, 100, len(dataloader) * config['num_epochs'])
    
    embed_dim, attack_dim = get_dimensions(embed_model, attack_model)

    if embed_dim != attack_dim:
        projection = ProjectionLayer(embed_dim, attack_dim).to(device)
        optimizer.add_param_group({'params': projection.parameters()})

    if config['noise'] and type(config['std']) != float:
        std_path = "overall_avg_diff_" + config['embed_model'] + ".pt"
        std_tensor = torch.load(std_path)
        # std_tensor = torch.sqrt(variances)

    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        embed_model.eval()
        for batch_text in tqdm(dataloader, desc="Training"):
            with torch.no_grad():
                embeddings = get_embeddings(embed_model, batch_text, embed_tokenizer)

                if config['noise'] and np.random.rand() > 0.25:
                    if type(config['std']) != float:
                        noise = torch.normal(mean=0.0, std=std_tensor.expand(embeddings.size(0), -1))
                    else:
                        noise = torch.normal(mean=0, std=config['std'], size=embeddings.shape).to(device)
                    embeddings += noise

            if embed_dim != attack_dim:
                embeddings = projection(embeddings)
            
            inputs = tokenizer(batch_text, return_tensors='pt', padding='max_length', truncation=True, max_length=40)

            train_on_batch(embeddings, inputs, attack_model, SequenceCrossEntropyLoss())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
    
    if embed_dim != attack_dim:
        torch.save(projection.state_dict(), config['proj_path'])
            
    attack_model.save_pretrained(config['attack_path'])


def test(config, data):
    if 'simcse' in config['embed_model']:
        embed_model = AutoModel.from_pretrained(model_cards[config['embed_model']]).to(device)
        embed_tokenizer = AutoTokenizer.from_pretrained(model_cards[config['embed_model']])
    else:
        embed_model = SentenceTransformer(model_cards[config['embed_model']], device=device)
        embed_tokenizer = None

    attack_model = AutoModelForCausalLM.from_pretrained(config['attack_path']).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_cards[config['attack_model']])

    dataset = text_dataset(data)
    dataloader = DataLoader(dataset, config['batch_size'], False, collate_fn=dataset.collate) # no shuffle for testing data

    embed_dim, attack_dim = get_dimensions(embed_model, attack_model)

    if embed_dim != attack_dim:
        projection = ProjectionLayer(embed_dim, attack_dim).to(device)
        projection.load_state_dict(torch.load(config['proj_path']))

    sent_dict = {'gt': [], 'pred': []}

    with torch.no_grad():
        for batch_text in tqdm(dataloader, desc="Testing"):
            embeddings = get_embeddings(embed_model, batch_text, embed_tokenizer)
            if embed_dim != attack_dim:
                embeddings = projection(embeddings).to(device)
            
            embeddings = embeddings.unsqueeze(1)
            
            if config['beam']:
                outputs = attack_model.generate(inputs_embeds=embeddings, max_new_tokens=250, num_beams=5, early_stopping=True)
            else:
                outputs = attack_model.generate(inputs_embeds=embeddings, max_new_tokens=250, do_sample=True, temperature=0.9, top_p=0.9, top_k=-1)
            
            sent_list = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            sent_dict['pred'].extend(sent_list)
            sent_dict['gt'].extend(batch_text)
        
        with open(config['output_path'], 'w') as f:
            json.dump(sent_dict, f, indent=4)


def train_on_batch(batch_X, inputs, model, criterion):
    input_ids = inputs['input_ids'].to(device) # tensors of input ids
    labels = input_ids.clone()
    
    input_emb = model.get_input_embeddings()(input_ids) # embed the input ids using GPT-2 embedding
    batch_X = batch_X.to(device)
    batch_X_unsqueeze = torch.unsqueeze(batch_X, 1)     # add extra dim to cat together
    inputs_embeds = torch.cat((batch_X_unsqueeze,input_emb),dim=1)  #[batch,max_length+1,emb_dim (1024)]
    
    logits, _ = model(inputs_embeds=inputs_embeds, past_key_values=None, return_dict=False)
    logits = logits[:, :-1].contiguous()
    target = labels.contiguous()
    target_mask = torch.ones_like(target).float()
    loss = criterion(logits, target, target_mask, label_smoothing=0.02, reduce="batch")  

    record_loss = loss.item()
    perplexity = np.exp(record_loss)
    loss.backward()

    return record_loss, perplexity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training the GEIA attacker on different sentence embedding models')
    parser.add_argument('--attack_model', type=str, default='gpt2_medium', help='Name of the attacker model')
    parser.add_argument('--embed_model', type=str, default='sent_t5_large', help='Name of embedding model')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--dataset', type=str, default='personachat', help='Name of dataset: PersonaChat or QNLI')
    parser.add_argument('--data_type', type=str, default='test', help='train/test')
    parser.add_argument('--beam', type=bool, default=True, help='Toggle beam decoding method (sampling/beam)')
    parser.add_argument('--noise', type=bool, default=False, help='Toggle adding Gaussian noise to the embeddings during training')
    parser.add_argument('--std', type=float, default=None, help='Size of the std of the Gaussian noise')

    args = parser.parse_args()

    config = {
        'attack_model': args.attack_model,
        'embed_model': args.embed_model,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'dataset': args.dataset,
        'data_type': args.data_type,
        'beam': args.beam,
        'base_path': f'models/{args.dataset}',
        'attack_path': f'models/{args.dataset}/attacker_{args.attack_model}_{args.embed_model}{"_noise" if args.noise else ""}{"_std_" + str(args.std) if args.std is not None else ""}',
        'proj_path': f'models/{args.dataset}/projection_{args.attack_model}_{args.embed_model}{"_noise" if args.noise else ""}{"_std_" + str(args.std) if args.std is not None else ""}',
        'output_path': f'models/{args.dataset}/output_{args.attack_model}_{args.embed_model}{"_beam" if args.beam else ""}{"_noise" if args.noise else ""}{"_std_" + str(args.std) if args.std is not None else ""}.log',
        'noise' : args.noise,
        'std' : args.std
    }
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    sent_list = get_sent_list(args.dataset, args.data_type)

    if args.data_type == 'train':
        train(config, sent_list)
    elif args.data_type == 'test':
        test(config, sent_list)
