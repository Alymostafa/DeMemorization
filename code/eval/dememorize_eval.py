from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM
from datasets import load_dataset
# from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

import torch
import wandb
import time
import os
import statistics
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()
from multiprocessing import Pool

from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, pipeline
import argparse
from evaluate import load

from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
tqdm_notebook.pandas()
from torchmetrics.functional import accuracy

import argparse



from accelerate.tracking import GeneralTracker, on_main_process
from typing import Optional

import wandb

parser = argparse.ArgumentParser(description='lm_extraction')
parser.add_argument('--dataset_type',  help='')
parser.add_argument('--lm_name',  help='')
parser.add_argument('--model_dir',  help='')

args = parser.parse_args()


config_sc = {
    "lm_name": str(args.lm_name),
    "alpha_bleu": 0.5,
    "beta_ppl": 0.5,
    "reward_type": 'bert',
    "ref_lm_name": str(args.lm_name),
    "cls_model_name": "null",
    "tk_name": "lm_extraction",
    "reward_fn": 'bert',
    "steps": 2,
    "batch_size": 32,
    "forward_batch_size": 8,
    "ppo_epochs": 2,   
    "lr": 1.41e-5,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
    "dataset_type":args.dataset_type
}

# lm_mode = 'org'
lm_mode = 'demem'


class WandbEd(GeneralTracker):
    name = "wandb"
    requires_logging_directory = False

    @on_main_process
    def __init__(self, project: str, run_name, group: str, entity: str, config: dict):
        self.group = group
        self.run_name = run_name
        self.project = project
        self.config = config
        self.entity = entity
        run = wandb.init(group = self.group, name = self.run_name, entity = self.entity, project = self.project, config=config)

    @property
    def tracker(self):
        return self.run.run

    @on_main_process
    def store_init_configuration(self, values: dict):
        wandb.config = values

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None):
        wandb.log(values)

if lm_mode == 'org':
    run_name = config_sc['lm_name'].split('/')[1] +'_'+str(lm_mode)+'_'+config_sc['dataset_type']
else:    
    run_name = config_sc['lm_name'].split('/')[1] +'_'+config_sc['dataset_type']
    
wandbed_obj = WandbEd(group=config_sc['lm_name'], project='lm_extraction_defence_eval', entity = "thesis_projects", run_name=run_name, config = config_sc)


lm_data = pd.read_csv('all_train.csv')
ds = Dataset.from_pandas(lm_data)

import random
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()



from datasets import load_metric
sacrebleu = load_metric('sacrebleu')
bertscore = load("bertscore")

def sacrebleu_fn(label, response):
    score = sacrebleu.compute(predictions=[response], references=[[label]])['score']
    return 100-score 

def calculatePerplexity(sentence):
     """
     exp(loss)
     """
     input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
     input_ids = input_ids.to('cuda')
     with torch.no_grad():
         outputs = ppl_model(input_ids, labels=input_ids)
     loss, logits = outputs[:2]
     return torch.exp(loss)

def perplexity_fn(text):
    ppl_lst = []
    for i in text:
        ppl_lst.append(calculatePerplexity(i).unsqueeze(0))
    ppl_tns = torch.cat(ppl_lst)
    return ppl_tns, torch.mean(ppl_tns)



tokenizer = AutoTokenizer.from_pretrained(config_sc['lm_name'])
tokenizer.pad_token = tokenizer.eos_token


def tokenize(sample):
    sample["tokens"] = tokenizer.encode(sample["prefix"])
    sample["query"] = tokenizer.decode(sample["tokens"])
    return sample

ds = ds.map(tokenize, batched=False)



gen_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}


def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


mode = 'inference'
if mode == 'train':
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config_sc['lm_name'],
        device_map="auto")

    model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(config_sc['lm_name'], device_map="auto")
elif mode == 'inference':
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_dir,
        device_map="auto")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = tokenizer.eos_token

if config_sc['lm_name'] == 'EleutherAI/gpt-neo-125m':
    lm_dir = 'models/models--EleutherAI--gpt-neo-125m/snapshots/b983397156c0991016feccfbcbe1fe2746d47b29'
    
if config_sc['lm_name'] == 'EleutherAI/gpt-neo-2.7B':
    lm_dir = 'models/models--EleutherAI--gpt-neo-2.7B/snapshots/a0e5677e8611d87c8ce501dfa6a713e42ba2f6ef/'
    
elif config_sc['lm_name'] == 'EleutherAI/gpt-neo-1.3B':
    lm_dir = 'models/models--EleutherAI--gpt-neo-1.3B/snapshots/0f35a20339a9e208bc061570981180cfb87c9572'
    
elif config_sc['lm_name'] == 'facebook/opt-125m':   
    lm_dir = 'models/models--facebook--opt-125m/snapshots/3d2b5f275bdf882b8775f902e1bfdb790e2cfc32/'
    
elif config_sc['lm_name'] == 'facebook/opt-1.3b':   
    lm_dir = 'models/models--facebook--opt-1.3b/snapshots/8c7b10754972749675d22364c25c428b29face51/'

elif config_sc['lm_name'] == 'facebook/opt-2.7b':   
    lm_dir = 'models/models--facebook--opt-2.7b/snapshots/397f71a473a150c00f0fe3fc4a2f78ff3ccaf82d/'    
    
ppl_model = AutoModelForCausalLM.from_pretrained(lm_dir, device_map="auto")    


if mode == 'train':
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config, model, ref_model=model_ref, tokenizer=tokenizer, dataset=ds, data_collator=collater, optimizer=optimizer
    )
else:
    pass


def validation_ma(max_len, input_ids):
    labels, preds = [], []
    for i in range(1, max_len):
        label = input_ids[..., i]
        prompt = input_ids[..., :i]
        try:
            pred = model.generate(prompt, max_length=i + 1, pad_token_id=tokenizer.eos_token_id)[:, -1]
        except IndexError:  # if batch == 1
            pred = model.generate(torch.squeeze(
                prompt), max_length=i + 1).squeeze()[-1]

        labels.append(torch.squeeze(label))
        preds.append(torch.squeeze(pred))
    
    preds = torch.stack(preds)
    labels = torch.stack(labels)
    score = accuracy(preds, labels, ignore_index=-100)
    return score



def inference_fn(test_data, col, max_len):
    test_logs = dict()
    test_scores = pd.DataFrame(columns=['idx','text','pred','true', 'bleu', 'ppl_gen', 'ppl_true', 'ma'])

    for i in tqdm(range(len(test_data))):
        prefix = test_data[col][i]
        inputs = tokenizer([prefix],  return_tensors="pt")
        sample  = model.generate(**inputs.to('cuda'), max_length=max_len, pad_token_id=tokenizer.eos_token_id)

        suffix_gen = tokenizer.decode(sample[0])
        sample_text = suffix_gen[len(prefix):]
        suffix = test_data['suffix'][i]

        ppl_true = calculatePerplexity(prefix + suffix)
        ppl_gen = calculatePerplexity(prefix + sample_text)
        bleu = sacrebleu_fn(suffix, sample_text)

        gt = tokenizer(test_data[col][i] + test_data['suffix'][i],  return_tensors="pt")
        gen_len = len(sample[0]) - len(inputs['input_ids'][0]) 
        if gen_len >1:
            ma = validation_ma(gen_len, gt['input_ids'].cuda())    

            test_scores.loc[i ,'idx'] = i
            test_scores.loc[i, 'text'] = prefix
            test_scores.loc[i, 'pred'] = sample_text
            test_scores.loc[i, 'true'] = suffix
            test_scores.loc[i, 'bleu'] = bleu
            test_scores.loc[i, 'ppl_gen'] = ppl_gen.cpu().tolist()
            test_scores.loc[i, 'ppl_true'] = ppl_true.cpu().tolist()
            test_scores.loc[i, 'ma'] = ma.cpu().tolist()
        else:
            pass
    wandb.log({"test_samples": test_scores})
    
    test_logs['test/ppl_gen'] = test_scores['ppl_gen'].mean()
    test_logs['test/ppl_true'] = test_scores['ppl_true'].mean()
    test_logs['test/bleu'] = test_scores['bleu'].mean()
    test_logs['test/ma'] = test_scores['ma'].mean()
    
    performance_table = wandb.Table(columns=["ppl_gen", "ppl_true", "bleu", "ma"], data=[[test_scores['ppl_gen'].mean(), test_scores['ppl_true'].mean(), test_scores['bleu'].mean(), test_scores['ma'].mean()]])
    wandb.log({"performance_table": performance_table})

    return test_scores 

test_data = pd.read_csv('test_data.csv')

if config_sc['dataset_type'] == 'reg':
    test_scores = inference_fn(test_data, 'prefix', 103)
elif config_sc['dataset_type'] == 'lc':
    test_scores = inference_fn(test_data, 'pre_and_prefix', 203)
else:
    raise ValueError(f"Expected type to be reg or lc. Got: {config_sc['dataset_type']}")
    
    
test_scores.to_csv('/scratch/aly2000/lm_extraction/ddp/dfs_results/'+str(run_name)+'.csv', index=False)
