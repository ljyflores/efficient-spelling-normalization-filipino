import pandas as pd
import numpy as np
import ast
import random
import string
random.seed(38)

from torch import nn
# from torch.nn import DataParallel
import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import T5ForConditionalGeneration, T5ForConditionalGeneration
from transformers import AdamW

from torch.utils.data import DataLoader

from datasets import Dataset

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from utils import perturb_test_sent, evaluate as evaluate_results

device = torch.device("cuda")

SOURCE_PATH = ""

# Load train/test datasets
df      = pd.read_csv(SOURCE_PATH + "data/train_words.csv")
df_test = pd.read_csv(SOURCE_PATH + "data/test_words.csv")

# Load vocab
f = open("TagalogStemmerPython/output/with_info.txt", "r", encoding='latin1')
f = f.readlines()
vocab_tl = set(ast.literal_eval(item.strip('\n'))['word'] for item in f)
vocab_tl = set(df['Target']).union(vocab_tl) # Add in vocab from dataframe
vocab_tl = set(df_test['Target']).union(vocab_tl) # Add in vocab from test dataframe

# Split into train and val
val_idx  = random.sample(range(df.shape[0]), round(df.shape[0]/5))
df_train = df.loc[list(set(range(df.shape[0])).difference(val_idx))].reset_index(drop=True)
df_val   = df.loc[val_idx].reset_index(drop=True)

dataset_train = Dataset.from_pandas(df_train)
dataset_val   = Dataset.from_pandas(df_val)
dataset_test  = Dataset.from_pandas(df_test)

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def train(model, dataloader, augment, mse_weight, tokenizer, optimizer, USE_BERT):
    model.train()
    loss, steps = 0.0, 0.0
    
    for batch in dataloader:
        model.zero_grad()

        inputs = encode(batch, tokenizer)
        output = model(input_ids      = inputs['input_ids'].to(device),
                       attention_mask = inputs['attention_mask'].to(device),
                       labels         = inputs['labels'].to(device))
        
        if augment=='perturb':
            perturb_batch = {}
            perturb_batch['Input']  = list(map(lambda x: perturb_test_sent(x, vocab_tl), batch['Input'])) 
            perturb_batch['Target'] = batch['Target']
            inputs2 = encode(perturb_batch, tokenizer)
            output2 = model(input_ids      = inputs2['input_ids'].to(device),
                            attention_mask = inputs2['attention_mask'].to(device),
                            labels         = inputs2['labels'].to(device))
            if USE_BERT:
                output2 = model(input_ids      = inputs2['input_ids'].to(device),
                                attention_mask = inputs2['attention_mask'].to(device))
            else:
                output2 = model(input_ids      = inputs2['input_ids'].to(device),
                                attention_mask = inputs2['attention_mask'].to(device),
                                labels         = inputs2['labels'].to(device))
                
            # Compute squared diff loss
            min_idx     = min(output.logits.shape[1], output2.logits.shape[1])
            diff_tensor = output.logits[:,:min_idx,:]-output2.logits[:,:min_idx,:]
            mse_loss    = torch.sqrt(torch.mean(diff_tensor**2)/output.logits.shape[0])
            
            # Compute total loss
            total_loss = (mse_weight*mse_loss)+((1-mse_weight)*output.loss)
            
            total_loss.backward()
            optimizer.step()
            loss += float(total_loss)

        else:
            output.loss.backward()
            optimizer.step()
            loss += float(output.loss)
        steps += 1

        if 'autoencode' in augment:
            augment_size = int(augment[-1])
            random_str_lst = [''.join(random.choices(string.ascii_lowercase, k=10)) for i in range(len(batch['Input'])*augment_size)]
            
            ae_batch = {}
            ae_batch['Input']  = random_str_lst 
            ae_batch['Target'] = random_str_lst
            inputs3 = encode(ae_batch, tokenizer)
            output3 = model(input_ids      = inputs3['input_ids'].to(device),
                            attention_mask = inputs3['attention_mask'].to(device),
                            labels         = inputs3['labels'].to(device))
            
            output3.loss.backward()
            optimizer.step()
            
    return loss/steps

def evaluate(model, dataloader, tokenizer):
    loss, steps = 0.0, 0.0
    with torch.no_grad():
        for batch in dataloader:
    
            inputs = encode(batch, tokenizer)
            output = model(input_ids      = inputs['input_ids'].to(device),
                           attention_mask = inputs['attention_mask'].to(device),
                           labels         = inputs['labels'].to(device))
            loss += float(output.loss)
            steps += 1
    return loss/steps

def clean_word(s):
    return s.replace('<pad>','').replace('</s>','')

# Generate top 5 words per candidate
def generate_k_candidates(model, dataloader, tokenizer, k=5):
    result = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = encode(batch, tokenizer)
            output = model.generate(input_ids      = inputs['input_ids'].to(device),
                                    attention_mask = inputs['attention_mask'].to(device),
                                    num_return_sequences = k,
                                    num_beams = k)
            output = tokenizer.batch_decode(output)
            output = list(map(clean_word, output))
            result.append(output)
    return result

def initialize(use_bert, lr, eps):
    # Initialize tokenizers, model, loss, optimizer
    if use_bert:
        tokenizer = AutoTokenizer.from_pretrained("jcblaise/roberta-tagalog-large")
        model = AutoModelForMaskedLM.from_pretrained("jcblaise/roberta-tagalog-large").to(device)
    else:
        model = T5ForConditionalGeneration.from_pretrained("google/byt5-small").to(device)
        tokenizer = AutoTokenizer.from_pretrained("google/byt5-small",
                                                  output_scores=True,
                                                  output_hidden_states=True)

    nll_loss = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(),
                      lr = lr,  # args.learning_rate - default is 5e-5
                      eps = eps # args.adam_epsilon  - default is 1e-8.
                    )
    return model, tokenizer, nll_loss, optimizer

def encode(examples, tokenizer):
    batch_size = len(examples['Input'])
    inputs = examples['Input']
    targets = examples['Target']
    
    tokenized_inputs = tokenizer(inputs+targets, 
                                 return_tensors='pt', 
                                 padding=True)
    model_inputs = {}
    model_inputs['input_ids']      = tokenized_inputs['input_ids'][:batch_size]
    model_inputs['attention_mask'] = tokenized_inputs['attention_mask'][:batch_size]
    model_inputs['labels']         = tokenized_inputs['input_ids'][batch_size:]

    return model_inputs

def train_loop(args):
    
    AUGMENT_MODE = args['augment_mode']   # False  if 'augment_mode'   not in args else args['augment_mode']
    MSE_WEIGHT   = args['mse_weight']     # 0.5    if 'mse_weight'     not in args else args['mse_weight']
    EARLY_STOP   = args['early_stopping'] # False  if 'early_stop'     not in args else args['early_stop']
    EPS          = args['eps']        # 1e-8   if 'eps'            not in args else args['eps'] #.sample()
    LR           = args['lr']         # 5e-5   if 'lr'             not in args else args['lr'] #.sample()
    USE_BERT     = args['use_bert']   # True   if 'use_bert'       not in args else args['use_bert']
    EPOCHS       = args['epochs']     # np.inf if 'epochs'         not in args else args['epochs'] #.sample()
    BATCH_SIZE   = args['batch_size'] # 8      if 'batch_size'     not in args else args['batch_size'] #.sample()
    REPORT       = args['report']     # True   if 'report'         not in args else args['report']
    MODEL_NAME   = args['model_name'] # False  if 'model_name'     not in args else args['model_name']

    model, tokenizer, nll_loss, optimizer = initialize(use_bert=USE_BERT, lr=LR, eps=EPS)
    
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(dataset_val,   batch_size=BATCH_SIZE, shuffle=False) 
    test_loader  = DataLoader(dataset_test,  batch_size=BATCH_SIZE, shuffle=False) 

    best_val_loss = np.inf
    epochs = 0

    while epochs<EPOCHS: 
        try:
            train_loss = train(model, train_loader, AUGMENT_MODE, MSE_WEIGHT, tokenizer, optimizer, USE_BERT)
            val_loss   = evaluate(model, val_loader, tokenizer)
        except RuntimeError:
            continue
        epochs += 1
        if REPORT:
            tune.report(TRAIN_LOSS=train_loss, VAL_LOSS=val_loss)
        else:
            print(f"Epoch {epochs}; Train: {train_loss}; Test: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if MODEL_NAME != '':
                model.save_pretrained(f"models/{MODEL_NAME}")
        else:
            if EARLY_STOP:
                break
        
    if not REPORT:
        output_lst = generate_k_candidates(model, test_loader, tokenizer, 5)
        result_dict = evaluate_results(output_lst, list(map(lambda x: x['Target'], dataset_test)))
        # print(result_dict)
        return result_dict
    print("Finished Training")
    
    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--augment_mode',   type=str,   required=True,  default='none')
parser.add_argument('--use_bert',       type=str,   required=True,  default='True')
parser.add_argument('--early_stopping', type=str,   required=True,  default='False')
parser.add_argument('--mode',           type=str,   required=False, default='train')
parser.add_argument('--lr',             type=float, required=False, default=5e-5)
parser.add_argument('--eps',            type=float, required=False, default=1e-8)
parser.add_argument('--epochs',         type=float, required=False, default=np.inf)
parser.add_argument('--batch_size',     type=int,   required=False, default=8)
parser.add_argument('--mse_weight',     type=float, required=False, default=0.5)
parser.add_argument('--model_name',     type=str,   required=False, default='')

args = parser.parse_args()

if args.mode=='train':
    args = {
        "augment_mode":   args.augment_mode,
        "use_bert":       False if args.use_bert=='False' else True,
        "early_stopping": False if args.early_stopping=='False' else True,
        "lr":             5e-5, # tune.loguniform(1e-6, 1e-4),
        "eps":            1e-8, # tune.loguniform(1e-9, 1e-6),
        "epochs":         tune.choice([10,30,50,70]),
        "batch_size":     tune.choice([1, 2, 4, 8, 16]),
        "mse_weight":     None if args.augment_mode=='False' else tune.choice([0.2, 0.4, 0.6, 0.8, 1.0]),
        "report":         True,
        "model_name":     args.model_name
    }
    # print(f"perturb:{args['perturb']}")
    # print(f"use_bert:{args['use_bert']}")
    scheduler = ASHAScheduler(
            max_t=70,
            grace_period=1,
            reduction_factor=2)
    result = tune.run(
            tune.with_parameters(train_loop),
            resources_per_trial={"gpu": 1},
            config=args,
            metric="VAL_LOSS",
            mode="min",
            num_samples=100,
            scheduler=scheduler
        )

elif 'eval' in args.mode:
    print(f"Early Stop {args.early_stopping}")
    print(f"Use BERT {args.use_bert}")
    print(f"Augment Mode {args.augment_mode}")
    
    args = {
        "augment_mode":   args.augment_mode,
        "use_bert":       False if args.use_bert=='False' else True,
        "early_stopping": False if args.early_stopping=='False' else True,
        "lr":             args.lr,
        "eps":            args.eps,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "mse_weight":     args.mse_weight,
        "model_name":     args.model_name,
        "report":         False
    }
    if 'cv' in args.mode:
        dataset_full = Dataset.from_pandas(df.append(df_val).reset_index(drop=True))
        partition_lst = partition(list(range(len(dataset_full))), 5)
        full_idxs = set(range(len(dataset_full)))
        result_dict = {}
        
        for idx_lst in partition_lst:
            dataset_train = torch.utils.data.dataset.Subset(dataset_full, list(full_idxs.difference(set(idx_lst))))
            dataset_val   = dataset_train
            dataset_test  = torch.utils.data.dataset.Subset(dataset_full, idx_lst)
            
            results = train_loop(args)
            for key in results:
                if key in result_dict:
                    result_dict[key].append(results[key])
                else:
                    result_dict[key] = [results[key]]
        for key in result_dict:
            print(f"{key}: {np.mean(result_dict[key])}, {np.std(result_dict[key])}")
    else:
        print(train_loop(args))
else:
    pass


