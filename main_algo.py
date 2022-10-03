import pandas as pd
import numpy as np
from utils import *
import ast
import random
import argparse

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def train_eval_loop(df_train, df_test, use_dld, max_sub, cutoff):
    # Generate rules
    og_dict = {}
    for i in range(df_train.shape[0]):
        for j in range(2, max_sub+1):
            og_dict = collate_dict(og_dict, 
                            collect_rules(df_train['Input'][i], 
                                            df_train['Target'][i], 
                                            j,
                                            j))
    for key in og_dict:
        og_dict[key] = og_dict[key]+[key]

    # Rank rules by frequency
    og_dict = collate_max(og_dict, 'test')
    
    result_lst = generate(word_lst=df_test['Input'], 
            rule_dict=og_dict, 
            vocab=vocab_tl, 
            use_dld=use_dld,
            max_sub=max_sub,
            cutoff=cutoff)
    results = evaluate(result_lst, df_test['Target'])
    return results

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

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--use_dld', type=str, required=True)
parser.add_argument('--max_sub', type=str, required=True)
parser.add_argument('--cutoff',  type=str, required=True)
args = parser.parse_args()

USE_DLD = True if args.use_dld=='True' else False
MAX_SUB = int(args.max_sub)
CUTOFF  = int(args.cutoff)
print(f"Use DLD: {USE_DLD}, Max Sub: {MAX_SUB}, Cutoff: {CUTOFF}")

if args.mode=='eval_cv':
    dataset_full = df.append(df_test).reset_index(drop=True)
    partition_lst = partition(list(range(len(dataset_full))), 5)
    full_idxs = set(range(len(dataset_full)))
    result_dict = {}

    fold_count = 1
    for idx_lst in partition_lst:
        
        print(f"Now on fold: {fold_count}")
        
        df_train = dataset_full.loc[list(full_idxs.difference(set(idx_lst)))].reset_index(drop=True)
        df_test  = dataset_full.loc[idx_lst].reset_index(drop=True)
        results = train_eval_loop(df_train, df_test, USE_DLD, MAX_SUB, CUTOFF)

        for key in results:
            if key in result_dict:
                result_dict[key].append(results[key])
            else:
                result_dict[key] = [results[key]]
                
        fold_count += 1
        
    for key in result_dict:
        print(f"{key}: {np.mean(result_dict[key])}, {np.std(result_dict[key])}")
else:
    print(train_eval_loop(df, df_test, USE_DLD, MAX_SUB, CUTOFF))