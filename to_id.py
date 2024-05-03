import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description="Run to id.")
parser.add_argument('--data', type=str, default='Digital_Music')
parser.add_argument('--method', type=str, default='zeros')
args = parser.parse_args()

visual_embeddings_folder = f'data/{args.data}/visual_embeddings_{args.method}'
textual_embeddings_folder = f'data/{args.data}/textual_embeddings_{args.method}'

visual_embeddings_folder_indexed = f'data/{args.data}/visual_embeddings_{args.method}_indexed'
textual_embeddings_folder_indexed = f'data/{args.data}/textual_embeddings_{args.method}_indexed'

train = pd.read_csv(f'data/{args.data}/train.tsv', sep='\t', header=None)
val = pd.read_csv(f'data/{args.data}/val.tsv', sep='\t', header=None)
test = pd.read_csv(f'data/{args.data}/test.tsv', sep='\t', header=None)

df = pd.concat([train, val, test], axis=0)

users = df[0].unique()
items = df[1].unique()

users_map = {u: idx for idx, u in enumerate(users)}
items_map = {i: idx for idx, i in enumerate(items)}

train[0] = train[0].map(users_map)
train[1] = train[1].map(items_map)

val[0] = val[0].map(users_map)
val[1] = val[1].map(items_map)

test[0] = test[0].map(users_map)
test[1] = test[1].map(items_map)

train.to_csv(f'data/{args.data}/train_indexed.tsv', sep='\t', index=False, header=None)
val.to_csv(f'data/{args.data}/val_indexed.tsv', sep='\t', index=False, header=None)
test.to_csv(f'data/{args.data}/test_indexed.tsv', sep='\t', index=False, header=None)

if not os.path.exists(visual_embeddings_folder_indexed):
    os.makedirs(visual_embeddings_folder_indexed)

if not os.path.exists(textual_embeddings_folder_indexed):
    os.makedirs(textual_embeddings_folder_indexed)

for key, value in items_map.items():
    np.save(f'data/{args.data}/{visual_embeddings_folder_indexed}/{value}.npy', np.load(f'data/{args.data}/{visual_embeddings_folder}/{key}.npy'))
    np.save(f'data/{args.data}/{textual_embeddings_folder_indexed}/{value}.npy', np.load(f'data/{args.data}/{textual_embeddings_folder}/{key}.npy'))
