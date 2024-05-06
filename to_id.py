import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description="Run to id.")
parser.add_argument('--data', type=str, default='Digital_Music')
parser.add_argument('--method', type=str, default='zeros')
args = parser.parse_args()

visual_folder_original = f'./data/{args.data}/visual_embeddings/torch/ResNet50/avgpool'
textual_folder_original = f'./data/{args.data}/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1'

visual_folder_imputed = f'data/{args.data}/visual_embeddings_{args.method}'
textual_folder_imputed = f'data/{args.data}/textual_embeddings_{args.method}'

visual_folder_original_indexed = f'./data/{args.data}/visual_embeddings_indexed'
textual_folder_original_indexed = f'./data/{args.data}/textual_embeddings_indexed'

visual_folder_imputed_indexed = f'data/{args.data}/visual_embeddings_{args.method}_indexed'
textual_folder_imputed_indexed = f'data/{args.data}/textual_embeddings_{args.method}_indexed'

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

try:
    missing_visual = pd.read_csv(f'data/{args.data}/missing_visual.tsv', sep='\t', header=None)
    missing_visual[0] = missing_visual[0].map(items_map)
    missing_visual.to_csv(f'data/{args.data}/missing_visual_indexed.tsv', index=False, header=None, sep='\t')
except pd.errors.EmptyDataError:
    pass

try:
    missing_textual = pd.read_csv(f'data/{args.data}/missing_textual.tsv', sep='\t', header=None)
    missing_textual[0] = missing_textual[0].map(items_map)
    missing_textual.to_csv(f'data/{args.data}/missing_textual_indexed.tsv', index=False, header=None, sep='\t')
except pd.errors.EmptyDataError:
    pass

users_map_df = pd.DataFrame(list(users_map.items()))
items_map_df = pd.DataFrame(list(items_map.items()))

users_map_df.to_csv(f'data/{args.data}/users_map.tsv', index=False, header=None, sep='\t')
items_map_df.to_csv(f'data/{args.data}/items_map.tsv', index=False, header=None, sep='\t')

train.to_csv(f'data/{args.data}/train_indexed.tsv', sep='\t', index=False, header=None)
val.to_csv(f'data/{args.data}/val_indexed.tsv', sep='\t', index=False, header=None)
test.to_csv(f'data/{args.data}/test_indexed.tsv', sep='\t', index=False, header=None)

visual_folder_original_indexed = f'./data/{args.data}/visual_embeddings_indexed'
textual_folder_original_indexed = f'./data/{args.data}/textual_embeddings_indexed'

visual_folder_imputed_indexed = f'data/{args.data}/visual_embeddings_{args.method}_indexed'
textual_folder_imputed_indexed = f'data/{args.data}/textual_embeddings_{args.method}_indexed'

if not os.path.exists(visual_folder_original_indexed):
    os.makedirs(visual_folder_original_indexed)

if not os.path.exists(textual_folder_original_indexed):
    os.makedirs(textual_folder_original_indexed)

if not os.path.exists(visual_folder_imputed_indexed):
    os.makedirs(visual_folder_imputed_indexed)

if not os.path.exists(textual_folder_imputed_indexed):
    os.makedirs(textual_folder_imputed_indexed)

for it in os.listdir(visual_folder_original):
    np.save(f'{visual_folder_original_indexed}/{items_map[int(it.split(".npy")[0])]}.npy', np.load(f'{visual_folder_original}/{it}.npy'))

for it in os.listdir(textual_folder_original):
    np.save(f'{textual_folder_original_indexed}/{items_map[int(it.split(".npy")[0])]}.npy', np.load(f'{textual_folder_original}/{it}.npy'))

for it in os.listdir(visual_folder_imputed):
    np.save(f'{visual_folder_imputed_indexed}/{items_map[int(it.split(".npy")[0])]}.npy', np.load(f'{visual_folder_imputed}/{it}.npy'))

for it in os.listdir(textual_folder_imputed):
    np.save(f'{textual_folder_imputed_indexed}/{items_map[int(it.split(".npy")[0])]}.npy', np.load(f'{textual_folder_imputed}/{it}.npy'))
