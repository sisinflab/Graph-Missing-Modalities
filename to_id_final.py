import numpy as np
import pandas as pd
import os


visual_embeddings_folder = 'visual_embeddings/torch/ResNet50/avgpool'
textual_embeddings_folder = 'textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1'

visual_embeddings_folder_indexed = 'visual_embeddings_final_indexed/torch/ResNet50/avgpool'
textual_embeddings_folder_indexed = 'textual_embeddings_final_indexed/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1'

train = pd.read_csv('train_final.tsv', sep='\t', header=None)
val = pd.read_csv('val_final.tsv', sep='\t', header=None)
test = pd.read_csv('test_final.tsv', sep='\t', header=None)

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

train.to_csv('train_final_indexed.tsv', sep='\t', index=False, header=None)
val.to_csv('val_final_indexed.tsv', sep='\t', index=False, header=None)
test.to_csv('test_final_indexed.tsv', sep='\t', index=False, header=None)

if not os.path.exists(visual_embeddings_folder_indexed):
    os.makedirs(visual_embeddings_folder_indexed)

if not os.path.exists(textual_embeddings_folder_indexed):
    os.makedirs(textual_embeddings_folder_indexed)

for key, value in items_map.items():
    np.save(f'{visual_embeddings_folder_indexed}/{value}.npy', np.load(f'{visual_embeddings_folder}/{key}.npy'))
    np.save(f'{textual_embeddings_folder_indexed}/{value}.npy', np.load(f'{textual_embeddings_folder}/{key}.npy'))
