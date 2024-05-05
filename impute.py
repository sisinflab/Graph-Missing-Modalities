import argparse
import os
import pandas as pd
import numpy as np
import shutil
import torch
from torch_sparse import SparseTensor, mul, sum, fill_diag, matmul
import scipy.sparse as sp

np.random.seed(42)


def compute_normalized_laplacian(adj, norm):
    adj = fill_diag(adj, 0.)
    deg = sum(adj, dim=-1)
    deg_inv_sqrt = deg.pow_(-norm)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t

parser = argparse.ArgumentParser(description="Run imputation.")
parser.add_argument('--data', type=str, default='Digital_Music')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--method', type=str, default='feat_prop')
parser.add_argument('--top_k', type=int, default=20)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

visual_folder = f'data/{args.data}/visual_embeddings/torch/ResNet50/avgpool'
textual_folder = f'data/{args.data}/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1'

output_visual = f'data/{args.data}/visual_embeddings_{args.method}'
output_textual = f'data/{args.data}/textual_embeddings_{args.method}'

try:
    missing_visual = pd.read_csv(os.path.join(f'data/{args.data}', 'missing_visual.tsv'), sep='\t', header=None)
    missing_visual = set(missing_visual[0].tolist())
except pd.errors.EmptyDataError:
    missing_visual = set()

try:
    missing_textual = pd.read_csv(os.path.join(f'data/{args.data}', 'missing_textual.tsv'), sep='\t', header=None)
    missing_textual = set(missing_textual[0].tolist())
except pd.errors.EmptyDataError:
    missing_textual = set()

visual_shape = np.load(os.path.join(visual_folder, os.listdir(visual_folder)[0])).shape
textual_shape = np.load(os.path.join(textual_folder, os.listdir(textual_folder)[0])).shape

if args.method == 'feat_prop':
    if not os.path.exists(output_visual + f'_{args.layers}_{args.top_k}_indexed'):
        os.makedirs(output_visual + f'_{args.layers}_{args.top_k}_indexed')
    if not os.path.exists(output_textual + f'_{args.layers}_{args.top_k}_indexed'):
        os.makedirs(output_textual + f'_{args.layers}_{args.top_k}_indexed')
else:
    if not os.path.exists(output_visual):
        os.makedirs(output_visual)
    if not os.path.exists(output_textual):
        os.makedirs(output_textual)

if args.method == 'zeros':
    for miss in missing_visual:
        np.save(os.path.join(output_visual, f'{miss}.npy'), np.zeros(visual_shape, dtype=np.float32))

    for miss in missing_textual:
        np.save(os.path.join(output_textual, f'{miss}.npy'), np.zeros(textual_shape, dtype=np.float32))

elif args.method == 'random':
    for miss in missing_visual:
        np.save(os.path.join(output_visual, f'{miss}.npy'), np.random.rand(*visual_shape))

    for miss in missing_textual:
        np.save(os.path.join(output_textual, f'{miss}.npy'), np.random.rand(*textual_shape))

elif args.method == 'mean':
    num_items_visual = len(os.listdir(visual_folder))
    num_items_textual = len(os.listdir(textual_folder))

    visual_features = np.empty((num_items_visual, visual_shape[-1])) if num_items_visual else None
    textual_features = np.empty((num_items_textual, textual_shape[-1])) if num_items_textual else None

    if visual_features is not None:
        visual_items = os.listdir(visual_folder)
        for idx, it in enumerate(visual_items):
            visual_features[idx, :] = np.load(os.path.join(visual_folder, it))
        mean_visual = visual_features.mean(axis=0, keepdims=True)
        for miss in missing_visual:
            np.save(os.path.join(output_visual, f'{miss}.npy'), mean_visual)

    if textual_features is not None:
        textual_items = os.listdir(textual_folder)
        for idx, it in enumerate(textual_items):
            textual_features[idx, :] = np.load(os.path.join(textual_folder, it))
        mean_textual = textual_features.mean(axis=0, keepdims=True)
        for miss in missing_textual:
            np.save(os.path.join(output_textual, f'{miss}.npy'), mean_textual)

elif args.method == 'feat_prop':

    output_visual = f'data/{args.data}/visual_embeddings_{args.method}_{args.layers}_{args.top_k}_indexed'
    output_textual = f'data/{args.data}/textual_embeddings_{args.method}_{args.layers}_{args.top_k}_indexed'

    try:
        train = pd.read_csv(f'data/{args.data}/train_indexed.tsv', sep='\t', header=None)
    except FileNotFoundError:
        print('Before imputing through feat_prop, split the dataset into train/val/test!')
        exit()

    try:
        num_items_visual = len(os.listdir(f'data/{args.data}/visual_embeddings_zeros_indexed'))
        num_items_textual = len(os.listdir(f'data/{args.data}/textual_embeddings_zeros_indexed'))
    except FileNotFoundError:
        print('Before imputing through feat_prop, impute through zeros!')
        exit()

    visual_features = torch.empty((num_items_visual, visual_shape[-1]))
    textual_features = torch.empty((num_items_textual, textual_shape[-1]))

    # compute item_item matrix
    user_item = sp.coo_matrix(([1.0]*len(train), (train[0].tolist(), train[1].tolist())),
                              shape=(train[0].nunique(), num_items_visual), dtype=np.float32)
    item_item = user_item.transpose().dot(user_item).toarray()
    knn_val, knn_ind = torch.topk(torch.tensor(item_item, device=device), args.top_k, dim=-1)
    items_cols = torch.flatten(knn_ind).to(device)
    ir = torch.tensor(list(range(item_item.shape[0])), dtype=torch.int64, device=device)
    items_rows = torch.repeat_interleave(ir, args.top_k).to(device)
    adj = SparseTensor(row=items_rows,
                       col=items_cols,
                       value=torch.tensor([1.0] * items_rows.shape[0], device=device),
                       sparse_sizes=(item_item.shape[0], item_item.shape[0]))
    # normalize adjacency matrix
    adj = compute_normalized_laplacian(adj, 0.5)

    try:
        missing_visual_indexed = pd.read_csv(os.path.join(f'data/{args.data}', 'missing_visual_indexed.tsv'), sep='\t', header=None)
        missing_visual_indexed = set(missing_visual_indexed[0].tolist())
    except pd.errors.EmptyDataError:
        missing_visual_indexed = set()

    try:
        missing_textual_indexed = pd.read_csv(os.path.join(f'data/{args.data}', 'missing_textual_indexed.tsv'), sep='\t', header=None)
        missing_textual_indexed = set(missing_textual_indexed[0].tolist())
    except pd.errors.EmptyDataError:
        missing_textual_indexed = set()

    # feat prop on visual features
    for f in os.listdir(f'data/{args.data}/visual_embeddings_zeros_indexed'):
        visual_features[int(f.split('.npy')[0]), :] = torch.from_numpy(np.load(os.path.join(f'data/{args.data}/visual_embeddings_zeros_indexed', f)))

    non_missing_items = list(set(list(range(num_items_visual))).difference(missing_visual_indexed))
    propagated_visual_features = visual_features.clone()

    for idx in range(args.layers):
        print(f'[VISUAL] Propagation layer: {idx + 1}')
        propagated_visual_features = matmul(adj.to(device), propagated_visual_features.to(device))
        propagated_visual_features[non_missing_items] = visual_features[non_missing_items].to(device)

    for miss in missing_visual_indexed:
        np.save(os.path.join(output_visual, f'{miss}.npy'), propagated_visual_features[miss].detach().cpu().numpy())

    # feat prop on textual features
    for f in os.listdir(f'data/{args.data}/textual_embeddings_zeros_indexed'):
        textual_features[int(f.split('.npy')[0]), :] = torch.from_numpy(np.load(os.path.join(f'data/{args.data}/textual_embeddings_zeros_indexed', f)))

    non_missing_items = list(set(list(range(num_items_textual))).difference(missing_textual_indexed))
    propagated_textual_features = textual_features.clone()

    for idx in range(args.layers):
        print(f'[TEXTUAL] Propagation layer: {idx + 1}')
        propagated_textual_features = matmul(adj.to(device), propagated_textual_features.to(device))
        propagated_textual_features[non_missing_items] = textual_features[non_missing_items].to(device)

    for miss in missing_textual_indexed:
        np.save(os.path.join(output_textual, f'{miss}.npy'), propagated_textual_features[miss].detach().cpu().numpy())

visual_items = os.listdir(visual_folder)
textual_items = os.listdir(textual_folder)

for it in visual_items:
    shutil.copy(os.path.join(visual_folder, it), os.path.join(output_visual))

for it in textual_items:
    shutil.copy(os.path.join(textual_folder, it), os.path.join(output_textual))
