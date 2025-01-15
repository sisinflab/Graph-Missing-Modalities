import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch_sparse import SparseTensor
import scipy.sparse as sp

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from torch_geometric.nn import GCNConv

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

class AutoEncoderVisual2Textual(nn.Module):
    def __init__(self):
        super(AutoEncoderVisual2Textual, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 768),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoEncoderTextual2Visual(nn.Module):
    def __init__(self):
        super(AutoEncoderTextual2Visual, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class GraphEncoderVisual2Textual(torch.nn.Module):
    def __init__(self):
        super(GraphEncoderVisual2Textual, self).__init__()
        self.conv1 = GCNConv(2048, 1024)
        self.conv2 = GCNConv(1024, 512)
        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 768),
            nn.Sigmoid()
        )

    def message_passing(self, x, edge_index):
        encoded = self.conv1(x, edge_index)
        encoded = torch.nn.functional.leaky_relu(encoded)
        encoded = self.conv2(encoded, edge_index)
        encoded = torch.nn.functional.leaky_relu(encoded)
        return encoded

    def forward(self, encoded):
        decoded = self.decoder(encoded)
        return decoded

class GraphEncoderTextual2Visual(torch.nn.Module):
    def __init__(self):
        super(GraphEncoderTextual2Visual, self).__init__()
        self.conv1 = GCNConv(768, 1024)
        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.Sigmoid()
        )

    def message_passing(self, x, edge_index):
        encoded = self.conv1(x, edge_index)
        encoded = torch.nn.functional.leaky_relu(encoded)
        return encoded

    def forward(self, encoded):
        decoded = self.decoder(encoded)
        return decoded


def get_item_item(num_items):
    # compute item_item matrix
    user_item = sp.coo_matrix(([1.0] * len(train), (train[0].tolist(), train[1].tolist())),
                              shape=(train[0].nunique(), num_items), dtype=np.float32)
    item_item = user_item.transpose().dot(user_item).toarray()
    knn_val, knn_ind = torch.topk(torch.tensor(item_item, device=device), args.top_k, dim=-1)
    items_cols = torch.flatten(knn_ind).to(device)
    ir = torch.tensor(list(range(item_item.shape[0])), dtype=torch.int64, device=device)
    items_rows = torch.repeat_interleave(ir, args.top_k).to(device)
    final_adj = SparseTensor(row=items_rows,
                             col=items_cols,
                             value=torch.tensor([1.0] * items_rows.shape[0], device=device),
                             sparse_sizes=(item_item.shape[0], item_item.shape[0]))
    return final_adj


parser = argparse.ArgumentParser(description="Run imputation.")
parser.add_argument('--data', type=str, default='Office_Products')
parser.add_argument('--method', type=str, default='gae')
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


if args.method == 'gae':
    if not os.path.exists(output_visual + f'_{args.top_k}_indexed'):
        os.makedirs(output_visual + f'_{args.top_k}_indexed')
    if not os.path.exists(output_textual + f'_{args.top_k}_indexed'):
        os.makedirs(output_textual + f'_{args.top_k}_indexed')
else:
    if not os.path.exists(output_visual):
        os.makedirs(output_visual)
    if not os.path.exists(output_textual):
        os.makedirs(output_textual)

if args.method == 'ae':
    visual_shape = np.load(os.path.join(visual_folder, os.listdir(visual_folder)[0])).shape
    textual_shape = np.load(os.path.join(textual_folder, os.listdir(textual_folder)[0])).shape
    num_items_visual = len(os.listdir(visual_folder))
    num_items_textual = len(os.listdir(textual_folder))

    visual_items = os.listdir(visual_folder)
    textual_items = os.listdir(textual_folder)

    all_present = [x for x in textual_items if x in visual_items]

    visual_features = np.empty((len(all_present), visual_shape[-1]), dtype=np.float32) if num_items_visual else None
    textual_features = np.empty((len(all_present), textual_shape[-1]), dtype=np.float32) if num_items_textual else None

    for idx, it in enumerate(all_present):
        visual_features[idx, :] = np.load(os.path.join(visual_folder, it))
        textual_features[idx, :] = np.load(os.path.join(textual_folder, it))

    # train AE to reconstruct textual through visual
    model_impute_textual = AutoEncoderVisual2Textual()
    model_impute_textual.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adagrad(model_impute_textual.parameters(), lr=0.01)

    visual_normalized = torch.from_numpy(visual_features)
    visual_normalized = visual_normalized - torch.from_numpy(visual_features).min(0, keepdim=True)[0]
    visual_normalized = visual_normalized / (torch.from_numpy(visual_features).max(0, keepdim=True)[0] - torch.from_numpy(visual_features).min(0, keepdim=True)[0])
    visual_normalized.to(device)

    textual_normalized = torch.from_numpy(textual_features)
    textual_normalized = textual_normalized - torch.from_numpy(textual_features).min(0, keepdim=True)[0]
    textual_normalized = textual_normalized / (torch.from_numpy(textual_features).max(0, keepdim=True)[0] - torch.from_numpy(textual_features).min(0, keepdim=True)[0])
    textual_normalized.to(device)

    dataset = TensorDataset(visual_normalized,
                            textual_normalized)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    cumulative_loss = 0
    for epoch in range(1000):
        for batch in dataloader:
            inputs, targets = batch
            outputs = model_impute_textual(inputs.to(device))
            loss = criterion(outputs.to(device), targets.to(device))
            cumulative_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {cumulative_loss / 100:.4f}")
            cumulative_loss = 0

    # train AE to reconstruct visual through textual
    model_impute_visual = AutoEncoderTextual2Visual()
    model_impute_visual.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adagrad(model_impute_visual.parameters(), lr=0.01)

    dataset = TensorDataset(textual_normalized,
                            visual_normalized)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    cumulative_loss = 0
    for epoch in range(1000):
        for batch in dataloader:
            inputs, targets = batch
            outputs = model_impute_visual(inputs.to(device))
            loss = criterion(outputs.to(device), targets.to(device))
            loss += (0.0001 * torch.sum(torch.abs(outputs)))
            cumulative_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {cumulative_loss / 100:.4f}")
            cumulative_loss = 0

    # impute missing visual from textual
    for miss in missing_visual:
        if miss in missing_textual:
            np.save(os.path.join(output_visual, f'{miss}.npy'), np.zeros(visual_shape, dtype=np.float32))
        else:
            textual_input = torch.from_numpy(np.load(os.path.join(textual_folder, f'{miss}.npy')))
            textual_input = textual_input - torch.from_numpy(textual_features).min(0, keepdim=True)[0]
            textual_input = textual_input / (torch.from_numpy(textual_features).max(0, keepdim=True)[0] -
                                             torch.from_numpy(textual_features).min(0, keepdim=True)[0])
            output = model_impute_visual(textual_input.to(device))
            output = (output * (torch.from_numpy(visual_features).to(device).max(0, keepdim=True)[0] -
                                torch.from_numpy(visual_features).to(device).min(0, keepdim=True)[0])) + \
                     torch.from_numpy(visual_features).to(device).min(0, keepdim=True)[0]
            np.save(os.path.join(output_visual, f'{miss}.npy'), output.detach().cpu().numpy())

    # impute missing textual from visual
    for miss in missing_textual:
        if miss in missing_visual:
            np.save(os.path.join(output_textual, f'{miss}.npy'), np.zeros(textual_shape, dtype=np.float32))
        else:
            visual_input = torch.from_numpy(np.load(os.path.join(visual_folder, f'{miss}.npy')))
            visual_input = visual_input - torch.from_numpy(visual_features).min(0, keepdim=True)[0]
            visual_input = visual_input / (torch.from_numpy(visual_features).max(0, keepdim=True)[0] -
                                             torch.from_numpy(visual_features).min(0, keepdim=True)[0])
            output = model_impute_textual(visual_input.to(device))
            output = (output * (torch.from_numpy(textual_features).to(device).max(0, keepdim=True)[0] -
                                torch.from_numpy(textual_features).to(device).min(0, keepdim=True)[0])) + \
                     torch.from_numpy(textual_features).to(device).min(0, keepdim=True)[0]
            np.save(os.path.join(output_textual, f'{miss}.npy'), output.detach().cpu().numpy())

elif args.method == 'gae':
    visual_folder = f'data/{args.data}/visual_embeddings_indexed'
    textual_folder = f'data/{args.data}/textual_embeddings_indexed'

    visual_shape = np.load(os.path.join(visual_folder, os.listdir(visual_folder)[0])).shape
    textual_shape = np.load(os.path.join(textual_folder, os.listdir(textual_folder)[0])).shape

    num_items_visual = len(os.listdir(visual_folder))
    num_items_textual = len(os.listdir(textual_folder))

    visual_items = os.listdir(visual_folder)
    textual_items = os.listdir(textual_folder)

    all_present = [int(x.split('.npy')[0]) for x in textual_items if x in visual_items]

    visual_features = np.empty((len(all_present), visual_shape[-1]), dtype=np.float32) if num_items_visual else None
    textual_features = np.empty((len(all_present), textual_shape[-1]), dtype=np.float32) if num_items_textual else None

    inner_map_item = {}

    for idx, it in enumerate(all_present):
        inner_map_item[it] = idx
        visual_features[idx, :] = np.load(os.path.join(visual_folder, f'{it}.npy'))
        textual_features[idx, :] = np.load(os.path.join(textual_folder, f'{it}.npy'))

    output_visual = f'data/{args.data}/visual_embeddings_{args.method}_{args.top_k}_indexed'
    output_textual = f'data/{args.data}/textual_embeddings_{args.method}_{args.top_k}_indexed'

    try:
        train = pd.read_csv(f'data/{args.data}/train_indexed.tsv', sep='\t', header=None)
        train = train[train[1].isin(all_present)]
    except FileNotFoundError:
        print('Before imputing through gae, split the dataset into train/val/test!')
        exit()

    inner_map_user = {u: idx for idx, u in enumerate(train[0].unique())}

    train[0] = train[0].map(inner_map_user)
    train[1] = train[1].map(inner_map_item)

    adj = get_item_item(num_items=len(all_present))

    try:
        missing_visual_indexed = pd.read_csv(os.path.join(f'data/{args.data}', 'missing_visual_indexed.tsv'), sep='\t',
                                             header=None)
        missing_visual_indexed = set(missing_visual_indexed[0].tolist())
    except (pd.errors.EmptyDataError, FileNotFoundError):
        missing_visual_indexed = set()

    try:
        missing_textual_indexed = pd.read_csv(os.path.join(f'data/{args.data}', 'missing_textual_indexed.tsv'),
                                              sep='\t', header=None)
        missing_textual_indexed = set(missing_textual_indexed[0].tolist())
    except (pd.errors.EmptyDataError, FileNotFoundError):
        missing_textual_indexed = set()

    # train GAE to reconstruct textual through visual
    model_impute_textual = GraphEncoderVisual2Textual()
    model_impute_textual.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_impute_textual.parameters(), lr=1e-4)

    visual_normalized = torch.from_numpy(visual_features)
    visual_normalized = visual_normalized - torch.from_numpy(visual_features).min(0, keepdim=True)[0]
    visual_normalized = visual_normalized / (torch.from_numpy(visual_features).max(0, keepdim=True)[0] -
                                             torch.from_numpy(visual_features).min(0, keepdim=True)[0])
    visual_normalized.to(device)

    textual_normalized = torch.from_numpy(textual_features)
    textual_normalized = textual_normalized - torch.from_numpy(textual_features).min(0, keepdim=True)[0]
    textual_normalized = textual_normalized / (torch.from_numpy(textual_features).max(0, keepdim=True)[0] -
                                               torch.from_numpy(textual_features).min(0, keepdim=True)[0])
    textual_normalized.to(device)

    items = torch.arange(visual_normalized.shape[0])

    dataset = TensorDataset(items, textual_normalized)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    cumulative_loss = 0
    for epoch in range(1000):
        for batch in dataloader:
            features = model_impute_textual.message_passing(visual_normalized.to(device), adj.to(device))
            items_id, targets = batch
            outputs = model_impute_textual.forward(features[items_id].to(device))
            loss = criterion(outputs.to(device), targets.to(device))
            cumulative_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {cumulative_loss / 100:.4f}")
            cumulative_loss = 0

    # train GAE to reconstruct visual through textual
    model_impute_visual = GraphEncoderTextual2Visual()
    model_impute_visual.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_impute_visual.parameters(), lr=1e-4)

    dataset = TensorDataset(items, visual_normalized)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    cumulative_loss = 0
    for epoch in range(1000):
        for batch in dataloader:
            features = model_impute_visual.message_passing(textual_normalized.to(device), adj.to(device))
            items_id, targets = batch
            outputs = model_impute_visual.forward(features[items_id].to(device))
            loss = criterion(outputs.to(device), targets.to(device))
            cumulative_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {cumulative_loss / 100:.4f}")
            cumulative_loss = 0

    # impute missing visual from textual
    for miss in missing_visual_indexed:
        if miss in missing_textual_indexed:
            np.save(os.path.join(output_visual, f'{miss}.npy'), np.zeros(visual_shape, dtype=np.float32))
        else:
            current_all_present = all_present + [miss]
            inner_map_item = {}
            for idx, it in enumerate(current_all_present):
                inner_map_item[it] = idx
            train = pd.read_csv(f'data/{args.data}/train_indexed.tsv', sep='\t', header=None)
            train = train[train[1].isin(current_all_present)]

            inner_map_user = {u: idx for idx, u in enumerate(train[0].unique())}

            train[0] = train[0].map(inner_map_user)
            train[1] = train[1].map(inner_map_item)

            adj = get_item_item(num_items=len(current_all_present))

            textual_input = torch.from_numpy(np.load(os.path.join(textual_folder, f'{miss}.npy')))
            textual_input = textual_input - torch.from_numpy(textual_features).min(0, keepdim=True)[0]
            textual_input = textual_input / (torch.from_numpy(textual_features).max(0, keepdim=True)[0] -
                                             torch.from_numpy(textual_features).min(0, keepdim=True)[0])
            features = model_impute_visual.message_passing(
                torch.concat((textual_normalized.to(device), textual_input.to(device)), dim=0).to(device),
                adj.to(device)
            )
            output = model_impute_visual.forward(features[-1].to(device))
            output = (output * (torch.from_numpy(visual_features).to(device).max(0, keepdim=True)[0] -
                                torch.from_numpy(visual_features).to(device).min(0, keepdim=True)[0])) + \
                     torch.from_numpy(visual_features).to(device).min(0, keepdim=True)[0]
            np.save(os.path.join(output_visual, f'{miss}.npy'), output.detach().cpu().numpy())

    # impute missing textual from visual
    for miss in missing_textual_indexed:
        if miss in missing_visual_indexed:
            np.save(os.path.join(output_textual, f'{miss}.npy'), np.zeros(textual_shape, dtype=np.float32))
        else:
            current_all_present = all_present + [miss]
            inner_map_item = {}
            for idx, it in enumerate(current_all_present):
                inner_map_item[it] = idx
            train = pd.read_csv(f'data/{args.data}/train_indexed.tsv', sep='\t', header=None)
            train = train[train[1].isin(current_all_present)]

            inner_map_user = {u: idx for idx, u in enumerate(train[0].unique())}

            train[0] = train[0].map(inner_map_user)
            train[1] = train[1].map(inner_map_item)

            adj = get_item_item(num_items=len(current_all_present))

            visual_input = torch.from_numpy(np.load(os.path.join(visual_folder, f'{miss}.npy')))
            visual_input = visual_input - torch.from_numpy(visual_features).min(0, keepdim=True)[0]
            visual_input = visual_input / (torch.from_numpy(visual_features).max(0, keepdim=True)[0] -
                                           torch.from_numpy(visual_features).min(0, keepdim=True)[0])

            features = model_impute_textual.message_passing(
                torch.concat((visual_normalized.to(device), visual_input.to(device)), dim=0).to(device),
                adj.to(device)
            )
            output = model_impute_textual.forward(features[-1].to(device))
            output = (output * (torch.from_numpy(textual_features).to(device).max(0, keepdim=True)[0] -
                                torch.from_numpy(textual_features).to(device).min(0, keepdim=True)[0])) + \
                     torch.from_numpy(textual_features).to(device).min(0, keepdim=True)[0]
            np.save(os.path.join(output_textual, f'{miss}.npy'), output.detach().cpu().numpy())
