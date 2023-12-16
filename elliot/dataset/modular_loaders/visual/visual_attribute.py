import typing as t
from ast import literal_eval
import os
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace
from torch_sparse import SparseTensor, mul, sum, fill_diag, matmul

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class VisualAttribute(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.visual_feature_folder_path = getattr(ns, "visual_features", None)
        self.masked_items_path = getattr(ns, "masked_items_path", None)
        self.strategy = getattr(ns, "strategy", None)
        self.feat_prop = getattr(ns, "feat_prop", None)
        self.prop_layers = getattr(ns, "prop_layers", None)
        self.visual_pca_feature_folder_path = getattr(ns, "visual_pca_features", None)
        self.visual_feat_map_feature_folder_path = getattr(ns, "visual_feat_map_features", None)
        self.images_folder_path = getattr(ns, "images_src_folder", None)

        self.item_mapping = {}
        self.visual_features_shape = None
        self.visual_pca_features_shape = None
        self.visual_feat_map_features_shape = None
        self.image_size_tuple = getattr(ns, "output_image_size", None)
        if self.image_size_tuple:
            self.image_size_tuple = literal_eval(self.image_size_tuple)

        inner_items = self.check_items_in_folder()

        self.users = users
        self.items = items & inner_items

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items

    @staticmethod
    def compute_normalized_laplacian(adj, norm):
        adj = fill_diag(adj, 0.)
        deg = sum(adj, dim=-1)
        deg_inv_sqrt = deg.pow_(-norm)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = "VisualAttribute"
        ns.object = self
        ns.visual_feature_folder_path = self.visual_feature_folder_path
        ns.visual_pca_feature_folder_path = self.visual_pca_feature_folder_path
        ns.visual_feat_map_feature_folder_path = self.visual_feat_map_feature_folder_path
        ns.masked_items_path = self.masked_items_path
        ns.strategy = self.strategy
        ns.feat_prop = self.feat_prop
        ns.prop_layers = self.prop_layers
        ns.images_folder_path = self.images_folder_path

        ns.item_mapping = self.item_mapping

        ns.visual_features_shape = self.visual_features_shape
        ns.visual_pca_features_shape = self.visual_pca_features_shape
        ns.visual_feat_map_features_shape = self.visual_feat_map_features_shape
        ns.image_size_tuple = self.image_size_tuple

        return ns

    def check_items_in_folder(self) -> t.Set[int]:
        items = set()
        if self.visual_feature_folder_path:
            items_folder = os.listdir(self.visual_feature_folder_path)
            items = items.union(set([int(f.split('.')[0]) for f in items_folder]))
            self.visual_features_shape = np.load(os.path.join(self.visual_feature_folder_path,
                                                              items_folder[0])).shape[0]
        if self.visual_pca_feature_folder_path:
            items_folder = os.listdir(self.visual_feature_folder_path)
            items = items.union(set([int(f.split('.')[0]) for f in items_folder]))
            self.visual_pca_features_shape = np.load(os.path.join(self.visual_pca_feature_folder_path,
                                                                  items_folder[0])).shape[0]
        if self.visual_feat_map_feature_folder_path:
            items_folder = os.listdir(self.visual_feature_folder_path)
            items = items.union(set([int(f.split('.')[0]) for f in items_folder]))
            self.visual_feat_map_features_shape = np.load(os.path.join(self.visual_feat_map_feature_folder_path,
                                                                       items_folder[0])).shape
        if self.images_folder_path:
            items_folder = os.listdir(self.visual_feature_folder_path)
            items = items.union(set([int(f.split('.')[0]) for f in items_folder]))

        if items:
            self.item_mapping = {item: val for val, item in enumerate(items)}
        return items

    def get_all_features(self, item_item):
        all_features = self.get_all_visual_features()
        if self.masked_items_path:
            masked_items = pd.read_csv(self.masked_items_path, sep='\t', header=None)[0].tolist()
            if self.strategy == 'zeros':
                all_features[masked_items] = np.zeros((1, all_features.shape[-1]))
            elif self.strategy == 'mean':
                mask = np.ones(all_features.shape[0], dtype=bool)
                mask[masked_items] = False
                result = all_features[mask]
                mean_ = result.mean(axis=0)
                all_features[masked_items] = mean_
            elif self.strategy == 'random':
                all_features[masked_items] = np.random.rand(len(masked_items), all_features.shape[1])
            elif self.strategy == 'feat_prop':
                if self.feat_prop == 'co':
                    item_item = item_item.toarray()
                    # get non masked items
                    non_masked_items = list(set(list(range(all_features.shape[0]))).difference(masked_items))
                    # binarize adjacency matrix
                    # item_item[item_item >= 1] = 1.0
                    # set zeros as initialization
                    all_features[masked_items] = np.zeros((1, all_features.shape[-1]))
                    # get sparse adjacency matrix
                    knn_val, knn_ind = torch.topk(torch.tensor(item_item), 20, dim=-1)
                    items_cols = torch.flatten(knn_ind)
                    ir = torch.tensor(list(range(item_item.shape[0])), dtype=torch.int64)
                    items_rows = torch.repeat_interleave(ir, 20)
                    # row, col = item_item.nonzero()
                    # edge_index = np.array([row, col])
                    # edge_index = torch.tensor(edge_index, dtype=torch.int64)
                    # adj = SparseTensor(row=edge_index[0],
                    #                    col=edge_index[1],
                    #                    sparse_sizes=(self.n_items, self.n_items))
                    adj = SparseTensor(row=items_rows,
                                       col=items_cols,
                                       value=torch.tensor([1.0] * items_rows.shape[0]),
                                       sparse_sizes=(item_item.shape[0], item_item.shape[0]))
                    # normalize adjacency matrix
                    adj = self.compute_normalized_laplacian(adj, 0.5)
                    # feature propagation
                    propagated_features = torch.tensor(all_features)
                    for _ in range(self.prop_layers):
                        propagated_features = matmul(adj, propagated_features)
                        propagated_features[non_masked_items] = torch.tensor(all_features[non_masked_items])
                    all_features[masked_items] = propagated_features[masked_items].detach().cpu().numpy()
                elif self.feat_prop == 'rev':
                    pass
                else:
                    raise NotImplementedError('This aggregation has not been implemented yet!')
            else:
                raise NotImplementedError('This strategy has not been implemented yet!')
        return all_features

    def get_all_visual_features(self):
        all_features = np.empty((len(self.items), self.visual_features_shape))
        if self.visual_feature_folder_path:
            for key, value in self.item_mapping.items():
                all_features[value] = np.load(self.visual_feature_folder_path + '/' + str(key) + '.npy')
        return all_features

    def get_all_visual_pca_features(self):
        all_features = np.empty((len(self.items), self.visual_pca_features_shape))
        if self.visual_pca_feature_folder_path:
            for key, value in self.item_mapping.items():
                all_features[value] = np.load(self.visual_pca_feature_folder_path + '/' + str(key) + '.npy')
        return all_features

    def get_all_visual_feat_map_features(self):
        all_features = np.empty((len(self.items), self.visual_feat_map_features_shape))
        if self.visual_feat_map_feature_folder_path:
            for key, value in self.item_mapping.items():
                all_features[value] = np.load(self.visual_feat_map_feature_folder_path + '/' + str(key) + '.npy')
        return all_features
