from abc import ABC

import torch
import numpy as np
import random
from torch_geometric.nn import LGConv
import torch_geometric


class MGCNModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 num_layers,
                 num_ui_layers,
                 learning_rate,
                 embed_k,
                 l_w,
                 cl_loss,
                 modalities,
                 top_k,
                 multimodal_features,
                 edge_index,
                 adj,
                 lr_sched,
                 random_seed,
                 name="MGCN",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.cl_loss = cl_loss
        self.modalities = modalities
        self.top_k = top_k
        self.n_layers = num_layers
        self.num_ui_layers = num_ui_layers
        self.adj = adj
        self.lr_sched = lr_sched

        # collaborative embeddings
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gu.to(self.device)
        self.Gi.to(self.device)

        self.R = edge_index

        # multimodal features
        self.projection_m = torch.nn.ModuleDict()
        self.Gim = torch.nn.ParameterDict()
        self.multimodal_features_shapes = [mf.shape[1] for mf in multimodal_features]
        self.Sim = dict()
        self.gate_m = torch.nn.ModuleDict()
        self.gate_prefer_m = torch.nn.ModuleDict()
        for m_id, m in enumerate(modalities):
            self.Gim[m] = torch.nn.Embedding.from_pretrained(
                torch.tensor(multimodal_features[m_id], dtype=torch.float32, device=self.device),
                freeze=False).weight
            self.Gim[m].to(self.device)
            self.projection_m[m] = torch.nn.Linear(in_features=self.multimodal_features_shapes[m_id],
                                                   out_features=self.embed_k)
            self.projection_m[m].to(self.device)
            current_sim = self.build_sim(self.Gim[m].detach())
            self.Sim[m] = self.build_knn_neighbourhood(current_sim)
            self.gate_m[m] = torch.nn.Sequential(
                torch.nn.Linear(self.embed_k, self.embed_k),
                torch.nn.Sigmoid()
            )
            self.gate_m[m].to(self.device)
            self.gate_prefer_m[m] = torch.nn.Sequential(
                torch.nn.Linear(self.embed_k, self.embed_k),
                torch.nn.Sigmoid()
            )
            self.gate_prefer_m[m].to(self.device)

        self.softmax = torch.nn.Softmax(dim=-1)
        self.softmax.to(self.device)

        self.query_common = torch.nn.Sequential(
            torch.nn.Linear(self.embed_k, self.embed_k),
            torch.nn.Tanh(),
            torch.nn.Linear(self.embed_k, 1, bias=False)
        )
        self.query_common.to(self.device)

        self.tau = 0.5

        propagation_network_list = []

        for _ in range(self.num_ui_layers):
            propagation_network_list.append((LGConv(normalize=False), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = self.set_lr_scheduler()

    @staticmethod
    def build_sim(context):
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        return sim

    @staticmethod
    def compute_normalized_laplacian(indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size)

    def build_knn_neighbourhood(self, sim):
        _, knn_ind = torch.topk(sim, self.top_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.top_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return self.compute_normalized_laplacian(indices, adj_size).to(self.device)

    def set_lr_scheduler(self):
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: self.lr_sched[0] ** (epoch / self.lr_sched[1]))
        return scheduler

    def propagate_embeddings(self, adj, train=False):
        # Behavior-Guided Purifier
        multimodal_item_embeddings = list()
        for m_id, m in enumerate(self.modalities):
            current_features = self.projection_m[m](self.Gim[m])
            multimodal_item_embeddings += [torch.multiply(self.Gi.weight, self.gate_m[m](current_features.to(self.device)))]

        # User-Item View
        item_embeds = self.Gi.weight
        user_embeds = self.Gu.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.num_ui_layers):
            side_embeddings = list(
                    self.propagation_network.children()
                )[i](ego_embeddings.to(self.device), adj.to(self.device))
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # Item-Item View
        multimodal_embeddings = list()
        for m_id, m in enumerate(self.modalities):
            for i in range(self.n_layers):
                multimodal_item_embeddings[m_id] = torch.sparse.mm(self.Sim[m], multimodal_item_embeddings[m_id])
            multimodal_user_embeddings = torch.sparse.mm(self.R.to(self.device), multimodal_item_embeddings[m_id])
            multimodal_embeddings += [torch.cat([multimodal_user_embeddings, multimodal_item_embeddings[m_id]], dim=0)]

        # Behavior-Aware Fuser
        att_common = torch.cat([self.query_common(m_emb) for m_emb in multimodal_embeddings], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeddings = 0
        multimodal_prefer = list()
        side_embeddings = 0
        for m_id, m in enumerate(self.modalities):
            common_embeddings += (weight_common[:, m_id].unsqueeze(dim=1) * multimodal_embeddings[m_id])
        for m_id, m in enumerate(self.modalities):
            sep_multimodal_embeddings = multimodal_embeddings[m_id] - common_embeddings
            multimodal_prefer += [self.gate_prefer_m[m](content_embeds)]
            sep_multimodal_embeddings = torch.multiply(multimodal_prefer[m_id], sep_multimodal_embeddings[m_id])
            side_embeddings += sep_multimodal_embeddings

        side_embeddings += common_embeddings
        side_embeddings = side_embeddings / (len(self.modalities) + 1)

        all_embeds = content_embeds + side_embeddings

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.num_users, self.num_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeddings, content_embeds

        return all_embeddings_users, all_embeddings_items

    def predict(self, u_embeddings, restore_item_e):
        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / len(users)

        maxi = torch.nn.functional.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.l_w * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = torch.nn.functional.normalize(view1, dim=1), torch.nn.functional.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def train_step(self, batch):
        users = batch[0]
        pos_items = batch[1]
        neg_items = batch[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.propagate_embeddings(self.adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.num_users, self.num_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.num_users, self.num_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)

        loss = batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
