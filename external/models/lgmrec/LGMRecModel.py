from abc import ABC

import torch
import numpy as np
import random
from torch_geometric.nn import LGConv
from .HGNNLayer import HGNNLayer
import torch_geometric


class LGMRecModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 num_hyper_layers,
                 num_ui_layers,
                 num_mm_layers,
                 hyper_num,
                 learning_rate,
                 embed_k,
                 embed_k_multimod,
                 cf_model,
                 l_w,
                 c_l,
                 alpha,
                 modalities,
                 multimodal_features,
                 edge_index,
                 norm_adj,
                 num_inters,
                 keep_rate,
                 lr_sched,
                 random_seed,
                 name="LGMRec",
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
        self.embed_k_multimod = embed_k_multimod
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.c_l = c_l
        self.modalities = modalities
        self.num_hyper_layers = num_hyper_layers
        self.num_mm_layers = num_mm_layers
        self.num_ui_layers = num_ui_layers
        self.R = edge_index
        self.cf_model = cf_model
        self.hyper_num = hyper_num
        self.keep_rate = keep_rate
        self.alpha = alpha
        self.lr_sched = lr_sched
        self.tau = 0.2
        self.num_inters = num_inters
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)
        self.norm_adj = norm_adj

        self.n_nodes = self.num_users + self.num_items
        self.hgnnLayer = HGNNLayer(self.num_hyper_layers)

        # collaborative embeddings
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gu.to(self.device)
        self.Gi.to(self.device)

        self.drop = torch.nn.Dropout(p=1 - self.keep_rate)

        # multimodal features
        self.Gim = list()
        self.item_multimodal_trs = torch.nn.ParameterDict()
        self.multimodal_hyper = torch.nn.ParameterDict()
        self.multimodal_features_shapes = [mf.shape[1] for mf in multimodal_features]
        for m_id, m in enumerate(modalities):
            self.Gim += [torch.nn.Embedding.from_pretrained(
                torch.tensor(multimodal_features[m_id], dtype=torch.float32, device=self.device),
                freeze=True)]
            self.Gim[m_id].to(self.device)
            self.item_multimodal_trs[m] = torch.nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.zeros(self.multimodal_features_shapes[m_id],
                                                          self.embed_k_multimod))
            )
            self.item_multimodal_trs[m].to(self.device)
            self.multimodal_hyper[m] = torch.nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.zeros(self.multimodal_features_shapes[m_id], self.hyper_num))
            )
            self.multimodal_hyper[m].to(self.device)

        propagation_network_list = []

        for _ in range(self.num_ui_layers):
            propagation_network_list.append((LGConv(normalize=False), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)

        propagation_network_list = []

        for _ in range(self.num_mm_layers):
            propagation_network_list.append((LGConv(normalize=False), 'x, edge_index -> x'))

        self.propagation_network_mm = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network_mm.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = self.set_lr_scheduler()

    def set_lr_scheduler(self):
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: self.lr_sched[0] ** (epoch / self.lr_sched[1]))
        return scheduler

    # collaborative graph embedding
    def cge(self):
        if self.cf_model == 'mf':
            cge_embs = torch.cat((self.Gu.weight, self.Gi.weight), dim=0)
        if self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.Gu.weight, self.Gi.weight), dim=0)
            cge_embs = [ego_embeddings]
            for layer in range(self.num_ui_layers):
                ego_embeddings = list(
                        self.propagation_network.children()
                    )[layer](ego_embeddings.to(self.device), self.norm_adj.to(self.device))
                cge_embs += [ego_embeddings]
            cge_embs = torch.stack(cge_embs, dim=1)
            cge_embs = cge_embs.mean(dim=1, keepdim=False)
        return cge_embs

    # modality graph embedding
    def mge(self, multimodal_features, modality):
        item_feats = torch.mm(multimodal_features.weight, self.item_multimodal_trs[modality].to(self.device))
        user_feats = torch.sparse.mm(self.R.to(self.device), item_feats) * torch.unsqueeze(self.num_inters[:self.num_users], -1)
        # user_feats = self.user_embedding.weight
        mge_feats = torch.concat([user_feats, item_feats], dim=0)
        for layer in range(self.num_mm_layers):
            mge_feats = list(
                        self.propagation_network_mm.children()
                    )[layer](mge_feats.to(self.device), self.norm_adj.to(self.device))
        return mge_feats

    def propagate_embeddings(self):
        # hyperedge dependencies constructing

        i_hyper, u_hyper = list(), list()

        for m_id, m in enumerate(self.modalities):
            i_hyper += [torch.mm(self.Gim[m_id].weight, self.multimodal_hyper[m].to(self.device))]
            u_hyper += [torch.mm(self.R.to(self.device), i_hyper[m_id])]
            i_hyper[m_id] = torch.nn.functional.gumbel_softmax(i_hyper[m_id], self.tau, dim=1, hard=False)
            u_hyper[m_id] = torch.nn.functional.gumbel_softmax(u_hyper[m_id], self.tau, dim=1, hard=False)

        # CGE: collaborative graph embedding
        cge_embs = self.cge()

        # MGE: modal graph embedding
        multimodal_features = list()
        for m_id, m in enumerate(self.modalities):
            multimodal_features += [self.mge(self.Gim[m_id], m)]

        # local embeddings = collaborative-related embedding + modality-related embedding
        mge_embs = 0
        for m_id, m in enumerate(self.modalities):
            mge_embs += torch.nn.functional.normalize(multimodal_features[m_id])
        lge_embs = cge_embs + mge_embs

        # GHE: global hypergraph embedding
        u_i_multimodal_hyper_embeds = list()
        a_multimodal_hyper_embs = list()
        for m_id, m in enumerate(self.modalities):
            u_hyper_embs, i_hyper_embs = self.hgnnLayer(self.drop(i_hyper[m_id]),
                                                        self.drop(u_hyper[m_id]),
                                                        cge_embs[self.num_users:])
            u_i_multimodal_hyper_embeds += [u_hyper_embs]
            u_i_multimodal_hyper_embeds += [i_hyper_embs]
            a_multimodal_hyper_embs += [torch.concat([u_hyper_embs, i_hyper_embs], dim=0)]
        ghe_embs = torch.sum(torch.stack(a_multimodal_hyper_embs, dim=0), dim=0)

        # local embeddings + alpha * global embeddings
        all_embs = lge_embs + self.alpha * torch.nn.functional.normalize(ghe_embs)

        u_embs, i_embs = torch.split(all_embs, [self.num_users, self.num_items], dim=0)

        return u_embs, i_embs, u_i_multimodal_hyper_embeds

    @staticmethod
    def bpr_loss(users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        bpr_loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
        return bpr_loss

    def ssl_triple_loss(self, emb1, emb2, all_emb):
        norm_emb1 = torch.nn.functional.normalize(emb1)
        norm_emb2 = torch.nn.functional.normalize(emb2)
        norm_all_emb = torch.nn.functional.normalize(all_emb)
        pos_score = torch.exp(torch.mul(norm_emb1, norm_emb2).sum(dim=1) / self.tau)
        ttl_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb.T) / self.tau).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss

    @staticmethod
    def reg_loss(*embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def predict(self, user_embs, item_embs):
        scores = torch.matmul(user_embs, item_embs.T)
        return scores

    def train_step(self, interaction):
        ua_embeddings, ia_embeddings, hyper_embeddings = self.propagate_embeddings()

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        [uv_embs, iv_embs, ut_embs, it_embs] = hyper_embeddings
        batch_hcl_loss = self.ssl_triple_loss(uv_embs[users], ut_embs[users], ut_embs) + self.ssl_triple_loss(
            iv_embs[pos_items], it_embs[pos_items], it_embs)

        batch_reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        loss = batch_bpr_loss + self.c_l * batch_hcl_loss + self.l_w * batch_reg_loss
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
