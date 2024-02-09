from datetime import datetime
import math
import os
import random
import sys
from time import time
from tqdm import tqdm
import dgl
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import visdom
import pandas as pd
from torch_sparse import SparseTensor, mul, sum, fill_diag, matmul

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse
from torch import autograd

import copy

from utility.parser import parse_args
from Models import MMSSL, Discriminator
from utility.batch_test import *
from utility.logging import Logger
from utility.norm import build_sim, build_knn_normalized_graph
from torch.utils.tensorboard import SummaryWriter

args = parse_args()


class Trainer(object):
    def __init__(self, data_config):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task_name = "%s_%s_%s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.dataset, args.cf_model,)
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.mess_dropout = eval(args.mess_dropout)
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.image_feats = np.load(args.data_path + '{}/image_feat.npy'.format(args.dataset))
        self.text_feats = np.load(args.data_path + '{}/text_feat.npy'.format(args.dataset))
        self.image_feat_dim = self.image_feats.shape[-1]
        self.text_feat_dim = self.text_feats.shape[-1]
        self.ui_graph = self.ui_graph_raw = pickle.load(open(args.data_path + args.dataset + '/train_mat', 'rb'))
        self.image_ui_graph_tmp = self.text_ui_graph_tmp = torch.tensor(self.ui_graph_raw.todense(), device=self.device)
        self.image_iu_graph_tmp = self.text_iu_graph_tmp = torch.tensor(self.ui_graph_raw.T.todense(),
                                                                        device=self.device)
        self.image_ui_index = {'x': [], 'y': []}
        self.text_ui_index = {'x': [], 'y': []}
        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]

        if args.masked_items_image:
            masked_items_image = pd.read_csv(args.masked_items_image, sep='\t', header=None)[0].tolist()
        else:
            raise FileNotFoundError('File for masked items not found!')

        if args.masked_items_text:
            masked_items_text = pd.read_csv(args.masked_items_text, sep='\t', header=None)[0].tolist()
        else:
            raise FileNotFoundError('File for masked items not found!')

        if args.missing_features:
            if args.strategy == 'zeros':
                self.image_feats[masked_items_image] = np.zeros((1, self.image_feat_dim))
                self.text_feats[masked_items_text] = np.zeros((1, self.text_feat_dim))
            elif args.strategy == 'mean':
                mask = np.ones(self.n_items, dtype=bool)
                mask[masked_items_image] = False
                result = self.image_feats[mask]
                mean_ = result.mean(axis=0)
                self.image_feats[masked_items_image] = mean_

                mask = np.ones(self.n_items, dtype=bool)
                mask[masked_items_text] = False
                result = self.text_feats[mask]
                mean_ = result.mean(axis=0)
                self.text_feats[masked_items_text] = mean_
            elif args.strategy == 'random':
                self.image_feats[masked_items_image] = np.random.rand(len(masked_items_image), self.image_feat_dim)
                self.text_feats[masked_items_text] = np.random.rand(len(masked_items_text), self.text_feat_dim)
            elif args.strategy == 'feat_prop':
                if args.feat_prop == 'co':
                    item_item = self.ui_graph.transpose().dot(self.ui_graph).toarray()
                    # get non masked items
                    non_masked_items = list(set(list(range(self.n_items))).difference(masked_items_image))
                    # binarize adjacency matrix
                    # item_item[item_item >= 1] = 1.0
                    # set zeros as initialization
                    self.image_feats[masked_items_image] = np.zeros((1, self.image_feat_dim))
                    # get sparse adjacency matrix
                    knn_val, knn_ind = torch.topk(torch.tensor(item_item, device=self.device), 20, dim=-1)
                    items_cols = torch.flatten(knn_ind).to(self.device)
                    ir = torch.tensor(list(range(item_item.shape[0])), dtype=torch.int64, device=self.device)
                    items_rows = torch.repeat_interleave(ir, 20).to(self.device)
                    # row, col = item_item.nonzero()
                    # edge_index = np.array([row, col])
                    # edge_index = torch.tensor(edge_index, dtype=torch.int64)
                    # adj = SparseTensor(row=edge_index[0],
                    #                    col=edge_index[1],
                    #                    sparse_sizes=(self.n_items, self.n_items))
                    adj = SparseTensor(row=items_rows,
                                       col=items_cols,
                                       value=torch.tensor([1.0] * items_rows.shape[0], device=self.device),
                                       sparse_sizes=(item_item.shape[0], item_item.shape[0]))
                    # normalize adjacency matrix
                    adj = self.compute_normalized_laplacian(adj, 0.5)
                    # feature propagation

                    propagated_features = torch.tensor(self.image_feats, device=self.device)
                    for idx in range(args.prop_layers):
                        print(f'Propagation layer: {idx + 1}')
                        propagated_features = matmul(adj.to(self.device), propagated_features.to(self.device))
                        propagated_features[non_masked_items] = torch.tensor(self.image_feats[non_masked_items],
                                                                             device=self.device)
                    self.image_feats[masked_items_image] = propagated_features[
                        masked_items_image].detach().cpu().numpy()

                    item_item = self.ui_graph.transpose().dot(self.ui_graph).toarray()
                    # get non masked items
                    non_masked_items = list(set(list(range(self.n_items))).difference(masked_items_text))
                    # binarize adjacency matrix
                    # item_item[item_item >= 1] = 1.0
                    # set zeros as initialization
                    self.text_feats[masked_items_text] = np.zeros((1, self.text_feat_dim))
                    # get sparse adjacency matrix
                    knn_val, knn_ind = torch.topk(torch.tensor(item_item, device=self.device), 20, dim=-1)
                    items_cols = torch.flatten(knn_ind).to(self.device)
                    ir = torch.tensor(list(range(item_item.shape[0])), dtype=torch.int64, device=self.device)
                    items_rows = torch.repeat_interleave(ir, 20).to(self.device)
                    # row, col = item_item.nonzero()
                    # edge_index = np.array([row, col])
                    # edge_index = torch.tensor(edge_index, dtype=torch.int64)
                    # adj = SparseTensor(row=edge_index[0],
                    #                    col=edge_index[1],
                    #                    sparse_sizes=(self.n_items, self.n_items))
                    adj = SparseTensor(row=items_rows,
                                       col=items_cols,
                                       value=torch.tensor([1.0] * items_rows.shape[0], device=self.device),
                                       sparse_sizes=(item_item.shape[0], item_item.shape[0]))
                    # normalize adjacency matrix
                    adj = self.compute_normalized_laplacian(adj, 0.5)
                    # feature propagation
                    propagated_features = torch.tensor(self.text_feats, device=self.device)
                    for idx in range(args.prop_layers):
                        print(f'Propagation layer: {idx + 1}')
                        propagated_features = matmul(adj.to(self.device), propagated_features.to(self.device))
                        propagated_features[non_masked_items] = torch.tensor(self.text_feats[non_masked_items],
                                                                             device=self.device)
                    self.text_feats[masked_items_text] = propagated_features[masked_items_text].detach().cpu().numpy()
                elif args.feat_prop == 'rev':
                    pass
                else:
                    raise NotImplementedError('This aggregation has not been implemented yet!')
            else:
                raise NotImplementedError('This strategy has not been implemented yet!')

        self.iu_graph = self.ui_graph.T
        self.ui_graph = self.matrix_to_tensor(self.csr_norm(self.ui_graph, mean_flag=True))
        self.iu_graph = self.matrix_to_tensor(self.csr_norm(self.iu_graph, mean_flag=True))
        self.image_ui_graph = self.text_ui_graph = self.ui_graph
        self.image_iu_graph = self.text_iu_graph = self.iu_graph
        self.model = MMSSL(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout,
                           self.image_feats, self.text_feats)
        self.model = self.model.to(self.device)
        self.D = Discriminator(self.n_items).to(self.device)
        self.D.apply(self.weights_init)
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.D_lr, betas=(0.5, 0.9))

        self.optimizer_D = optim.AdamW(
            [
                {'params': self.model.parameters()},
            ]
            , lr=self.lr)
        self.scheduler_D = self.set_lr_scheduler()

    @staticmethod
    def compute_normalized_laplacian(adj, norm):
        adj = fill_diag(adj, 0.)
        deg = sum(adj, dim=-1)
        deg_inv_sqrt = deg.pow_(-norm)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler_D = optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=fac)
        return scheduler_D

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum + 1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum + 1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)

        if mean_flag == False:
            return rowsum_diag * csr_mat * colsum_diag
        else:
            return rowsum_diag * csr_mat

    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
        values = torch.from_numpy(cur_matrix.data)  #
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).to(self.device)  #

    def innerProduct(self, u_pos, i_pos, u_neg, j_neg):
        pred_i = torch.sum(torch.mul(u_pos, i_pos), dim=-1)
        pred_j = torch.sum(torch.mul(u_neg, j_neg), dim=-1)
        return pred_i, pred_j

    def sampleTrainBatch_dgl(self, batIds, pos_id=None, g=None, g_neg=None, sample_num=None, sample_num_neg=None):

        sub_g = dgl.sampling.sample_neighbors(g.cpu(), {'user': batIds}, sample_num, edge_dir='out', replace=True)
        row, col = sub_g.edges()
        row = row.reshape(len(batIds), sample_num)
        col = col.reshape(len(batIds), sample_num)

        if g_neg == None:
            return row, col
        else:
            sub_g_neg = dgl.sampling.sample_neighbors(g_neg, {'user': batIds}, sample_num_neg, edge_dir='out',
                                                      replace=True)
            row_neg, col_neg = sub_g_neg.edges()
            row_neg = row_neg.reshape(len(batIds), sample_num_neg)
            col_neg = col_neg.reshape(len(batIds), sample_num_neg)
            return row, col, col_neg

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    def gradient_penalty(self, D, xr, xf):

        LAMBDA = 0.3

        xf = xf.detach()
        xr = xr.detach()

        alpha = torch.rand(args.batch_size * 2, 1).to(self.device)
        alpha = alpha.expand_as(xr)

        interpolates = alpha * xr + ((1 - alpha) * xf)
        interpolates.requires_grad_()

        disc_interpolates = D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(disc_interpolates),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

        return gp

    def weighted_sum(self, anchor, nei, co):

        ac = torch.multiply(anchor, co).sum(-1).sum(-1)
        nc = torch.multiply(nei, co).sum(-1).sum(-1)

        an = (anchor.permute(1, 0, 2)[0])
        ne = (nei.permute(1, 0, 2)[0])

        an_w = an * (ac.unsqueeze(-1).repeat(1, args.embed_size))
        ne_w = ne * (nc.unsqueeze(-1).repeat(1, args.embed_size))

        res = (args.anchor_rate * an_w + (1 - args.anchor_rate) * ne_w).reshape(-1, args.sample_num_ii,
                                                                                args.embed_size).sum(1)

        return res

    def sample_topk(self, u_sim, users, emb_type=None):
        topk_p, topk_id = torch.topk(u_sim, args.ad_topk * 10, dim=-1)
        topk_data = topk_p.reshape(-1).cpu()
        topk_col = topk_id.reshape(-1).cpu().int()
        topk_row = torch.tensor(np.array(users)).unsqueeze(1).repeat(1, args.ad_topk * args.ad_topk_multi_num).reshape(
            -1).int()  #
        topk_csr = csr_matrix((topk_data.detach().numpy(), (topk_row.detach().numpy(), topk_col.detach().numpy())),
                              shape=(self.n_users, self.n_items))
        topk_g = dgl.heterograph({('user', 'ui', 'item'): topk_csr.nonzero()})
        _, topk_id = self.sampleTrainBatch_dgl(users, g=topk_g, sample_num=args.ad_topk, pos_id=None, g_neg=None,
                                               sample_num_neg=None)
        self.gene_fake[emb_type] = topk_id

        topk_id_u = torch.arange(len(users)).unsqueeze(1).repeat(1, args.ad_topk)
        topk_p = u_sim[topk_id_u, topk_id]
        return topk_p, topk_id

    def ssl_loss_calculation(self, ssl_image_logit, ssl_text_logit, ssl_common_logit):
        ssl_label_1_s2 = torch.ones(1, self.n_items).to(self.device)
        ssl_label_0_s2 = torch.zeros(1, self.n_items).to(self.device)
        ssl_label_s2 = torch.cat((ssl_label_1_s2, ssl_label_0_s2), 1)
        ssl_image_s2 = self.bce(ssl_image_logit, ssl_label_s2)
        ssl_text_s2 = self.bce(ssl_text_logit, ssl_label_s2)
        ssl_loss_s2 = ssl_image_s2 + ssl_text_s2

        ssl_label_1_c2 = torch.ones(1, self.n_items * 2).to(self.device)
        ssl_label_0_c2 = torch.zeros(1, self.n_items * 2).to(self.device)
        ssl_label_c2 = torch.cat((ssl_label_1_c2, ssl_label_0_c2), 1)
        ssl_result_c2 = self.bce(ssl_common_logit, ssl_label_c2)
        ssl_loss_c2 = ssl_result_c2

        ssl_loss2 = args.ssl_s_rate * ssl_loss_s2 + args.ssl_c_rate * ssl_loss_c2
        return ssl_loss2

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        # z1 = z1/((z1**2).sum(-1) + 1e-8)
        # z2 = z2/((z2**2).sum(-1) + 1e-8)
        return torch.mm(z1, z2.t())

    def batched_contrastive_loss(self, z1, z2, batch_size=1024):

        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / args.tau)  #

        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            tmp_i = indices[i * batch_size:(i + 1) * batch_size]

            tmp_refl_sim_list = []
            tmp_between_sim_list = []
            for j in range(num_batches):
                tmp_j = indices[j * batch_size:(j + 1) * batch_size]
                tmp_refl_sim = f(self.sim(z1[tmp_i], z1[tmp_j]))
                tmp_between_sim = f(self.sim(z1[tmp_i], z2[tmp_j]))

                tmp_refl_sim_list.append(tmp_refl_sim)
                tmp_between_sim_list.append(tmp_between_sim)

            refl_sim = torch.cat(tmp_refl_sim_list, dim=-1)
            between_sim = torch.cat(tmp_between_sim_list, dim=-1)

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag() / (
                    refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:,
                                                           i * batch_size:(i + 1) * batch_size].diag()) + 1e-8))

            del refl_sim, between_sim, tmp_refl_sim_list, tmp_between_sim_list

        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def feat_reg_loss_calculation(self, g_item_image, g_item_text, g_user_image, g_user_text):
        feat_reg = 1. / 2 * (g_item_image ** 2).sum() + 1. / 2 * (g_item_text ** 2).sum() \
                   + 1. / 2 * (g_user_image ** 2).sum() + 1. / 2 * (g_user_text ** 2).sum()
        feat_reg = feat_reg / self.n_items
        feat_emb_loss = args.feat_reg_decay * feat_reg
        return feat_emb_loss

    def fake_gene_loss_calculation(self, u_emb, i_emb, emb_type=None):
        if self.gene_u != None:
            gene_real_loss = (-F.logsigmoid((u_emb[self.gene_u] * i_emb[self.gene_real]).sum(-1) + 1e-8)).mean()
            gene_fake_loss = (1 - (
                -F.logsigmoid((u_emb[self.gene_u] * i_emb[self.gene_fake[emb_type]]).sum(-1) + 1e-8))).mean()

            gene_loss = gene_real_loss + gene_fake_loss
        else:
            gene_loss = 0

        return gene_loss

    def reward_loss_calculation(self, users, re_u, re_i, topk_id, topk_p):
        self.gene_u = torch.tensor(np.array(users)).unsqueeze(1).repeat(1, args.ad_topk)
        reward_u = re_u[self.gene_u]
        reward_i = re_i[topk_id]
        reward_value = (reward_u * reward_i).sum(-1)

        reward_loss = -(((topk_p * reward_value).sum(-1)).mean() + 1e-8).log()

        return reward_loss

    def u_sim_calculation(self, users, user_final, item_final):
        topk_u = user_final[users]
        u_ui = torch.tensor(self.ui_graph_raw[users].todense(), device=self.device)

        num_batches = (self.n_items - 1) // args.batch_size + 1
        indices = torch.arange(0, self.n_items).to(self.device)
        u_sim_list = []

        for i_b in range(num_batches):
            index = indices[i_b * args.batch_size:(i_b + 1) * args.batch_size]
            sim = torch.mm(topk_u, item_final[index].T)
            sim_gt = torch.multiply(sim, (1 - u_ui[:, index]))
            u_sim_list.append(sim_gt)

        u_sim = F.normalize(torch.cat(u_sim_list, dim=-1), p=2, dim=1)
        return u_sim

    def test(self, users_to_test, is_val):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, *rest = self.model(self.ui_graph, self.iu_graph, self.image_ui_graph,
                                                             self.image_iu_graph, self.text_ui_graph,
                                                             self.text_iu_graph)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
        return result

    def train(self):

        now_time = datetime.now()
        run_time = datetime.strftime(now_time, '%Y_%m_%d__%H_%M_%S')

        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        line_var_loss, line_g_loss, line_d_loss, line_cl_loss, line_var_recall, line_var_precision, line_var_ndcg = [], [], [], [], [], [], []
        stopping_step = 0
        should_stop = False
        cur_best_pre_0 = 0.
        # tb_writer = SummaryWriter(log_dir="/home/ww/Code/work5/MICRO2Ours/tensorboard/")
        # tensorboard_cnt = 0

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            contrastive_loss = 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            sample_time = 0.
            self.gene_u, self.gene_real, self.gene_fake = None, None, {}
            self.topk_p_dict, self.topk_id_dict = {}, {}

            for idx in tqdm(range(n_batch)):
                self.model.train()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1

                with torch.no_grad():
                    ua_embeddings, ia_embeddings, image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds \
                        , _, _, _, _, _, _ \
                        = self.model(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph,
                                     self.text_ui_graph, self.text_iu_graph)
                ui_u_sim_detach = self.u_sim_calculation(users, ua_embeddings, ia_embeddings).detach()
                image_u_sim_detach = self.u_sim_calculation(users, image_user_embeds, image_item_embeds).detach()
                text_u_sim_detach = self.u_sim_calculation(users, text_user_embeds, text_item_embeds).detach()
                inputf = torch.cat((image_u_sim_detach, text_u_sim_detach), dim=0)
                predf = (self.D(inputf))
                lossf = (predf.mean())
                u_ui = torch.tensor(self.ui_graph_raw[users].todense(), device=self.device)
                u_ui = F.softmax(u_ui - args.log_log_scale * torch.log(-torch.log(
                    torch.empty((u_ui.shape[0], u_ui.shape[1]), dtype=torch.float32).uniform_(0,
                                                                                              1).to(
                        self.device) + 1e-8) + 1e-8) / args.real_data_tau,
                                 dim=1)  # 0.002
                u_ui += ui_u_sim_detach * args.ui_pre_scale
                u_ui = F.normalize(u_ui, dim=1)
                inputr = torch.cat((u_ui, u_ui), dim=0)
                predr = (self.D(inputr))
                lossr = - (predr.mean())
                gp = self.gradient_penalty(self.D, inputr, inputf.detach())
                loss_D = lossr + lossf + args.gp_rate * gp
                self.optim_D.zero_grad()
                loss_D.backward()
                self.optim_D.step()
                line_d_loss.append(loss_D.detach().data)

                G_ua_embeddings, G_ia_embeddings, G_image_item_embeds, G_text_item_embeds, G_image_user_embeds, G_text_user_embeds \
                    , G_user_emb, _, G_image_user_id, G_text_user_id, _, _ \
                    = self.model(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph,
                                 self.text_ui_graph, self.text_iu_graph)

                G_u_g_embeddings = G_ua_embeddings[users]
                G_pos_i_g_embeddings = G_ia_embeddings[pos_items]
                G_neg_i_g_embeddings = G_ia_embeddings[neg_items]
                G_batch_mf_loss, G_batch_emb_loss, G_batch_reg_loss = self.bpr_loss(G_u_g_embeddings,
                                                                                    G_pos_i_g_embeddings,
                                                                                    G_neg_i_g_embeddings)
                G_image_u_sim = self.u_sim_calculation(users, G_image_user_embeds, G_image_item_embeds)
                G_text_u_sim = self.u_sim_calculation(users, G_text_user_embeds, G_text_item_embeds)
                G_image_u_sim_detach = G_image_u_sim.detach()
                G_text_u_sim_detach = G_text_u_sim.detach()

                if idx % args.T == 0 and idx != 0:
                    self.image_ui_graph_tmp = csr_matrix((torch.ones(len(self.image_ui_index['x'])),
                                                          (self.image_ui_index['x'], self.image_ui_index['y'])),
                                                         shape=(self.n_users, self.n_items))
                    self.text_ui_graph_tmp = csr_matrix(
                        (torch.ones(len(self.text_ui_index['x'])), (self.text_ui_index['x'], self.text_ui_index['y'])),
                        shape=(self.n_users, self.n_items))
                    self.image_iu_graph_tmp = self.image_ui_graph_tmp.T
                    self.text_iu_graph_tmp = self.text_ui_graph_tmp.T
                    self.image_ui_graph = self.sparse_mx_to_torch_sparse_tensor( \
                        self.csr_norm(self.image_ui_graph_tmp, mean_flag=True)
                    ).to(self.device)
                    self.text_ui_graph = self.sparse_mx_to_torch_sparse_tensor(
                        self.csr_norm(self.text_ui_graph_tmp, mean_flag=True)
                    ).to(self.device)
                    self.image_iu_graph = self.sparse_mx_to_torch_sparse_tensor(
                        self.csr_norm(self.image_iu_graph_tmp, mean_flag=True)
                    ).to(self.device)
                    self.text_iu_graph = self.sparse_mx_to_torch_sparse_tensor(
                        self.csr_norm(self.text_iu_graph_tmp, mean_flag=True)
                    ).to(self.device)

                    self.image_ui_index = {'x': [], 'y': []}
                    self.text_ui_index = {'x': [], 'y': []}

                else:
                    _, image_ui_id = torch.topk(G_image_u_sim_detach, int(self.n_items * args.m_topk_rate), dim=-1)
                    self.image_ui_index['x'] += np.array(
                        torch.tensor(users).repeat(1, int(self.n_items * args.m_topk_rate)).view(-1)).tolist()
                    self.image_ui_index['y'] += np.array(image_ui_id.cpu().view(-1)).tolist()
                    _, text_ui_id = torch.topk(G_text_u_sim_detach, int(self.n_items * args.m_topk_rate), dim=-1)
                    self.text_ui_index['x'] += np.array(
                        torch.tensor(users).repeat(1, int(self.n_items * args.m_topk_rate)).view(-1)).tolist()
                    self.text_ui_index['y'] += np.array(text_ui_id.cpu().view(-1)).tolist()

                feat_emb_loss = self.feat_reg_loss_calculation(G_image_item_embeds, G_text_item_embeds,
                                                               G_image_user_embeds, G_text_user_embeds)

                batch_contrastive_loss = 0
                batch_contrastive_loss1 = self.batched_contrastive_loss(G_image_user_id[users], G_user_emb[users])
                batch_contrastive_loss2 = self.batched_contrastive_loss(G_text_user_id[users], G_user_emb[users])

                batch_contrastive_loss = batch_contrastive_loss1 + batch_contrastive_loss2

                G_inputf = torch.cat((G_image_u_sim, G_text_u_sim), dim=0)
                G_predf = (self.D(G_inputf))

                G_lossf = -(G_predf.mean())
                batch_loss = G_batch_mf_loss + G_batch_emb_loss + G_batch_reg_loss + feat_emb_loss + args.cl_rate * batch_contrastive_loss + args.G_rate * G_lossf  # feat_emb_loss

                line_var_loss.append(batch_loss.detach().data)
                line_g_loss.append(G_lossf.detach().data)
                line_cl_loss.append(batch_contrastive_loss.detach().data)

                self.optimizer_D.zero_grad()
                batch_loss.backward(retain_graph=False)
                self.optimizer_D.step()

                loss += float(batch_loss)
                mf_loss += float(G_batch_mf_loss)
                emb_loss += float(G_batch_emb_loss)
                reg_loss += float(G_batch_reg_loss)

            del ua_embeddings, ia_embeddings, G_ua_embeddings, G_ia_embeddings, G_u_g_embeddings, G_neg_i_g_embeddings, G_pos_i_g_embeddings

            if math.isnan(loss) == True:
                self.logger.logging('ERROR: loss is nan.')
                sys.exit(1)

            if (epoch + 1) % args.verbose != 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f  + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss, contrastive_loss)
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_val, is_val=True)
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'].data)
            pre_loger.append(ret['precision'].data)
            ndcg_loger.append(ret['ndcg'].data)
            hit_loger.append(ret['hit_ratio'].data)

            line_var_recall.append(ret['recall'][1])
            line_var_precision.append(ret['precision'][1])
            line_var_ndcg.append(ret['ndcg'][1])

            tags = ["recall", "precision", "ndcg"]
            # tb_writer.add_scalar(tags[0], ret['recall'][1], epoch)
            # tb_writer.add_scalar(tags[1], ret['precision'][1], epoch)
            # tb_writer.add_scalar(tags[2], ret['ndcg'][1], epoch)

            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f, %.5f, %.5f], ' \
                           'precision=[%.5f, %.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0],
                            ret['recall'][1], ret['recall'][2],
                            ret['recall'][-1],
                            ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][-1],
                            ret['hit_ratio'][0], ret['hit_ratio'][1], ret['hit_ratio'][2], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][-1])
                self.logger.logging(perf_str)

            if ret['recall'][1] > best_recall:
                best_recall = ret['recall'][1]
                test_ret = self.test(users_to_test, is_val=False)
                self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (
                    eval(args.Ks)[1], test_ret['recall'][1], test_ret['precision'][1], test_ret['ndcg'][1]))
                stopping_step = 0
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                self.logger.logging('#####Early stopping steps: %d #####' % stopping_step)
            else:
                self.logger.logging('#####Early stop! #####')
                break
        self.logger.logging(str(test_ret))

        return best_recall, run_time

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':

    if args.missing_features == 'True':
        args.missing_features = True
    else:
        args.missing_features = False

    if args.dataset == 'baby':
        args.D_lr = 3e-4
        args.cl_rate = 0.03
        args.drop_rate = 0.2
        args.model_cat_rate = 0.55
        args.head_num = 4
        args.G_rate = 0.0001
        args.G_drop1 = 0.31
        args.G_drop2 = 0.5

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    trainer = Trainer(data_config=config)
    trainer.train()
