from ast import literal_eval as make_tuple

from tqdm import tqdm
import torch
import os
import numpy as np

from elliot.utils.write import store_recommendation
from .custom_sampler import Sampler
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .LGMRecModel import LGMRecModel
import math
import random

from torch_sparse import SparseTensor

from torch_sparse import mul, fill_diag, sum


def apply_norm(edge_index, add_self_loops=True):
    adj_t = edge_index
    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.)
    deg = sum(adj_t, dim=1) + 1e-7
    deg_inv = deg.pow_(-0.5)
    norm_adj_t = mul(adj_t, deg_inv.view(-1, 1))
    norm_adj_t = mul(norm_adj_t, deg_inv.view(1, -1))
    return norm_adj_t


class LGMRec(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_c_l", "c_l", "c_l", 0.01, float, None),
            ("_a", "a", "a", 0.01, float, None),
            ("_cf", "cf", "cf", 'mf', str, None),
            ("_n_h_l", "n_h_l", "n_h_l", 1, int, None),
            ("_h_n", "h_n", "h_n", 1, int, None),
            ("_n_mm_l", "n_mm_l", "n_mm_l", 1, int, None),
            ("_n_ui_l", "n_ui_l", "n_ui_l", 3, int, None),
            ("_f_m", "f_m", "f_m", 64, int, None),
            ("_modalities", "modalities", "modalites", "('visual','textual')", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_keep_rate", "keep_rate", "keep_rate", 0.9, float, None),
            ("_lr_sched", "lr_sched", "lr_sched", "(0.96,50)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_loaders", "loaders", "loads", "('VisualAttribute','TextualAttribute')", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-"))
        ]
        self.autoset_params()

        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

        self._sampler = Sampler(self._data.i_train_dict,
                                self._data.transactions,
                                self._batch_size,
                                self._data.edge_index['itemId'].unique().tolist(),
                                self._seed)

        if self._batch_size < 1:
            self._batch_size = self._num_users

        row, col = data.sp_i_train.nonzero()
        col = [c + self._num_users for c in col]
        edge_index = np.array([row, col])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        self.edge_index = torch.sparse_coo_tensor(
            torch.tensor(list(zip(data.sp_i_train.nonzero()[0], data.sp_i_train.nonzero()[1])), dtype=torch.int64).t(),
            torch.FloatTensor([1.0] * edge_index.shape[1]),
            [self._num_users, self._num_items])
        self.adj = torch.sparse_coo_tensor(torch.vstack([torch.cat([edge_index[0], edge_index[1]]), torch.cat([edge_index[1], edge_index[0]])]),
                                           torch.FloatTensor([1.0] * edge_index.shape[1] * 2),
                                           [self._num_users + self._num_items, self._num_users + self._num_items])
        num_inters = self.edge_index.sum(dim=1).values()

        norm_adj = SparseTensor(row=torch.cat([edge_index[0], edge_index[1]], dim=0),
                                col=torch.cat([edge_index[1], edge_index[0]], dim=0),
                                sparse_sizes=(self._num_users + self._num_items,
                                              self._num_users + self._num_items))

        self.norm_adj = apply_norm(norm_adj, add_self_loops=False)

        for m_id, m in enumerate(self._modalities):
            self.__setattr__(f'''_side_{m}''',
                             self._data.side_information.__getattribute__(f'''{self._loaders[m_id]}'''))

        all_multimodal_features = []
        for m_id, m in enumerate(self._modalities):
            all_multimodal_features.append(self.__getattribute__(
                f'''_side_{self._modalities[m_id]}''').object.get_all_features())

        self._model = LGMRecModel(
            num_users=self._num_users,
            num_items=self._num_items,
            num_hyper_layers=self._n_h_l,
            num_ui_layers=self._n_ui_l,
            num_mm_layers=self._n_mm_l,
            hyper_num=self._h_n,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            embed_k_multimod=self._f_m,
            cf_model=self._cf,
            l_w=self._l_w,
            c_l=self._c_l,
            modalities=self._modalities,
            alpha=self._a,
            edge_index=self.edge_index,
            norm_adj=self.norm_adj,
            num_inters=num_inters,
            multimodal_features=all_multimodal_features,
            keep_rate=self._keep_rate,
            lr_sched=self._lr_sched,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "LGMRec" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            self._model.train()
            n_batch = int(
                self._data.transactions / self._batch_size) if self._data.transactions % self._batch_size == 0 else int(
                self._data.transactions / self._batch_size) + 1
            self._data.edge_index = self._data.edge_index.sample(frac=1, replace=False, random_state=self._seed).reset_index(drop=True)
            edge_index = np.array([self._data.edge_index['userId'].tolist(), self._data.edge_index['itemId'].tolist()])
            with tqdm(total=n_batch, disable=not self._verbose) as t:
                for batch in self._sampler.step(edge_index):
                    user, pos, neg = batch
                    steps += 1
                    loss += self._model.train_step((user, pos, neg))

                    if math.isnan(loss) or math.isinf(loss) or (not loss):
                        break

                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()
                self._model.lr_scheduler.step()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        self._model.eval()
        with torch.no_grad():
            gum, gim, _ = self._model.propagate_embeddings()
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(gum[offset: offset_stop], gim)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss/(it + 1):.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._saving_filepath)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False
