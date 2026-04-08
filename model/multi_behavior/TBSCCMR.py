import torch
import torch.nn as nn
import torch.nn.functional as F
from base.mbgraph_recommender import MBGraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.ui_graph import InteractionPlus
from torch.nn import init
from time import time


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim, eps=1e-8)

    def forward(self, x):
        return self.fn(self.norm(x))


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


class TBSCCMR(MBGraphRecommender):
    def __init__(self, conf, training_p, test_set, training_c, training_v, type_num):
        super(TBSCCMR, self).__init__(conf, training_p, test_set, training_c, training_v, type_num)
        args = OptionConf(self.config['TBSCCMR'])
        self.data = InteractionPlus(conf, training_p, test_set, training_c, training_v, type_num)
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.temperature = float(args['-t'])
        self.layer_cl = int(args['-l*'])
        self.model = TBSCCMR_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.temperature, self.layer_cl)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            st = time()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                model.train()
                rec_user_emb_list, rec_item_emb_list, cl_user_emb_list, cl_item_emb_list, u_cls, i_cls, target_emb_, aux1_emb_ = model(
                    True)
                aaaa, bbbb = torch.split(target_emb_, [self.data.user_num, self.data.item_num])
                cl_loss2 = self.cl_rate * model.cal_cl_loss([user_idx, pos_idx], aaaa,
                                                            rec_user_emb_list[-1], bbbb,
                                                            rec_item_emb_list[-1])

                aaaaa, bbbbb = torch.split(aux1_emb_, [self.data.user_num, self.data.item_num])
                cl_loss3 = self.cl_rate * model.cal_cl_loss([user_idx, pos_idx], aaaaa,
                                                            rec_user_emb_list[-1], bbbbb,
                                                            rec_item_emb_list[-1])

                total_rec_loss = 0
                total_cl_loss = 0
                total_l2_reg_loss = 0
                total_cl1_loss = 0
                for i in range(len(rec_user_emb_list)):
                    cl_loss1 = self.cl_rate * model.cal_cl_loss([user_idx, pos_idx], u_cls[-1],
                                                                u_cls[i], i_cls[-1],
                                                                i_cls[i])
                    total_cl1_loss = total_cl1_loss + cl_loss1
                for i in range(len(rec_user_emb_list)):
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb_list[i][user_idx], rec_item_emb_list[i][
                        pos_idx], rec_item_emb_list[i][neg_idx]
                    cl_loss = self.cl_rate * model.cal_cl_loss([user_idx, pos_idx], rec_user_emb_list[i],
                                                               cl_user_emb_list[i], rec_item_emb_list[i],
                                                               cl_item_emb_list[i])
                    total_cl_loss = total_cl_loss + cl_loss

                    rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                    total_rec_loss = total_rec_loss + rec_loss
                    total_l2_reg_loss = total_l2_reg_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb)

                batch_loss = total_rec_loss + total_l2_reg_loss + total_cl1_loss + cl_loss2 + total_cl_loss + cl_loss3
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', total_rec_loss.item(), 'cl_loss',
                          total_cl_loss.item())
            print(time() - st)
            model.eval()

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation1(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class TBSCCMR_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, temperature, layer_cl):
        super(TBSCCMR_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temperature = temperature
        self.embedding_dict = self._init_model()
        self.layer_cl = layer_cl
        self.norm_adj_p = data.norm_adj_p
        self.sparse_norm_adj_p = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj_p).cuda()
        self.norm_adj_c = data.norm_adj_c
        self.sparse_norm_adj_c = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj_c).cuda()
        self.norm_adj_v = data.norm_adj_v
        self.sparse_norm_adj_v = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj_v).cuda()

        self.view_mlp = PreNormResidual(self.emb_size, FeedForward(self.emb_size, 5, dropout=0.2))
        self.cart_mlp = PreNormResidual(self.emb_size, FeedForward(self.emb_size, 5, dropout=0.1))
        self.purchase_mlp = PreNormResidual(self.emb_size, FeedForward(self.emb_size, 5, dropout=0))
        self.act1 = torch.nn.PReLU()
        self.act = torch.nn.Sigmoid()
        self.gating_weightu1 = nn.Parameter(
            torch.FloatTensor(self.emb_size, self.emb_size))
        nn.init.xavier_normal_(self.gating_weightu1.data)

        self.gating_weightib1 = nn.Parameter(
            torch.FloatTensor(1, self.emb_size))
        nn.init.xavier_normal_(self.gating_weightib1.data)

        self.gating_weightu2 = nn.Parameter(
            torch.FloatTensor(self.emb_size, self.emb_size))
        nn.init.xavier_normal_(self.gating_weightu2.data)

        self.gating_weightib2 = nn.Parameter(
            torch.FloatTensor(1, self.emb_size))
        nn.init.xavier_normal_(self.gating_weightib2.data)

        self.gating_weighti1 = nn.Parameter(
            torch.FloatTensor(self.emb_size, self.emb_size))
        nn.init.xavier_normal_(self.gating_weighti1.data)

        self.gating_weightbi1 = nn.Parameter(
            torch.FloatTensor(1, self.emb_size))
        nn.init.xavier_normal_(self.gating_weightbi1.data)

        self.gating_weighti2 = nn.Parameter(
            torch.FloatTensor(self.emb_size, self.emb_size))
        nn.init.xavier_normal_(self.gating_weighti2.data)

        self.gating_weightbi2 = nn.Parameter(
            torch.FloatTensor(1, self.emb_size))
        nn.init.xavier_normal_(self.gating_weightbi2.data)

        self.i_w = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))
        self.u_w = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))
        self.ii_w = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))
        self.uu_w = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))
        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)
        init.xavier_uniform_(self.ii_w)
        init.xavier_uniform_(self.uu_w)
        self.sigmoid = torch.nn.Sigmoid()

        self.dropout = nn.Dropout(0.2, inplace=True)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        print("data describe", self.data.user_num, self.data.item_num)
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        user_embedding = self.embedding_dict['user_emb']
        item_embedding = self.embedding_dict['item_emb']

        uu_embed1 = torch.multiply(user_embedding, self.sigmoid(
            torch.matmul(user_embedding, self.gating_weightu1) + self.gating_weightib1))
        ii_embed1 = torch.multiply(item_embedding, self.sigmoid(
            torch.matmul(item_embedding, self.gating_weighti1) + self.gating_weightbi1))
        uu_embed2 = torch.multiply(user_embedding, self.sigmoid(
            torch.matmul(user_embedding, self.gating_weightu2) + self.gating_weightib2))
        ii_embed2 = torch.multiply(item_embedding, self.sigmoid(
            torch.matmul(item_embedding, self.gating_weighti2) + self.gating_weightbi2))
        behavior_matrix_dict = {'view': self.sparse_norm_adj_v, 'cart': self.sparse_norm_adj_c,
                                'purchase': self.sparse_norm_adj_p}
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        ego_embeddings_ori = ego_embeddings
        ego_embeddings1 = torch.cat((uu_embed1, ii_embed1),
                                    dim=0)

        ego_embeddings2 = torch.cat((uu_embed2, ii_embed2),
                                    dim=0)
        behavior_mlp_dict = {'view': self.view_mlp, 'cart': self.cart_mlp, 'purchase': self.purchase_mlp}

        if perturbed:

            target_emb_ori = []
            target_emb = []
            aux0_emb = []
            aux1_emb = []
            u_embs = []
            i_embs = []
            target_emb_ori.append(ego_embeddings)
            target_emb.append(ego_embeddings)
            aux0_emb.append(ego_embeddings1)
            aux1_emb.append(ego_embeddings2)

            for k in range(self.n_layers):
                embeddings_list = []
                user_embedding_list = []
                item_embedding_list = []
                for behavior, matrix in behavior_matrix_dict.items():
                    if behavior == "view":
                        ego_embeddings = torch.sparse.mm(matrix, ego_embeddings1)
                    if behavior == "cart":
                        ego_embeddings = torch.sparse.mm(matrix, ego_embeddings2)
                    if behavior == "buy":
                        ego_embeddings = torch.sparse.mm(matrix, ego_embeddings)
                    embeddings_list.append(ego_embeddings)
                ego_embeddings1 = embeddings_list[0]
                ego_embeddings2 = embeddings_list[1]
                ego_embeddings = embeddings_list[-1]
                target_emb_ori.append(ego_embeddings)
                uu0, ii0 = torch.split(ego_embeddings1, [self.data.user_num, self.data.item_num])
                uu1, ii1 = torch.split(ego_embeddings2, [self.data.user_num, self.data.item_num])
                uu, ii = torch.split(ego_embeddings, [self.data.user_num, self.data.item_num])
                user_embedding_list.append(uu0)
                user_embedding_list.append(uu1)
                user_embedding_list.append(uu)
                item_embedding_list.append(ii0)
                item_embedding_list.append(ii1)
                item_embedding_list.append(ii)
                user_embeddingssss = torch.stack(user_embedding_list, dim=0)
                item_embeddingssss = torch.stack(item_embedding_list, dim=0)
                if k == 0:
                    user_embedding = self.act1(torch.matmul(torch.mean(user_embeddingssss, dim=0), self.u_w))
                    item_embedding = self.act1(torch.matmul(torch.mean(item_embeddingssss, dim=0), self.i_w))
                if k == 1:
                    user_embedding = self.act1(torch.matmul(torch.mean(user_embeddingssss, dim=0), self.uu_w))
                    item_embedding = self.act1(torch.matmul(torch.mean(item_embeddingssss, dim=0), self.ii_w))
                ego_embeddings = torch.cat([user_embedding, item_embedding], 0)
                target_emb.append(ego_embeddings)
                aux0_emb.append(ego_embeddings1)
                aux1_emb.append(ego_embeddings2)
                # F/no
                u_embs.append(user_embeddingssss)
                i_embs.append(item_embeddingssss)

            target_emb_ = torch.stack(target_emb, dim=1)
            target_emb_ = torch.mean(target_emb_, dim=1)

            target_emb_ori = torch.stack(target_emb_ori, dim=1)
            target_emb_ori = torch.mean(target_emb_ori, dim=1)
            target_emb_ori_onetop = target_emb_ori[1]
            aux0_emb_ = torch.stack(aux0_emb, dim=1)
            aux0_emb_ = torch.mean(aux0_emb_, dim=1)
            aux0_emb_ = F.normalize(aux0_emb_)
            aux1_emb_ = torch.stack(aux1_emb, dim=1)
            aux1_emb_ = torch.mean(aux1_emb_, dim=1)
            u_cls = torch.stack(u_embs, dim=0).mean(dim=0)
            i_cls = torch.stack(i_embs, dim=0).mean(dim=0)

            target_emb_ = F.normalize(target_emb_)
            all_gcn_embeddings = {}
            all_cl_embeddings = {}

            total_embeddings = target_emb_
            for behavior, matrix in behavior_matrix_dict.items():
                layer_embeddings = total_embeddings

                layer_embeddings, cl_embeddings, one_top = self.gcn_propagate(layer_embeddings, matrix, perturbed)
                if behavior == "view":
                    input = layer_embeddings
                if behavior == "cart":
                    input = layer_embeddings + aux1_emb_
                if behavior == "buy":
                    input = layer_embeddings

                temp_mlp = behavior_mlp_dict[behavior]
                total_embeddings = 0.6 * temp_mlp(input) + total_embeddings + 0.4 * input
                all_gcn_embeddings[behavior] = total_embeddings
                all_cl_embeddings[behavior] = cl_embeddings

            final_user_embeddings = []
            final_item_embeddings = []
            final_user_embeddings_cl = []
            final_item_embeddings_cl = []
            for behavior in all_gcn_embeddings:
                user_embeddings, item_embeddings = torch.split(all_gcn_embeddings[behavior],
                                                               [self.data.user_num, self.data.item_num])
                final_user_embeddings.append(user_embeddings)
                final_item_embeddings.append(item_embeddings)
                user_embeddings_cl, item_embeddings_cl = torch.split(all_cl_embeddings[behavior],
                                                                     [self.data.user_num, self.data.item_num])
                final_user_embeddings_cl.append(user_embeddings_cl)
                final_item_embeddings_cl.append(item_embeddings_cl)
            return final_user_embeddings, final_item_embeddings, final_user_embeddings_cl, final_item_embeddings_cl, u_cls, i_cls, target_emb_, ego_embeddings_ori
        else:
            target_emb_ori = []
            target_emb = []
            aux0_emb = []
            aux1_emb = []
            u_embs = []
            i_embs = []
            target_emb_ori.append(ego_embeddings)
            target_emb.append(ego_embeddings)
            aux0_emb.append(ego_embeddings1)
            aux1_emb.append(ego_embeddings2)

            for k in range(self.n_layers):
                embeddings_list = []
                user_embedding_list = []
                item_embedding_list = []
                for behavior, matrix in behavior_matrix_dict.items():
                    if behavior == "view":
                        ego_embeddings = torch.sparse.mm(matrix, ego_embeddings1)
                    if behavior == "cart":
                        ego_embeddings = torch.sparse.mm(matrix, ego_embeddings2)
                    if behavior == "buy":
                        ego_embeddings = torch.sparse.mm(matrix, ego_embeddings)
                    embeddings_list.append(ego_embeddings)
                ego_embeddings1 = embeddings_list[0]
                ego_embeddings2 = embeddings_list[1]
                ego_embeddings = embeddings_list[-1]
                target_emb_ori.append(ego_embeddings)
                uu0, ii0 = torch.split(ego_embeddings1, [self.data.user_num, self.data.item_num])
                uu1, ii1 = torch.split(ego_embeddings2, [self.data.user_num, self.data.item_num])
                uu, ii = torch.split(ego_embeddings, [self.data.user_num, self.data.item_num])
                user_embedding_list.append(uu0)
                user_embedding_list.append(uu1)
                user_embedding_list.append(uu)
                item_embedding_list.append(ii0)
                item_embedding_list.append(ii1)
                item_embedding_list.append(ii)
                user_embeddingssss = torch.stack(user_embedding_list, dim=0)
                item_embeddingssss = torch.stack(item_embedding_list, dim=0)
                if k == 0:
                    user_embedding = self.act1(torch.matmul(torch.mean(user_embeddingssss, dim=0), self.u_w))
                    item_embedding = self.act1(torch.matmul(torch.mean(item_embeddingssss, dim=0), self.i_w))
                if k == 1:
                    user_embedding = self.act1(torch.matmul(torch.mean(user_embeddingssss, dim=0), self.uu_w))
                    item_embedding = self.act1(torch.matmul(torch.mean(item_embeddingssss, dim=0), self.ii_w))
                ego_embeddings = torch.cat([user_embedding, item_embedding], 0)
                target_emb.append(ego_embeddings)
                aux0_emb.append(ego_embeddings1)
                aux1_emb.append(ego_embeddings2)

                # F/no
                u_embs.append(user_embeddingssss)
                i_embs.append(item_embeddingssss)
            target_emb_ = torch.stack(target_emb, dim=1)
            target_emb_ = torch.mean(target_emb_, dim=1)

            target_emb_ori = torch.stack(target_emb_ori, dim=1)
            target_emb_ori = torch.mean(target_emb_ori, dim=1)
            target_emb_ori_onetop = target_emb_ori[1]
            aux0_emb_ = torch.stack(aux0_emb, dim=1)
            aux0_emb_ = torch.mean(aux0_emb_, dim=1)
            aux1_emb_ = torch.stack(aux1_emb, dim=1)
            aux1_emb_ = torch.mean(aux1_emb_, dim=1)
            aux0_emb_ = F.normalize(aux0_emb_)
            target_emb_ = F.normalize(target_emb_)

            all_gcn_embeddings = {}
            total_embeddings = target_emb_
            for behavior, matrix in behavior_matrix_dict.items():
                layer_embeddings = total_embeddings
                layer_embeddings, one_top = self.gcn_propagate(layer_embeddings, matrix, perturbed)
                if behavior == "view":
                    input = layer_embeddings
                if behavior == "cart":
                    input = layer_embeddings + aux1_emb_
                if behavior == "buy":
                    input = layer_embeddings
                temp_mlp = behavior_mlp_dict[behavior]
                total_embeddings = 0.4 * temp_mlp(input) + total_embeddings + 0.6 * input
                all_gcn_embeddings[behavior] = total_embeddings
            final_user_embeddings = []
            final_item_embeddings = []

            for behavior in all_gcn_embeddings:
                user_embeddings, item_embeddings = torch.split(all_gcn_embeddings[behavior],
                                                               [self.data.user_num, self.data.item_num])
                final_user_embeddings.append(user_embeddings)
                final_item_embeddings.append(item_embeddings)
            return final_user_embeddings[-1], final_item_embeddings[-1]

    def gcn_propagate(self, embeddings, norm_adj, perturbed=False):
        ego_embeddings = embeddings
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            if k == self.layer_cl - 1:
                all_embeddings_cl = ego_embeddings
        one_top = all_embeddings[0]
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        if perturbed:
            return final_embeddings, all_embeddings_cl, one_top
        else:
            return final_embeddings, one_top

    def gcn(self, embeddings, norm_adj, perturbed=False):
        all_embeddings = []
        all_embeddings.append(embeddings)
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(norm_adj, embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temperature)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temperature)
        return user_cl_loss + item_cl_loss

    def cal_cl_loss1(self, idx, user_view1, user_view2, item_view1, item_view2, temperature):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], temperature)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], temperature)
        return user_cl_loss + item_cl_loss