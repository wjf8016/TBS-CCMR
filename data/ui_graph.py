import numpy as np
from collections import defaultdict
from data.data import MBData
from data.graph import Graph
import scipy.sparse as sp


class InteractionPlus(MBData,Graph):
    def __init__(self, conf, training_p, test, training_c, training_v, type_num):
        Graph.__init__(self)
        MBData.__init__(self,conf,training_p, test, training_c, training_v, type_num)

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()
        self.rAdjNorm = float(conf['r-adjNorm'])
        self.__generate_set()
        self.user_num = len(self.user)
        self.item_num = len(self.item)
        self.type_num = self.type_num
        self.ui_adj_p = self.__create_sparse_adjacency(self.training_p)
        self.ui_adj_c = self.__create_sparse_adjacency(self.training_c)
        self.ui_adj_v = self.__create_sparse_adjacency(self.training_v)

        self.norm_adj_p = self.r_adj_normalize_graph_mat(self.ui_adj_p, self.rAdjNorm)
        self.norm_adj_c = self.r_adj_normalize_graph_mat(self.ui_adj_c, self.rAdjNorm)
        self.norm_adj_v = self.r_adj_normalize_graph_mat(self.ui_adj_v, self.rAdjNorm)
        self.all_adj_norm = self.ui_adj_p + self.ui_adj_c + self.ui_adj_v

        self.degree = {}
        self.cal_degree()

    def __generate_set(self):
        rating = 1
        for entry in self.training_p:
            user, item = entry
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                # userList.append
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating

        for entry in self.training_c:
            user, item = entry
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item

        for entry in self.training_v:
            user, item = entry
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item

        for entry in self.test_data:
            user, item = entry
            if user not in self.user:
                continue
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item

            self.test_set[user][item] = rating
            self.test_set_item.add(item)


    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_p]
        col_idx = [self.item[pair[1]] for pair in self.training_p]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def __create_sparse_adjacency(self, data, self_connection=False):
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in data]
        col_idx = [self.item[pair[1]] for pair in data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0]+adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        row, col, entries = [], [], []
        for pair in self.training_p:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num,self.item_num),dtype=np.float32)
        return interaction_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_p)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        'whether user u rated item i'
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.user_rated(u)
            vec = np.zeros(len(self.item))
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m

    def cal_degree(self):
        for key, value in self.item.items():
            self.degree[key] = self.all_adj_norm[self.user_num + value, :].sum()