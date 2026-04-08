import numpy as np
import scipy.sparse as sp


class Graph(object):
    def __init__(self):
        pass

    @staticmethod
    def normalize_graph_mat(adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    @staticmethod
    def r_adj_normalize_graph_mat(adj_mat, r):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv_left = np.power(rowsum, -r).flatten()
            d_inv_left[np.isinf(d_inv_left)] = 0.
            d_mat_inv_left = sp.diags(d_inv_left)

            norm_adj_tmp = d_mat_inv_left.dot(adj_mat)

            d_inv_right = np.power(rowsum, -(1-r)).flatten()
            d_inv_right[np.isinf(d_inv_right)] = 0.
            d_mat_inv_right = sp.diags(d_inv_right)

            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv_right)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        pass
