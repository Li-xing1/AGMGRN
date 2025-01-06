"""Adaptive Gated Meta Graph Retention Network(AGMGRN)"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit


def nozero_softmax(data, dim):
    mask = data.ne(0)
    max = torch.max(data, dim=-1, keepdim=True)[0]
    data = data - max
    exps = torch.exp(data)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + 0.00001
    alpha = masked_exps / masked_sums
    return alpha


def nonan_softmax(data, dim):
    max, _ = torch.max(data, dim=dim, keepdim=True)
    data = data - max
    out = torch.softmax(data, dim=dim)
    return out


def multihead_linear_transform(W, inputs):
    B, P, N, H, d_k, d_model = W.shape
    inputs = inputs.reshape((B, P, N, 1, d_model, 1))
    out = torch.matmul(W, inputs).squeeze(-1)  # (B, P, N, H, d_k)
    return out


def multihead_temporal_attention(Q, K, V, causal=False):
    B, P1, N, H, d_k = Q.shape
    P = K.shape[1]
    Q = Q.permute((0, 2, 3, 1, 4))  # (B, N, H, P1, d_k)
    K = K.permute((0, 2, 3, 4, 1))  # (B, N, H, d_k, P)

    scaled_dot_product = torch.matmul(Q, K) / math.sqrt(d_k)  # (B, N, H, P1, P)
    if causal is True:
        assert P1 == P
        mask = scaled_dot_product.new_full((P, P), -np.inf).triu(
            diagonal=1)  # 掩码矩阵为一个上三角为—inf的矩阵，其余位置为0，目的是使算softmax时上三角位置为0，就不会考虑后边时刻信息。
        scaled_dot_product += mask
    alpha = nonan_softmax(scaled_dot_product, dim=-1)
    V = V.permute((0, 2, 3, 1, 4))  # (B, N, H, P, d_k)
    out = torch.matmul(alpha, V)  # (B, N, H, P1, d_k)
    out = out.permute((2, 0, 3, 1, 4))  # (H,B, P1, N,  d_k)
    return out


def multihead_temporal_regnet(Q, K, V, D):
    _, B, P1, N, H, d_k = Q.shape

    Q = Q.permute((0, 1, 3, 4, 2, 5))  # (B, N, H, P,d_k)
    K = K.permute((0, 1, 3, 4, 5, 2))  # (B, N, H, d_k, P)
    V = V.permute((0, 2, 3, 1, 4))  # (B, N, H, P, d_k)

    retention = torch.sum(torch.matmul(Q, K) / math.sqrt(d_k), dim=0)  # (B, N, H, P, P)
    retention = retention * D  # (B, N, H, P, P)
    retention_sum = torch.abs(torch.sum(retention, dim=-1, keepdim=True))
    retention_sum[retention_sum < 1.] = 1.
    retention = retention / retention_sum
    output = torch.matmul(retention, V)  # (B, N, H, P, d_k)
    output = output.permute((0, 2, 3, 1, 4))
    return output


def multihead_spatial_attention_predefine_graph(Q, K, V, transition_matrix):
    '''
    math: [softmax(QK/SQRT(d_k))*transition_matrix]V
    :param Q:shape：[B,P,N,H,d_K]
    :param K:shape：[B,P,N,H,d_K]
    :param V:shape：[B,P,N,H,d_K]
    :param transition_matrix: 类似与邻接矩阵 shape:[N,N]
    :return: shapr[B, P, N, H * d_k]
    '''
    B, P, N, H, d_k = Q.shape

    Q = Q.permute((0, 1, 3, 2, 4))  # (B, P, H, N, d_k)
    K = K.permute((0, 1, 3, 2, 4))  # (B, P, H, N, d_k)
    V = V.permute((0, 1, 3, 2, 4))  # (B, P, H, N, d_k)

    index = transition_matrix.nonzero()
    Q = Q[:, :, :, index[:, 0], :].unsqueeze(dim=-2)
    K = K[:, :, :, index[:, 1], :].unsqueeze(dim=-1)
    scaled = torch.matmul(Q, K).squeeze(dim=-1).squeeze(dim=-1) / math.sqrt(d_k)
    scaled_dot_product = torch.full(size=[B, P, H, N, N], fill_value=0., dtype=torch.float32, device=Q.device)
    scaled_dot_product[:, :, :, index[:, 0], index[:, 1]] = scaled
    alpha = nozero_softmax(scaled_dot_product, dim=-1)  # (B, P, H, N, N)
    out = torch.matmul(alpha * transition_matrix, V)  # (B, P, H, N, d_k)

    out = out.permute((2, 0, 1, 3, 4))
    return out


def multihead_spatial_attention_adaptive_graph(Q, K, V, transition_matrix):
    '''
    math: [softmax(QK/SQRT(d_k))*transition_matrix]V
    :param Q:shape：[B,P,N,H,d_K]
    :param K:shape：[B,P,N,H,d_K]
    :param V:shape：[B,P,N,H,d_K]
    :param transition_matrix: 类似与邻接矩阵 shape:[N,N]
    :return: shapr[B, P, N, H * d_k]
    '''
    B, P, N, H, d_k = Q.shape
    Q = Q.permute((0, 1, 3, 2, 4))  # (B, P, H, N, d_k)
    K = K.permute((0, 1, 3, 2, 4))  # (B, P, H, N, d_k)
    V = V.permute((0, 1, 3, 2, 4))  # (B, P, H, N, d_k)
    index = transition_matrix.nonzero()
    Q = Q[index[:, 0], index[:, 1], :, index[:, 2], :].unsqueeze(dim=-2)
    K = K[index[:, 0], index[:, 1], :, index[:, 3], :].unsqueeze(dim=-1)
    scaled = torch.matmul(Q, K).squeeze(dim=-1).squeeze(dim=-1) / math.sqrt(d_k)
    scaled_dot_product = torch.full(size=[B, P, H, N, N], fill_value=0., dtype=torch.float32, device=Q.device)
    scaled_dot_product[index[:, 0], index[:, 1], :, index[:, 2], index[:, 3]] = scaled
    alpha = nozero_softmax(scaled_dot_product, dim=-1)  # (B, P, H, N, N)
    out = torch.matmul(alpha * torch.stack((transition_matrix,) * H, dim=2), V)  # (B, P, H, N, d_k)
    out = out.permute((2, 0, 1, 3, 4))
    return out


class Swish(nn.Module):
    def __init__(self, d_model):
        super(Swish, self).__init__()
        self.linear_wg = nn.Linear(d_model, d_model)
        self.linear_wo = nn.Linear(d_model, d_model)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, inputs, out):
        out = self.linear_wo(self.swish(self.linear_wg(inputs) * out))

        return out


class TemporalEmbedding(nn.Module):
    def __init__(self, temporal_embedding_list, d_model, noGC=False, max_len=12):
        '''
        时间编码模块
        :param num_embeddings: list,每个编码最大编号，one-hot
        :param d_model:
        :param max_len: max(P,Q,in_len,out_len)
        '''
        super(TemporalEmbedding, self).__init__()
        self.num_embeddings, self.index_embeddings, self.index_feature = self.embeddings(temporal_embedding_list)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        # pos编码
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
        self.register_buffer('pe', pe)  # 不可学参数

        self.embedding_modules = nn.ModuleList([nn.Embedding(item, d_model) for item in self.num_embeddings])
        if len(self.index_feature) > 0:
            self.feature_tran = nn.Linear(len(self.index_feature), d_model)
            self.gat = Gated_Dynamic_Connection(len(self.index_embeddings) + 2, d_model, d_model, noGC)
        else:
            self.gat = Gated_Dynamic_Connection(len(self.index_embeddings) + 1, d_model, d_model, noGC)

    def embeddings(self, embedding_list):
        num_embeddings = []
        index_embeddings = []
        index_feature = []
        for i in range(len(embedding_list)):
            if embedding_list[i] == 1:
                index_feature.append(i)
            if embedding_list[i] > 1:
                num_embeddings.append(embedding_list[i])
                index_embeddings.append(i)
        return num_embeddings, index_embeddings, index_feature

    def forward(self, temporal_characterisrics):
        '''

        :param extras: 时间编码列表[in_time0,tar_time0,in_time1,tar_time1,......,in_time'len(num_embeddings)',tar_time'len(num_embeddings)']
                                in_time0:shape:[B,P],tar_time0:shape:[B,Q],extra:[len(num_embeddings)*([B,P],[B,Q])]
        :return:
        inputs_extras_embedding:shape:[B,P,d_model]
        targets_extras_embedding:shape:[B,Q,d_model]
        '''
        inputs_extras_onehot = temporal_characterisrics[0][self.index_embeddings].to(torch.int32)
        targets_extras_onehot = temporal_characterisrics[1][self.index_embeddings].to(torch.int32)
        B, P = inputs_extras_onehot[0].shape
        _, Q = targets_extras_onehot[0].shape

        inputs_pe = self.pe[:P, :].expand(B, P, self.d_model)
        targets_pe = self.pe[:Q, :].expand(B, Q, self.d_model)

        if len(self.index_feature) > 0:
            inputs_extras_feature = temporal_characterisrics[0][self.index_feature].permute(1, 2, 0)
            targets_extras_feature = temporal_characterisrics[1][self.index_feature].permute(1, 2, 0)
            inputs_extras_feature = self.feature_tran(inputs_extras_feature)
            targets_extras_feature = self.feature_tran(targets_extras_feature)

            inputs_extras_embedding = torch.stack([self.embedding_modules[i](inputs_extras_onehot[i])
                                                   for i in range(len(self.num_embeddings))] + [inputs_extras_feature,
                                                                                                inputs_pe],
                                                  dim=0).unsqueeze(dim=2)
            targets_extras_embedding = torch.stack([self.embedding_modules[i](targets_extras_onehot[i])
                                                    for i in range(len(self.num_embeddings))] + [targets_extras_feature,
                                                                                                 targets_pe],
                                                   dim=0).unsqueeze(dim=2)
        else:
            inputs_extras_embedding = torch.stack([self.embedding_modules[i](inputs_extras_onehot[i])
                                                   for i in range(len(self.num_embeddings))] + [inputs_pe],
                                                  dim=0).unsqueeze(dim=2)
            targets_extras_embedding = torch.stack([self.embedding_modules[i](targets_extras_onehot[i])
                                                    for i in range(len(self.num_embeddings))] + [targets_pe],
                                                   dim=0).unsqueeze(dim=2)

        inputs_extras_embedding = self.gat(inputs_extras_embedding).squeeze(dim=1)
        targets_extras_embedding = self.gat(targets_extras_embedding).squeeze(dim=1)

        return inputs_extras_embedding, targets_extras_embedding


class SpatialEmbedding(nn.Module):
    def __init__(self, spatial_embedding_list, eigenmaps_k, d_model, noGC=False):
        super(SpatialEmbedding, self).__init__()
        self.num_embeddings, self.index_embeddings, self.index_feature = self.embeddings(spatial_embedding_list)
        self.linear = nn.Linear(eigenmaps_k, d_model)
        if len(self.index_embeddings) > 0:
            self.embedding_modules = nn.ModuleList([nn.Embedding(item, d_model) for item in self.num_embeddings])
        if len(self.index_feature) > 0:
            self.feature_tran = nn.Linear(len(self.index_feature), d_model)
            self.gat = Gated_Dynamic_Connection(len(self.index_embeddings) + 2, d_model, d_model, noGC)
        else:
            self.gat = Gated_Dynamic_Connection(len(self.index_embeddings) + 1, d_model, d_model, noGC)

    def embeddings(self, embedding_list):
        num_embeddings = []
        index_embeddings = []
        index_feature = []
        for i in range(len(embedding_list)):
            if embedding_list[i] == 1:
                index_feature.append(i)
            if embedding_list[i] > 1:
                num_embeddings.append(embedding_list[i])
                index_embeddings.append(i)
        return num_embeddings, index_embeddings, index_feature

    def forward(self, spatial_characterisrics, eigenmaps):
        '''

        :param eigenmaps: shape[N,eigenmaps_k]
        :return: u:shape[N,d_model]
        '''

        eigenmap_embedding = self.linear(eigenmaps)
        if spatial_characterisrics is not None:
            extras_list = [eigenmap_embedding]
            if len(self.index_feature) > 0:
                extras_feature = spatial_characterisrics[:, self.index_feature]
                extras_feature = self.feature_tran(extras_feature)
                extras_list.append(extras_feature)
            if len(self.index_embeddings) > 0:
                extras_onehot = spatial_characterisrics[:, self.index_embeddings].to(torch.int32)
                extras_list += [self.embedding_modules[i](extras_onehot[:, i]) for i in range(len(self.num_embeddings))]
            extras_embedding = torch.stack(extras_list, dim=0).unsqueeze(dim=1).unsqueeze(dim=1)
            spatial_embedding = self.gat(extras_embedding).squeeze(dim=0).squeeze(dim=0)

        else:
            spatial_embedding = eigenmap_embedding

        return spatial_embedding


class SpatialTemporalEmbedding(nn.Module):
    def __init__(self, d_model, noGC=False):
        super(SpatialTemporalEmbedding, self).__init__()
        self.gat = Gated_Dynamic_Connection(2, d_model, d_model, noGC=noGC)

    def forward(self, z_inputs, z_targets, u):  # (B, P, d_model), (B, Q, d_model), (N, d_model)
        '''

        :param z_inputs: shape(B, P, d_model)
        :param z_targets: shape(B, Q, d_model)
        :param u: shape(N, d_model)
        :return:
        c_inputs:shape[B,P,N,d_model]
        c_targets:shape[B,P,N,d_model]
        '''
        z_inputs = torch.stack((z_inputs,) * len(u), dim=2)
        z_targets = torch.stack((z_targets,) * len(u), dim=2)
        u_inputs = u.expand_as(z_inputs)
        u_targets = u.expand_as(z_targets)

        c_inputs = self.gat(torch.stack((z_inputs, u_inputs), dim=0))
        c_targets = self.gat(torch.stack((z_targets, u_targets), dim=0))

        return c_inputs, c_targets


class MetaLearner(nn.Module):
    def __init__(self, d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=3):
        '''

        :param d_model:
        :param d_k:
        :param d_hidden_mt:
        :param num_heads:
        :param num_weight_matrices: 生产转换矩阵数量，自注意力为3
        '''
        super(MetaLearner, self).__init__()
        self.num_weight_matrices = num_weight_matrices
        self.num_heads = num_heads
        self.d_k = d_k

        self.linear1 = nn.Linear(d_model, d_hidden_mt)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_hidden_mt, num_weight_matrices * num_heads * d_k * d_model)

    def forward(self, c_inputs):
        '''

        :param c_inputs: shape[B,P,N,d_model]
        :return: W shape[3,B,P,N,H,d_k,d_model]
        '''
        B, P, N, d_model = c_inputs.shape
        out = self.relu(self.linear1(c_inputs))
        out = self.linear2(out)
        out = out.reshape((B, P, N, self.num_weight_matrices, self.num_heads, self.d_k, d_model))
        out = out.permute((3, 0, 1, 2, 4, 5, 6))  # (num_weight_matrices, B, P, N, num_heads, d_k, d_model

        return out


class Gated_Dynamic_Connection(nn.Module):
    def __init__(self, num_gate, d_k, d_model, noGC=False):
        super(Gated_Dynamic_Connection, self).__init__()
        self.noGC = noGC
        if not noGC:
            self.Weight1 = nn.Parameter(torch.Tensor(num_gate, d_k, d_model))
            self.Weight2 = nn.Parameter(torch.Tensor(num_gate, d_k, d_model))
            nn.init.xavier_normal_(self.Weight1)
            nn.init.xavier_normal_(self.Weight2)
        else:
            self.Linear = nn.Linear(d_k * num_gate, d_model)

    def forward(self, data):
        if not self.noGC:

            data = data.permute(1, 2, 3, 0, 4).unsqueeze(-2)
            data_out = torch.matmul(data, self.Weight1).transpose(-1, -3)
            data_softmax = nonan_softmax(F.relu(torch.matmul(data, self.Weight2)).transpose(-1, -3), dim=-1).transpose(
                -1, -2)
            out = torch.matmul(data_out, data_softmax).squeeze(dim=-1).squeeze(dim=-1)
        else:
            nn, B, P, N, C = data.shape
            data = data.reshape(B, P, N, -1)
            out = self.Linear(data)
        return out


class Adaptive_Graph(nn.Module):
    def __init__(self, d_model, d_hidden_gm, top_k):
        super(Adaptive_Graph, self).__init__()
        self.top_k = top_k
        self.weight = nn.Parameter(torch.Tensor(2, d_model, d_hidden_gm))
        nn.init.xavier_normal_(self.weight)

    def forward(self, c_input):
        A1 = torch.matmul(c_input, self.weight[0])
        A2 = torch.matmul(c_input, self.weight[1]).transpose(2, 3)
        gen_graph = torch.matmul(A1, A2)
        gen_graph = self.top_kf(F.relu(gen_graph), self.top_k)
        gen_graph = nozero_softmax(gen_graph, dim=-1)
        return gen_graph

    def top_kf(self, matrix, top_k):
        top_k_per_row = matrix.topk(top_k, dim=3)[0]
        min = top_k_per_row[:, :, :, top_k - 1:top_k]
        matrix2 = matrix.clone()
        matrix2[matrix < min] = 0.
        return matrix2


class GroupNorm(nn.Module):
    def __init__(self):
        super(GroupNorm, self).__init__()

    def forward(self, x):
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        x = (x - mean) / (var).sqrt()
        nan = torch.isnan(x)
        x[nan == True] = 0.
        return x


class TemporalSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_hidden_mt, num_heads, noML, noGC, causal=False):
        '''

        :param d_model:
        :param d_k:
        :param d_hidden_mt:
        :param num_heads:
        :param noML: 是否采用meta 采用则用meta生产转换矩阵，不采用则使用线性层
        :param causal: 是否采用时间掩码
        '''
        super(TemporalSelfAttention, self).__init__()
        self.noML = noML
        self.noGC = noGC
        self.causal = causal

        if self.noML:
            self.num_heads = num_heads
            self.d_k = d_k
            self.linear_q = nn.Linear(d_model, d_k * num_heads, bias=False)
            self.linear_k = nn.Linear(d_model, d_k * num_heads, bias=False)
            self.linear_v = nn.Linear(d_model, d_k * num_heads, bias=False)
        else:
            self.meta_learner = MetaLearner(d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=3)
        self.gat = Gated_Dynamic_Connection(num_gate=num_heads, d_k=d_k, d_model=d_model, noGC=noGC)
        self.swish = Swish(d_model)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, inputs, c_inputs):
        '''

        :param inputs: shape：[B,P,N,d_model]
        :param c_inputs: shape:[B,P,N,d_model]
        :return: out：[B,P,N,d_model]
        '''

        if self.noML:
            B, P, N, _ = inputs.shape
            Q = self.linear_q(inputs).reshape((B, P, N, self.num_heads, self.d_k))
            K = self.linear_k(inputs).reshape((B, P, N, self.num_heads, self.d_k))
            V = self.linear_v(inputs).reshape((B, P, N, self.num_heads, self.d_k))
        else:
            W_q, W_k, W_v = self.meta_learner(c_inputs)  # (B, P, N, H, d_k, d_model)
            Q = multihead_linear_transform(W_q, inputs)  # (B, P, N, H, d_k)
            K = multihead_linear_transform(W_k, inputs)  # (B, P, N, H, d_k)
            V = multihead_linear_transform(W_v, inputs)  # (B, P, N, H, d_k)
        out = multihead_temporal_attention(Q, K, V, causal=self.causal)  # (B, P, N, d_model)
        out = self.gat(out)  # (B, P, N, d_model)
        out = self.swish(inputs, out)
        out = self.layer_norm(out + inputs)  # (B, P, N, d_model)

        return out


class TemporalSelfRetention(nn.Module):
    def __init__(self, d_model, d_k, d_hidden_mt, num_heads, noML, noGC):
        '''

        :param d_model:
        :param d_k:
        :param d_hidden_mt:
        :param num_heads:
        :param noML: 是否采用meta 采用则用meta生产转换矩阵，不采用则使用线性层
        '''
        super(TemporalSelfRetention, self).__init__()
        self.noML = noML
        self.noGC = noGC
        self.d_k = d_k
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_len = 12
        self.D_and_pos(num_heads, self.max_len)

        self.kv = 0.
        if self.noML:
            self.num_heads = num_heads
            self.d_k = d_k
            self.linear_q = nn.Linear(d_model, d_k * num_heads, bias=False)
            self.linear_k = nn.Linear(d_model, d_k * num_heads, bias=False)
            self.linear_v = nn.Linear(d_model, d_k * num_heads, bias=False)
        else:
            self.meta_learner = MetaLearner(d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=3)
        self.gat = Gated_Dynamic_Connection(num_gate=num_heads, d_k=d_k, d_model=d_model, noGC=noGC)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.group_norm = GroupNorm()
        self.swish = Swish(d_model)

    def D_and_pos(self, num_heads, len):
        r = 1. - 2. ** (-5. - torch.arange(0, num_heads))
        self.register_buffer('r', r)
        D = torch.zeros([num_heads, len, len], dtype=torch.float32)
        for h in range(num_heads):
            for n in range(len):
                for m in range(len):
                    if n >= m:
                        D[h, n, m] = r[h] ** (n - m)
        D_sum = torch.sqrt(torch.sum(D, dim=-1, keepdim=True))
        D = D / D_sum
        self.register_buffer('D', D)  # 不可学参数
        self.theta = 1
        n = torch.arange(1, len + 1)  # 生成从0到M-1的整数序列
        pos = torch.exp(1j * n * self.theta)  # 计算einθ
        real_pos = torch.real(pos)
        imag_pos = torch.imag(pos)
        pos = torch.stack([real_pos, imag_pos], dim=0)
        self.register_buffer('pos', pos)  # 不可学参数

    def init_kv(self):
        self.kv = 0.

    def forward(self, inputs, c_inputs, Num):
        '''

        :param inputs: shape：[B,P,N,d_model]
        :param c_inputs: shape:[B,P,N,d_model]
        :return: out：[B,P,N,d_model]
        '''
        B, T, N, d_model = inputs.shape
        if self.noML:
            B, P, N, _ = inputs.shape
            Q = self.linear_q(inputs).reshape((B, P, N, self.num_heads, self.d_k))
            K = self.linear_k(inputs).reshape((B, P, N, self.num_heads, self.d_k))
            V = self.linear_v(inputs).reshape((B, P, N, self.num_heads, self.d_k))
        else:
            pass
            W_q, W_k, W_v = self.meta_learner(c_inputs)  # (B, P, N, H, d_k, d_model)
            Q = multihead_linear_transform(W_q, inputs)  # (B, P, N, H, d_k)
            K = multihead_linear_transform(W_k, inputs)  # (B, P, N, H, d_k)
            V = multihead_linear_transform(W_v, inputs)  # (B, P, N, H, d_k)
        if T > 1 or Num == None:
            Q = Q * (self.pos[:, :T].reshape(2, 1, -1, 1, 1, 1))
            K = K * (self.pos[:, :T].reshape(2, 1, -1, 1, 1, 1))
            out = multihead_temporal_regnet(Q, K, V, self.D[:, :T, :T]).reshape(-1, self.num_heads, self.d_k)
            out = self.group_norm(out).reshape(B, -1, T, N, self.d_k)
        else:
            Q = Q * self.pos[:, Num - 1].reshape(2, 1, 1, 1, 1, 1)
            K = K * self.pos[:, Num - 1].reshape(2, 1, 1, 1, 1, 1)
            Q = Q.permute((0, 1, 4, 3, 2, 5))
            K = K.permute((0, 1, 4, 3, 5, 2))
            V = V.permute((0, 3, 2, 1, 4))
            self.kv = self.r.reshape(-1, 1, 1, 1) * self.kv + torch.matmul(K, V)
            out = torch.sum(torch.matmul(Q, self.kv), dim=0).reshape(-1, self.num_heads, self.d_k)
            out = self.group_norm(out).reshape(B, -1, T, N, self.d_k)
        out = 3 * torch.tanh(out.transpose(0, 1))
        out = self.gat(out)  # (B, P, N, d_model)
        out = self.swish(inputs, out)
        out = self.layer_norm(out + inputs)  # (B, P, N, d_model

        return out


class SpatialSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_hidden_mt, num_heads, which_transition_matrices, dropout, noML, noAG,
                 noGC):
        '''

        :param d_model:
        :param d_k:
        :param d_hidden_mt:
        :param num_heads:
        :param which_transition_matrices: list,len=特征图数量，[是否采用的意思]
        :param dropout:
        :param noML:
        '''
        super(SpatialSelfAttention, self).__init__()
        if "get_para":
            self.which_transition_matrices = which_transition_matrices
            self.num_transition_matrices = sum(which_transition_matrices)
            assert self.num_transition_matrices > 0
            self.noML = noML
            self.noAG = noAG
            if self.noAG:
                hid = self.num_transition_matrices
            else:
                hid = self.num_transition_matrices + 1
        if self.noML:
            self.num_heads = num_heads
            self.d_k = d_k
            self.linear_q = nn.ModuleList([nn.Linear(d_model, d_model, bias=False)
                                           for _ in range(hid)])
            self.linear_k = nn.ModuleList([nn.Linear(d_model, d_model, bias=False)
                                           for _ in range(hid)])
            self.linear_v = nn.ModuleList([nn.Linear(d_model, d_model, bias=False)
                                           for _ in range(hid)])
        else:
            self.meta_learners = nn.ModuleList([MetaLearner(
                d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=3)
                for _ in range(hid)])
        self.gat1 = nn.ModuleList(
            [Gated_Dynamic_Connection(num_gate=num_heads, d_k=d_k, d_model=d_model, noGC=noGC) for _ in
             range(hid)])
        self.gat2 = Gated_Dynamic_Connection(num_gate=hid, d_k=d_model, d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.swish = Swish(d_model)

    def forward(self, inputs, c_inputs, transition_matrices, adaptive_graph):
        '''

        :param inputs: shape[B,P,N,d_model]
        :param c_inputs: shape[B,P,N,d_model]
        :param transition_matrices: 转换矩阵，与图数量相同[num_maps,N,N]
        :return: shape[B,P,N,d_model]
        '''
        assert transition_matrices.shape[0] == len(self.which_transition_matrices)
        transition_matrices = transition_matrices[self.which_transition_matrices]

        out = []
        fu = []
        for i in range(self.num_transition_matrices):
            fu.append(jit.fork(self.fork, i, inputs, c_inputs, transition_matrices))
        if not self.noAG:
            fu.append(jit.fork(self.forkgg, inputs, c_inputs, adaptive_graph))
        for i in fu:
            i = jit.wait(i)
            out.append(i)

        out = torch.stack(out, dim=0)
        out = self.gat2(out)
        out = self.dropout(out)
        out = self.swish(inputs, out)
        out = self.layer_norm(out + inputs)  # (B, P, N, d_model)

        return out

    def fork(self, i, inputs, c_inputs, transition_matrices):
        if self.noML:
            B, P, N, _ = inputs.shape
            Q = self.linear_q[i](inputs).reshape((B, P, N, self.num_heads, self.d_k))
            K = self.linear_k[i](inputs).reshape((B, P, N, self.num_heads, self.d_k))
            V = self.linear_v[i](inputs).reshape((B, P, N, self.num_heads, self.d_k))
        else:
            W_q, W_k, W_v = self.meta_learners[i](c_inputs)  # (B, P, N, H, d_k, d_model)
            Q = multihead_linear_transform(W_q, inputs)  # (B, P, N, H, d_k)
            K = multihead_linear_transform(W_k, inputs)  # (B, P, N, H, d_k)
            V = multihead_linear_transform(W_v, inputs)  # (B, P, N, H, d_k)
        mid_result = multihead_spatial_attention_predefine_graph(Q, K, V, transition_matrices[i])
        return self.gat1[i](mid_result)

    def forkgg(self, inputs, c_inputs, adaptive_graph):
        if self.noML:
            B, P, N, _ = inputs.shape
            Q = self.linear_q[-1](inputs).reshape((B, P, N, self.num_heads, self.d_k))
            K = self.linear_k[-1](inputs).reshape((B, P, N, self.num_heads, self.d_k))
            V = self.linear_v[-1](inputs).reshape((B, P, N, self.num_heads, self.d_k))
        else:
            W_q, W_k, W_v = self.meta_learners[-1](c_inputs)  # (B, P, N, H, d_k, d_model)
            Q = multihead_linear_transform(W_q, inputs)  # (B, P, N, H, d_k)
            K = multihead_linear_transform(W_k, inputs)  # (B, P, N, H, d_k)
            V = multihead_linear_transform(W_v, inputs)  # (B, P, N, H, d_k)
        mid_result = multihead_spatial_attention_adaptive_graph(Q, K, V, adaptive_graph)
        return self.gat1[-1](mid_result)


class TemporalEncoderDecoderAttention(nn.Module):
    def __init__(self, d_model, d_k, d_hidden_mt, num_heads, noML, noGC):
        super(TemporalEncoderDecoderAttention, self).__init__()
        self.noML = noML

        if self.noML:
            self.num_heads = num_heads
            self.d_k = d_k
            self.linear_q = nn.Linear(d_model, d_model, bias=False)
        else:
            # 只需要学习Q对应的
            self.meta_learner = MetaLearner(d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=1)

        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.gat = Gated_Dynamic_Connection(num_gate=num_heads, d_k=d_k, d_model=d_model, noGC=noGC)
        self.swish = Swish(d_model)

    def forward(self, inputs, enc_K, enc_V, c_targets):
        '''

        :param inputs: [B,P,N,d_model]
        :param enc_K: [B,P,N,H,d_model]
        :param enc_V: [B,P,N,H,d_model]
        :param c_targets: [B,P,N,d_model]
        :return: [B,P,N,d_model]
        '''
        if self.noML:
            B, P1, N, _ = inputs.shape
            Q = self.linear_q(inputs).reshape((B, P1, N, self.num_heads, self.d_k))
        else:
            W_q, = self.meta_learner(c_targets)  # (B, P1, N, H, d_k, d_model)
            Q = multihead_linear_transform(W_q, inputs)  # (B, P1, N, H, d_k)
        # 不采用掩码
        out = multihead_temporal_attention(Q, enc_K, enc_V, causal=False)  # (B, P1, N, d_model)
        out = self.gat(out)  # (B, P1, N, d_model)
        out = self.swish(inputs, out)
        out = self.layer_norm(out + inputs)  # (B, P1, N, d_model)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_hidden_ff, d_model)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, inputs):
        out = self.relu(self.linear1(inputs))
        out = self.linear2(out)
        out = self.layer_norm(out + inputs)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, cfgs):
        super(EncoderLayer, self).__init__()
        if "get_para":
            d_model = cfgs['d_model']
            d_k = cfgs['d_k']
            d_hidden_mt = cfgs['d_hidden_mt']
            d_hidden_ff = cfgs['d_hidden_ff']
            num_heads = cfgs['num_heads']
            which_transition_matrices = cfgs['which_transition_matrices']
            dropout = cfgs['dropout']
            self.noTSA = cfgs.get('noTSA', False)
            self.noSSA = cfgs.get('noSSA', False)
            self.noML = cfgs.get('noML', False)
            self.noTE = cfgs.get('noTE', False)
            self.noSE = cfgs.get('noSE', False)
            self.noAG = cfgs.get('noAG', False)
            self.noGC = cfgs.get('noGC', False)
            if self.noTE and self.noSE:
                self.noML = True

        if not self.noTSA:
            self.temporal_self_attention = TemporalSelfAttention(
                d_model, d_k, d_hidden_mt, num_heads, self.noML, noGC=self.noGC)
        if not self.noSSA:
            self.spatial_self_attention = SpatialSelfAttention(
                d_model, d_k, d_hidden_mt, num_heads, which_transition_matrices, dropout, self.noML, self.noAG,
                noGC=self.noGC)
        self.feed_forward = FeedForward(d_model, d_hidden_ff)

    def forward(self, inputs, c_inputs, transition_matrices, adaptive_graph_inputs):

        out = inputs
        if not self.noTSA:
            out = self.temporal_self_attention(out, c_inputs)
        if not self.noSSA:
            out = self.spatial_self_attention(out, c_inputs, transition_matrices, adaptive_graph_inputs)
        out = self.feed_forward(out)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, cfgs):
        super(DecoderLayer, self).__init__()
        if "get_para":
            d_model = cfgs['d_model']
            d_k = cfgs['d_k']
            d_hidden_mt = cfgs['d_hidden_mt']
            d_hidden_ff = cfgs['d_hidden_ff']
            num_heads = cfgs['num_heads']
            which_transition_matrices = cfgs['which_transition_matrices']
            dropout = cfgs['dropout']
            self.noTSA = cfgs.get('noTSA', False)
            self.noSSA = cfgs.get('noSSA', False)
            self.noML = cfgs.get('noML', False)
            self.noAG = cfgs.get('noAG', False)
            self.noTE = cfgs.get('noTE', False)
            self.noSE = cfgs.get('noSE', False)
            self.noGC = cfgs.get('noGC', False)
            if self.noTE and self.noSE:
                self.noML = True
        # 考虑掩码
        if not self.noTSA:
            self.temporal_self_attention = TemporalSelfRetention(
                d_model, d_k, d_hidden_mt, num_heads, self.noML, noGC=self.noGC)

        if not self.noSSA:
            self.spatial_self_attention = SpatialSelfAttention(
                d_model, d_k, d_hidden_mt, num_heads, which_transition_matrices, dropout, self.noML, self.noAG,
                noGC=self.noGC)
        self.temporal_encoder_decoder_attention = TemporalEncoderDecoderAttention(
            d_model, d_k, d_hidden_mt, num_heads, self.noML, noGC=self.noGC)
        self.feed_forward = FeedForward(d_model, d_hidden_ff)

    def forward(self, inputs, enc_K, enc_V, c_targets, transition_matrices, gen_graph_targets, num):

        out = inputs
        if not self.noTSA:
            out = self.temporal_self_attention(out, c_targets, num)
        if not self.noSSA:
            out = self.spatial_self_attention(out, c_targets, transition_matrices, gen_graph_targets)
        out = self.temporal_encoder_decoder_attention(out, enc_K, enc_V, c_targets)
        out = self.feed_forward(out)
        return out


class Project(nn.Module):
    def __init__(self, d_model, num_features):
        super(Project, self).__init__()
        self.linear = nn.Linear(d_model, num_features)

    def forward(self, inputs):
        out = self.linear(inputs)

        return out


class Encoder(nn.Module):
    def __init__(self, cfgs):
        super(Encoder, self).__init__()
        if "get_para":
            num_features = cfgs['num_features']
            d_model = cfgs['d_model']
            num_encoder_layers = cfgs['num_encoder_layers']
            self.noML = cfgs.get('noML', False)
            self.noTE = cfgs.get('noTE', False)
            self.noSE = cfgs.get('noSE', False)

        self.linear = nn.Linear(num_features, d_model)
        self.layer_stack = nn.ModuleList([EncoderLayer(cfgs) for _ in range(num_encoder_layers)])

    def init_kv(self):
        for lay in self.layer_stack:
            lay.temporal_self_attention.init_kv()

    def forward(self, inputs, c_inputs, transition_matrices, adaptive_graph_inputs):
        if self.noML and ((not self.noTE) or (not self.noSE)):
            out = F.relu(self.linear(inputs) + c_inputs)
        else:
            out = F.relu(self.linear(inputs))
        skip = 0
        for encoder_layer in self.layer_stack:
            out = encoder_layer(out, c_inputs, transition_matrices, adaptive_graph_inputs)
            skip += out

        return skip


class Decoder(nn.Module):
    def __init__(self, cfgs):
        super(Decoder, self).__init__()
        if "get_para":
            d_model = cfgs['d_model']
            d_k = cfgs['d_k']
            d_hidden_mt = cfgs['d_hidden_mt']
            num_features = cfgs['num_features']
            num_heads = cfgs['num_heads']
            num_decoder_layers = cfgs['num_decoder_layers']
            self.out_len = cfgs['out_len']
            self.use_curriculum_learning = cfgs['use_curriculum_learning']
            self.cl_decay_steps = cfgs['cl_decay_steps']
            self.noAG = cfgs.get('noAG', False)
            self.noML = cfgs.get('noML', False)
            self.noTE = cfgs.get('noTE', False)
            self.noSE = cfgs.get('noSE', False)
            self.noTSA = cfgs.get('noTSA', False)
        if self.noML or (self.noTE and self.noSE):
            self.num_heads = num_heads
            self.d_k = d_k
            self.linear_k = nn.Linear(d_model, d_model, bias=False)
            self.linear_v = nn.Linear(d_model, d_model, bias=False)
        else:
            self.meta_learner = MetaLearner(d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=2)

        self.linear = nn.Linear(num_features, d_model)
        self.layer_stack = nn.ModuleList([DecoderLayer(cfgs) for _ in range(num_decoder_layers)])
        self.project = Project(d_model, num_features)

    def _compute_sampling_threshold(self, batches_seen):
        '''
        衰减的因子，用于控制真实值与预测值的数量（真实输入，预测输入）
        :param batches_seen:
        :return:
        '''
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def init_kv(self):
        for lay in self.layer_stack:
            if not self.noTSA:
                lay.temporal_self_attention.init_kv()

    def stationary(self, out, enc_K, enc_V, c_targets, transition_matrices, gen_graph_targets):
        out1 = out
        skip = 0
        for decoder_layer in self.layer_stack:
            out = decoder_layer(out, enc_K, enc_V, c_targets, transition_matrices, gen_graph_targets, num=None)
            skip += out
        a = skip
        outputs = []
        for i in range(out.shape[1]):
            skip = 0
            out = out1[:, i:(i + 1), :, :]
            for decoder_layer in self.layer_stack:
                out = decoder_layer(out, enc_K, enc_V, c_targets[:, i:(i + 1), :, :], transition_matrices,
                                    gen_graph_targets[:, i:(i + 1), :, :], num=i + 1)
                skip += out
            outputs.append(skip)
        outputs = torch.cat(outputs, dim=1)
        b = outputs
        mape = 200 * torch.sum(torch.abs(a - b)) / (torch.sum(torch.abs(a) + torch.abs(b)))
        print(mape.item())

    def forward(self, inputs, targets, c_inputs, c_targets, enc_outputs, transition_matrices, gen_graph_targets,
                batches_seen):

        if self.noML or (self.noTE and self.noSE):
            B, P, N, _ = enc_outputs.shape
            enc_K = self.linear_k(enc_outputs).reshape((B, P, N, self.num_heads, self.d_k))
            enc_V = self.linear_v(enc_outputs).reshape((B, P, N, self.num_heads, self.d_k))
        else:
            W_k, W_v = self.meta_learner(c_inputs)  # (B, P, N, H, d_k, d_model)
            enc_K = multihead_linear_transform(W_k, enc_outputs)  # (B, P, N, H, d_k)
            enc_V = multihead_linear_transform(W_v, enc_outputs)  # (B, P, N, H, d_k)

        use_targets = False  # 使用真值
        if self.training and (targets is not None) and self.use_curriculum_learning:  # 课程设置，训练模式有真值，用真值
            c = np.random.uniform(0, 1)
            if c < self._compute_sampling_threshold(batches_seen):  # 刚开始全用，后来真值越来越小，随着batch_seen的增加，范围减小
                use_targets = True

        if use_targets is True:
            dec_inputs = torch.cat((inputs[:, -1, :, :].unsqueeze(1), targets[:, :-1, :, :]),
                                   dim=1)  # (B, Q, N, C)
            if self.noML and ((not self.noTE) or (not self.noSE)):
                out = F.relu(self.linear(dec_inputs) + c_targets)
            else:
                out = F.relu(self.linear(dec_inputs))  # (B, Q, N, d_model)
            # self.stationary(out, enc_K, enc_V, c_targets, transition_matrices, gen_graph_targets)
            skip = 0
            for decoder_layer in self.layer_stack:
                out = decoder_layer(out, enc_K, enc_V, c_targets, transition_matrices, gen_graph_targets, num=None)
                skip += out
            outputs = self.project(skip)  # (B, Q, N, C)
        else:
            self.init_kv()
            dec_inputs = inputs[:, -1, :, :].unsqueeze(1)  # (B, 1, N, C)
            outputs = []
            for i in range(self.out_len):
                if self.noML and ((not self.noTE) or (not self.noSE)):
                    out = F.relu(self.linear(dec_inputs) + c_targets[:, i:(i + 1), :, :])
                else:
                    out = F.relu(self.linear(dec_inputs))  # (B, *, N, d_model)
                skip = 0
                for decoder_layer in self.layer_stack:
                    if (not self.noTE) or (not self.noSE):
                        if not self.noAG:
                            out = decoder_layer(out, enc_K, enc_V, c_targets[:, i:(i + 1), :, :], transition_matrices,
                                                gen_graph_targets[:, i:(i + 1), :, :], num=i + 1)
                        else:
                            out = decoder_layer(out, enc_K, enc_V, c_targets[:, i:(i + 1), :, :], transition_matrices,
                                                None, num=i + 1)
                    else:
                        out = decoder_layer(out, enc_K, enc_V, None, transition_matrices, None, num=i + 1)
                    skip += out
                out = self.project(skip)  # (B, *, N, C)
                outputs.append(out)
                dec_inputs = out
            outputs = torch.cat(outputs, dim=1)  # (B, Q, N, C)

        return outputs


class AGMGRN(nn.Module):
    def __init__(self, cfgs):
        super(AGMGRN, self).__init__()
        if "get_para":
            d_model = cfgs['d_model']
            temporal_num_embeddings = cfgs['temporal_num_embeddings']
            spatial_num_embeddings = cfgs['spatial_num_embeddings']
            eigenmaps_k = cfgs['eigenmaps_k']
            self.noTE = cfgs.get('noTE', False)
            self.noSE = cfgs.get('noSE', False)
            self.noGC = cfgs.get('noGC', False)
            self.noAG = cfgs.get('noAG', False)
            self.batches_seen = 0
            self.in_len = cfgs['in_len']
            self.out_len = cfgs['out_len']
            if "NO":
                if self.noTE and self.noSE:
                    self.noAG = True
                if not self.noAG:
                    self.adaptive_graph = Adaptive_Graph(d_model=cfgs['d_model'], d_hidden_gm=cfgs['d_hidden_gm'],
                                                         top_k=cfgs['top_k'])
                if not self.noTE:
                    self.temporal_embedding = TemporalEmbedding(temporal_num_embeddings, d_model, self.noGC)
                if not self.noSE:
                    self.spatial_embedding = SpatialEmbedding(spatial_num_embeddings, eigenmaps_k, d_model, self.noGC)

                if (not self.noTE) and (not self.noSE):
                    self.spatial_temporal_embedding = SpatialTemporalEmbedding(d_model, self.noGC)

        self.encoder = Encoder(cfgs)
        self.decoder = Decoder(cfgs)

    def init_kv(self):
        self.encoder.init_kv()
        self.Num = 1

    def forward(self, inputs, targets, extras, **statics):
        a = 1
        if "ste_and_ag":
            if not self.noTE:
                z_inputs, z_targets = self.temporal_embedding(
                    [extras['input_temporal_characterisrics'].transpose(0, 1),
                     extras['target_temporal_characterisrics'].transpose(0, 1)])  # (B, P, d_model), (B, Q, d_model)
            if not self.noSE:
                u = self.spatial_embedding(statics['spatial_characterisrics'], statics['eigenmaps'])  # (N, d_model)
            if (not self.noTE) and (not self.noSE):
                c_inputs, c_targets = self.spatial_temporal_embedding(
                    z_inputs, z_targets, u)  # (B, P, N, d_model), (B, Q, N, d_model)
            elif self.noTE and (not self.noSE):
                B = inputs.size(0)
                P = self.in_len
                Q = self.out_len
                N = u.size(0)
                d_model = u.size(1)
                c_inputs = u.expand(B, P, N, d_model)
                c_targets = u.expand(B, Q, N, d_model)
            elif (not self.noTE) and self.noSE:
                N = inputs.size(2)
                c_inputs = torch.stack((z_inputs,) * N, dim=2)
                c_targets = torch.stack((z_targets,) * N, dim=2)
            else:
                c_inputs = None
                c_targets = None
            if not self.noAG:
                gen_graph_inputs = self.adaptive_graph(c_inputs)
                gen_graph_targets = self.adaptive_graph(c_targets)
            else:
                gen_graph_inputs = None
                gen_graph_targets = None
        transition_matrices = statics['transition_matrices']
        enc_outputs = self.encoder(inputs, c_inputs, transition_matrices, gen_graph_inputs)

        outputs = self.decoder(inputs, targets, c_inputs, c_targets, enc_outputs, transition_matrices,
                               gen_graph_targets, self.batches_seen)
        if self.training:
            self.batches_seen += 1

        return outputs
