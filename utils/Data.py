from datetime import datetime
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
import torch
import os
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from utils.Auxiliary import *


# compute_eigenmaps
def compute_eigenmaps(adj_mx, k):
    A = adj_mx.copy()
    row, col = A.nonzero()
    A[row, col] = A[col, row] = 1  # 0/1 matrix, symmetric

    # n_components = connected_components(csr_matrix(A), directed=False, return_labels=False)
    # assert n_components == 1  # the graph should be connected

    n = A.shape[0]
    # A = zero_diagonals(A)
    D = np.sum(A, axis=1) ** (-1 / 2)
    # D = np.nan_to_num(D, nan=1.0, posinf=1.0, neginf=1.0)
    L = np.eye(n) - (A * D).T * D  # normalized Laplacian

    _, v = eigh(L)
    eigenmaps = v[:, 1:(k + 1)]  # eigenvectors corresponding to the k smallest non-trivial eigenvalues

    return eigenmaps


# zero_diagonals
def zero_diagonals(x):
    y = x.copy()
    y[np.diag_indices_from(y)] = 0

    return y


# row_normalize
def row_normalize(A):
    A = A.astype(np.float32)
    S = (A.T * np.sum(A, axis=1) ** (-1)).T

    return S


# add_self_loop
def add_self_loop(A):
    B = A.copy()
    B[np.diag_indices_from(B)] = 1.0

    return B


# compute_normalized_laplacian
def compute_normalized_laplacian(adj_mx):
    A = zero_diagonals(adj_mx)  # remove self-loops
    A = np.maximum(A, A.T)  # symmetrization

    D = A.sum(1)  # degrees
    D[D == 0] = np.inf
    D_rs = D ** (-1 / 2)

    n = A.shape[0]
    I = np.eye(n)
    normalized_L = I - (A * D_rs).T * D_rs  # I - D^(-1/2)AD^(-1/2)

    return normalized_L


# compute_scaled_laplacian
def compute_scaled_laplacian(adj_mx):
    n = adj_mx.shape[0]
    I = np.eye(n)

    normalized_L = compute_normalized_laplacian(adj_mx)
    w, _ = eigh(normalized_L)
    lambda_max = w.max()

    scaled_L = 2 * normalized_L / lambda_max - I

    return scaled_L


# compute_mean_std
def compute_mean_std(data):
    mean = np.mean(data, axis=(0, 1, 2))
    std = np.std(data, axis=(0, 1, 2))
    return mean, std


# generate_time_and_flags
def generate_time_and_flags(start_data, end_data, interval, time_ranges, start_time="00:00", end_time="23:59"):
    """
    生成时间点序列和标记向量。

    参数:
    - start_data: 起始日期 (格式: 'YYYY-MM-DD')
    - end_data: 结束日期 (格式: 'YYYY-MM-DD')
    - interval: 时间间隔（分钟）
    - time_ranges: 开始和结束标记时间的列表（格式：[[开始时间，结束时间], [开始时间，结束时间]]，时间格式：'HH：MM'）
    - start_time: 每天的开始时间 (格式: 'HH:MM')
    - end_time: 每天的结束时间 (格式: 'HH:MM')

    返回:
    - time_vector: 时间点序列（numpy数组）
    - flag_vector: 标记向量（numpy数组，0或1）
    """
    # 转换日期为datetime对象
    start_date = datetime.strptime(start_data, "%Y-%m-%d")
    end_date = datetime.strptime(end_data, "%Y-%m-%d")

    # 计算每天的开始时间和结束时间的 timedelta
    start_time_delta = timedelta(hours=int(start_time.split(":")[0]), minutes=int(start_time.split(":")[1]))
    end_time_delta = timedelta(hours=int(end_time.split(":")[0]), minutes=int(end_time.split(":")[1]))

    time_deltas = []
    # 如果given time_ranges存在，转换成时间间隔
    if time_ranges:
        for time_range in time_ranges:
            time1 = time_range[0]
            time2 = time_range[1]
            time1_delta = timedelta(hours=int(time1.split(":")[0]), minutes=int(time1.split(":")[1]))
            time2_delta = timedelta(hours=int(time2.split(":")[0]), minutes=int(time2.split(":")[1]))
            time_deltas.append((time1_delta, time2_delta))

    # 计算时间点
    current_date = start_date
    time_list = []
    while current_date <= end_date:
        day_start = current_date + start_time_delta
        day_end = current_date + end_time_delta
        current_time = day_start
        while current_time <= day_end:
            time_list.append(current_time)
            current_time += timedelta(minutes=interval)
        current_date += timedelta(days=1)

    time_vector = np.array(time_list)

    # 初始化标记向量
    flag_vector = np.zeros(time_vector.shape, dtype=int)

    # 遍历时间向量，判断是否在对应的时间范围内
    for i, t in enumerate(time_vector):
        day_start = datetime(t.year, t.month, t.day)  # 每一天的开始时间
        for time1_delta, time2_delta in time_deltas:
            # 计算每天的绝对时间范围
            abs_time1 = day_start + time1_delta
            abs_time2 = day_start + time2_delta

            if abs_time1 <= t <= abs_time2:
                flag_vector[i] = 1

    return flag_vector


# normalize
def normalize(data, mean, std):
    data_nor = (data - mean) / std
    return data_nor


def top_k_and_normalize(matrix, k):
    top_k_per_row = np.partition(-matrix, k - 1, axis=1)[:, :k]
    top_k_per_row = -top_k_per_row
    result = np.zeros(matrix.shape)
    sum_top_k = np.sum(top_k_per_row, axis=1, keepdims=True)
    sum_top_k[sum_top_k == 0.] = 1.
    normalize_values = top_k_per_row / sum_top_k
    row_idx = np.arange(matrix.shape[0])
    col_idx = np.argpartition(-matrix, k - 1, axis=1)[:, :k]
    result[row_idx[:, None], col_idx] = normalize_values
    return result


def compute_graph_sml(data):
    n, c = data.shape
    graph_sml = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            a = np.linalg.norm(data[i] - data[j]) ** 2
            b = np.minimum(np.linalg.norm(data[i]) ** 2, np.linalg.norm(data[j]) ** 2)
            c = np.exp(-a / b)
            graph_sml[j, i] = graph_sml[i, j] = c

    return graph_sml


class Metro():
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.num_intervals = 73
        self.start_time = '5:30'
        self.interval = 15
        self.root = self.cfgs['root']
        self.top_k = self.cfgs['top_k']
        self.eigenmaps_k = self.cfgs.get('eigenmaps_k', 8)

        data, _ = self.open_data(self.root, 'train', self.cfgs['debug'])
        _, self.in_len, self.num_nodes, self.num_features = data['x'].shape
        self.out_len = data['y'].shape[1]
        self.mean, self.std = compute_mean_std(data['x'])

    def open_data(self, root, split, debug):
        with open(osp.join(root, f'{split}.pkl'), 'rb') as f:
            data = pickle.load(f)
        y_time = data['ytime']
        y_time = (y_time - y_time.astype('datetime64[D]')).astype('timedelta64[s]')

        time_ranges = self.cfgs['time_ranges']
        time_flag = 0
        for time_range in time_ranges:
            start_time = time_range[0]
            end_time = time_range[1]
            start_time = np.timedelta64(
                int(int(start_time.split(':')[0]) * 60 * 60 + int(start_time.split(':')[1]) * 60), 's')  # "8:30"
            end_time = np.timedelta64(
                int(int(end_time.split(':')[0]) * 60 * 60 + int(end_time.split(':')[1]) * 60), 's')  # "8:30"
            result_matrix = ((y_time >= start_time) & (y_time <= end_time)).astype(int)
            time_flag += result_matrix

        if debug:
            for key in data.keys():
                len = max(min(int(data[key].shape[0] * 0.1), 100), 20)
                data[key] = data[key][:len]
                time_flag = time_flag[:len]

        return data, time_flag

    def gen_complete_time_series(self, data, num_intervals, in_len, out_len, num_nodes, num_features):
        x, y = data['x'], data['y']
        num_samples = x.shape[0]  # number of samples
        m = num_intervals - in_len - out_len + 1  # number of samples in a day
        d = int(num_samples / m)  # number of days

        z = np.concatenate((x, y), axis=1)  # (num_samples, in_len + out_len, num_nodes, num_features)

        temp = [np.concatenate(
            (z[(u * m):((u + 1) * m):(in_len + out_len)].reshape(-1, num_nodes, num_features),
             z[((u + 1) * m - m % (in_len + out_len) + 1):((u + 1) * m), -1]), axis=0)
            for u in range(d)]
        complete_time_series = np.concatenate(temp, axis=0)  # (total_intervals, num_nodes, num_features)

        return complete_time_series

    def time_transform(self, data, start_time, interval, num_samples):
        result = np.zeros_like(data, dtype=int)
        for i in range(num_samples):
            time = data[i]
            dt = [t.astype('datetime64[s]').astype(datetime) for t in time]
            hour, minute = [int(s) for s in start_time.split(':')]
            dt = [t.replace(hour=hour, minute=minute) for t in dt]
            dt = np.array([np.datetime64(t) for t in dt])
            time_ind = ((time - dt) / np.timedelta64(interval, 'm')).astype(np.int64)
            result[i] = time_ind
        return result

    def rest_transform(self, data, restday, num_samples):
        result = np.zeros_like(data, dtype=int)
        for i in range(num_samples):
            time = data[i]
            dt = [t.astype('datetime64[s]').astype(datetime) for t in time]
            dates = [t.strftime('%Y-%m-%d') for t in dt]
            rest_ind = restday.loc[dates].to_numpy().flatten().astype(np.int64)  # 0: workday, 1: restday
            result[i] = rest_ind
        return result

    def weather(self, time, root):
        num_sampel, T = time.shape
        time = time.reshape(-1)
        df = pd.read_excel(osp.join(root, 'weather.xlsx'))
        nearest_time_indices = np.abs(df['upTime'].values[:, np.newaxis] - time).argmin(axis=0)
        nearest_time_data = df.iloc[nearest_time_indices]
        upTime = nearest_time_data['upTime'].to_numpy().reshape(num_sampel, T)
        wtNm = nearest_time_data['wtNm'].to_numpy().reshape(num_sampel, T)
        wtTemp = nearest_time_data['wtTemp'].to_numpy().reshape(num_sampel, T)
        wtHumi = nearest_time_data['wtHumi'].to_numpy().reshape(num_sampel, T)
        wtWinp = nearest_time_data['wtWinp'].to_numpy().reshape(num_sampel, T)
        wtAqi = nearest_time_data['wtAqi'].to_numpy().reshape(num_sampel, T)

        return upTime, wtNm, wtTemp, wtHumi, wtWinp, wtAqi

    def gen_graph_conn(self, root):
        with open(osp.join(root, 'graph_conn.pkl'), 'rb') as f:
            graph_conn = pickle.load(f).astype(np.float32)  # symmetric, with self-loops

        return graph_conn

    def gen_graph_sml(self, complete_time_series, top_k, num_nodes):
        x = complete_time_series.transpose((1, 0, 2)).reshape(num_nodes, -1)
        graph_sml1 = cosine_similarity(x)
        graph_sml2 = compute_graph_sml(x)
        graph_sml = graph_sml1 + graph_sml2
        graph_sml = top_k_and_normalize(graph_sml, top_k)
        return graph_sml

    def gen_graph_sml_dtw(self, root):
        with open(osp.join(root, 'graph_sml.pkl'), 'rb') as f:
            graph_sml_dtw = pickle.load(f).astype(np.float32)  # symmetric, with self-loops

        return graph_sml_dtw

    def gen_graph_cor(self, root):
        with open(osp.join(root, 'graph_cor.pkl'), 'rb') as f:
            graph_cor = pickle.load(f).astype(np.float32)  # asymmetric, graph_cor[i, j] is the weight from j to i

        return graph_cor

    def gen_transition_matrices(self, graphs):
        S_conn = row_normalize(add_self_loop(graphs['graph_conn']))
        S_sml = row_normalize(add_self_loop(graphs['graph_sml']))
        S_cor = row_normalize(add_self_loop(graphs['graph_cor']))
        S = np.stack((S_conn, S_sml, S_cor), axis=0)

        return S

    def get_temporal_characterisrics(self, data, num_samples):
        xtime = data['xtime']
        ytime = data['ytime']
        restday = pd.read_csv(osp.join(self.root, 'restday.csv'), parse_dates=['time'], index_col='time')
        x_time = self.time_transform(xtime, self.start_time, self.interval, num_samples)
        y_time = self.time_transform(ytime, self.start_time, self.interval, num_samples)
        x_rest = self.rest_transform(xtime, restday, num_samples)
        y_rest = self.rest_transform(ytime, restday, num_samples)
        x_upTime, x_wtNm, x_wtTemp, x_wtHumi, x_wtWinp, x_wtAqi = self.weather(xtime, self.root)
        y_upTime, y_wtNm, y_wtTemp, y_wtHumi, y_wtWinp, y_wtAqi = self.weather(ytime, self.root)
        x_temporal_characterisrics = np.stack((x_time, x_rest, x_wtNm, x_wtWinp, x_wtTemp, x_wtHumi, x_wtAqi), axis=1)
        y_temporal_characterisrics = np.stack((y_time, y_rest, y_wtNm, y_wtWinp, y_wtTemp, y_wtHumi, y_wtAqi), axis=1)

        return x_temporal_characterisrics, y_temporal_characterisrics

    def get_spatial_characterisrics(self, num_nodes):
        # max_index = [10, 20]
        # spatial_characterisrics = np.concatenate(
        #     [np.random.randint(max_embedding, size=[num_nodes, 1]) for max_embedding in max_index] + [
        #         np.random.random(size=[num_nodes, 2])], axis=1)
        spatial_characterisrics = None
        return spatial_characterisrics

    def get_predefine_graphs(self, data):
        complete_time_series = self.gen_complete_time_series(data, self.num_intervals, self.in_len, self.out_len,
                                                             self.num_nodes,
                                                             self.num_features)
        graph_conn = self.gen_graph_conn(self.root)  # provided by PVCGN
        graph_sml = self.gen_graph_sml(complete_time_series, self.top_k, self.num_nodes)
        graph_sml_dtw = self.gen_graph_sml_dtw(self.root)  # provided by PVCGN
        graph_cor = self.gen_graph_cor(self.root)  # provided by PVCGN
        graphs = {'graph_conn': graph_conn, 'graph_sml': graph_sml,
                  'graph_sml_dtw': graph_sml_dtw, 'graph_cor': graph_cor}
        eigenmaps = compute_eigenmaps(graph_conn, self.eigenmaps_k)
        transition_matrices = self.gen_transition_matrices(graphs)
        return eigenmaps, transition_matrices

    def process(self, split):
        data, time_flag = self.open_data(self.root, split, self.cfgs['debug'])
        x = data['x']
        y = data['y']
        x_norm = normalize(x, self.mean, self.std)
        y_norm = normalize(y, self.mean, self.std)

        x_temporal_characterisrics, y_temporal_characterisrics = self.get_temporal_characterisrics(data, x.shape[0])

        x_norm, y_norm, y, time_flag, x_temporal_characterisrics, y_temporal_characterisrics = totensor(
            [x_norm, y_norm, y, time_flag, x_temporal_characterisrics, y_temporal_characterisrics])
        data_out = {'input_norm': x_norm,
                    'target_norm': y_norm,
                    'target_unnorm': y,
                    'target_time_flag': time_flag,
                    'input_temporal_characterisrics': x_temporal_characterisrics,
                    'target_temporal_characterisrics': y_temporal_characterisrics
                    }

        if split == 'train':
            eigenmaps, transition_matrices = self.get_predefine_graphs(data)
            spatial_characterisrics = self.get_spatial_characterisrics(self.num_nodes)

            mean, std, eigenmaps, transition_matrices, spatial_characterisrics = totensor(
                [self.mean, self.std, eigenmaps, transition_matrices, spatial_characterisrics])
            nor_base = [mean, std]
            statics = {'nor_base': nor_base,
                       'eigenmaps': eigenmaps,
                       'transition_matrices': transition_matrices,
                       'spatial_characterisrics': spatial_characterisrics}
            data_out['statics'] = statics

        folder_path = f'data/Preprocessing-data-sets/{self.cfgs["name"]}/{self.cfgs["debug"]}/{self.cfgs["in_len"]}-{self.cfgs["out_len"]}'  # 文件夹路径
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(data_out, osp.join(folder_path, f'{split}.pt'))

    def draw_net(self):
        adj = self.gen_graph_conn(self.root)
        for i in range(adj.shape[0]):
            adj[i, i] = 0.

        plt.figure(figsize=[18, 12])
        G = nx.Graph(adj)
        # 绘制无向图
        pos = nx.spring_layout(G, k=0.1, iterations=500)
        # 绘制网络图
        nx.draw(G, pos, with_labels=True, node_color='skyblue')
        # 显示图形
        plt.savefig(self.root + '/net.png')


class Highway():
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.root = cfgs['root']
        self.top_k = cfgs['top_k']
        self.eigenmaps_k = cfgs['eigenmaps_k']
        self.in_len = cfgs['in_len']
        self.out_len = cfgs['out_len']
        self.ratio = cfgs['ratio']

        if self.cfgs['debug']:
            self.data = self.get_flow_data(cfgs['root'])[0:100, :, cfgs['features']]  # [T,N,C]
            self.time_flag = generate_time_and_flags(start_data=cfgs['start_data'], end_data=cfgs['end_data'],
                                                     interval=cfgs['interval'], time_ranges=cfgs['time_ranges'])[0:100]
        else:
            self.data = self.get_flow_data(cfgs['root'])[:, :, cfgs['features']]  # [T,N,C]
            self.time_flag = generate_time_and_flags(start_data=cfgs['start_data'], end_data=cfgs['end_data'],
                                                     interval=cfgs['interval'], time_ranges=cfgs['time_ranges'])

        self.T, self.N, self.C = self.data.shape
        self.mean, self.std = compute_mean_std(self.data)  # [3,1]
        self.nor_base = [self.mean, self.std]
        self.data_nor = normalize(self.data, self.mean, self.std)  # [T,N,C]

    def get_flow_data(self, root):
        flow_file = root + '/data.npz'
        data = np.load(flow_file)
        data = data['data']
        return data

    def time_transform(self, data):
        T, N, C = data.shape
        time = np.arange(0, T)
        time = time % (24 * 12)

        return time

    def rest_transform(self, data, root):
        T, N, C = data.shape
        time = np.arange(0, T)
        time = time // (24 * 12)
        restday = pd.read_excel(osp.join(root, 'restday.xlsx'))['restday'].to_numpy()
        time = restday[time]
        return time

    def get_adjacent_matrix(self, root, N, top_k):
        distance_file = root + '/distance.csv'
        data = np.loadtxt(distance_file, comments='#', delimiter=',', skiprows=1)
        A = np.array(data[:, 0:2], dtype=int)
        distance = data[:, 2]
        std_dis = np.std(distance)
        distance = np.exp(-(distance ** 2) / (std_dis ** 2))
        distance[distance < 0.01] = 0.
        adj = np.eye(N, dtype=int)
        adj_dis = np.zeros([N, N], dtype=float)
        adj[A[:, 0], A[:, 1]] = adj[A[:, 1], A[:, 0]] = 1
        adj_dis[A[:, 0], A[:, 1]] = adj_dis[A[:, 1], A[:, 0]] = distance
        for i in range(N):
            adj_dis[i, i] = 1
        adj_dis = top_k_and_normalize(adj_dis, k=top_k)
        return adj, adj_dis

    def more_graph(self, data, top_k):
        T, N, C = data.shape
        similarities = cosine_similarity(data.reshape(N, -1))
        graph_sml = compute_graph_sml(data.reshape(N, -1))
        graph = similarities + graph_sml
        graph = top_k_and_normalize(matrix=graph, k=top_k)
        graph = graph.reshape(1, N, N)

        return graph

    def data_split(self, data, in_len, out_len, axis=0):
        input_list = []
        target_list = []
        if axis != 0:
            data = torch.transpose(data, 0, axis)
        T = data.shape[0]
        for i in range(T - in_len - out_len):
            start = i
            mid = start + in_len
            end = mid + out_len
            if len(data.shape) == 3:
                input = data[start:mid, :, :]
                target = data[mid:end, :, :]
            elif len(data.shape) == 2:
                input = data[start:mid, :]
                target = data[mid:end, :]
            else:
                input = data[start:mid]
                target = data[mid:end]
            if axis != 0:
                input = torch.transpose(input, 0, axis)
                target = torch.transpose(target, 0, axis)
            input_list.append(input)
            target_list.append(target)
        return input_list, target_list

    def get_temporal_characterisrics(self):
        day = self.time_transform(self.data)
        week = self.rest_transform(self.data, self.root)
        temporal_characterisrics = np.stack((day, week))  # [2,T,1]
        return temporal_characterisrics

    def get_spatial_characterisrics(self, num_nodes):
        # max_index = [10, 20]
        # spatial_characterisrics = np.concatenate(
        #     [np.random.randint(max_embedding, size=[num_nodes, 1]) for max_embedding in max_index] + [
        #         np.random.random(size=[num_nodes, 2])], axis=1)
        spatial_characterisrics = None
        return spatial_characterisrics

    def get_predefine_graphs(self):
        graph_conn, graph_dis = self.get_adjacent_matrix(self.root, self.N, self.top_k)  # [N,N]
        eigenmaps = compute_eigenmaps(graph_conn, self.eigenmaps_k)
        mor_graph = self.more_graph(self.data, self.top_k)  # [1,N,N]
        graph_dis = graph_dis.reshape([1, self.N, self.N])
        transition_matrices = np.concatenate([graph_dis, mor_graph], axis=0)  # [2,N,N]
        return eigenmaps, transition_matrices

    def process(self, split):
        temporal_characterisrics = self.get_temporal_characterisrics()
        data, data_nor, nor_base, temporal_characterisrics, time_flag = totensor(
            [self.data, self.data_nor, self.nor_base, temporal_characterisrics, self.time_flag])

        if split == "train":
            start = 0
            end = self.ratio[0] * 288
        elif split == 'val':
            start = self.ratio[0] * 288
            end = (self.ratio[0] + self.ratio[1]) * 288
        elif split == 'test':
            start = (self.ratio[0] + self.ratio[1]) * 288
            end = (self.ratio[0] + self.ratio[1] + self.ratio[2]) * 288
        else:
            print('split error')

        data, data_nor, temporal_characterisrics, time_flag = data[
                                                              start:end], data_nor[
                                                                          start:end], temporal_characterisrics[:,
                                                                                      start:end], time_flag[
                                                                                                  start:end]

        input_norm, target_norm = self.data_split(data_nor, self.in_len, self.out_len, axis=0)
        _, target_unnorm = self.data_split(data, self.in_len, self.out_len, axis=0)
        input_temporal_characterisrics, target_temporal_characterisrics = self.data_split(temporal_characterisrics,
                                                                                          self.in_len, self.out_len,
                                                                                          axis=1)
        _, target_time_flag = self.data_split(time_flag, self.in_len, self.out_len, axis=0)

        data = {'input_norm': input_norm,
                'target_norm': target_norm,
                'target_unnorm': target_unnorm,
                'target_time_flag': target_time_flag,
                'input_temporal_characterisrics': input_temporal_characterisrics,
                'target_temporal_characterisrics': target_temporal_characterisrics}

        if split == 'train':
            eigenmaps, transition_matrices = self.get_predefine_graphs()
            spatial_characterisrics = self.get_spatial_characterisrics(self.root)
            eigenmaps, transition_matrices, spatial_characterisrics = totensor(
                [eigenmaps, transition_matrices, spatial_characterisrics])
            data['statics'] = {'nor_base': nor_base,
                               'eigenmaps': eigenmaps,
                               'transition_matrices': transition_matrices,
                               'spatial_characterisrics': spatial_characterisrics}

        folder_path = f'data/Preprocessing-data-sets/{self.cfgs["name"]}/{self.cfgs["debug"]}/{self.cfgs["in_len"]}-{self.cfgs["out_len"]}'  # 文件夹路径
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(data, osp.join(folder_path, f'{split}.pt'))
        return data

    def draw_net(self):
        adj, _ = self.get_adjacent_matrix(self.root, self.N, 8)
        for i in range(adj.shape[0]):
            adj[i, i] = 0.

        adj[adj > 0.] = 1.

        plt.figure(figsize=[18, 12])
        G = nx.Graph(adj)
        # 绘制无向图
        pos = nx.spring_layout(G, k=0.15)
        pos = nx.kamada_kawai_layout(G)
        # 绘制网络图
        nx.draw(G, pos, with_labels=True, node_color='skyblue')
        # 显示图形
        plt.savefig(self.root + '/net.png')
        plt.show()

class CityBT():
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.root = cfgs['root']
        self.top_k = cfgs['top_k']
        self.eigenmaps_k = cfgs['eigenmaps_k']
        self.in_len = cfgs['in_len']
        self.out_len = cfgs['out_len']
        self.ratio = cfgs['ratio']

        if self.cfgs['debug']:
            self.data = self.get_flow_data(cfgs['root'])[0:100, :, cfgs['features']]  # [T,N,C]
            self.time_flag = generate_time_and_flags(start_data=cfgs['start_data'], end_data=cfgs['end_data'],
                                                     interval=cfgs['interval'], time_ranges=cfgs['time_ranges'])[0:100]
        else:
            self.data = self.get_flow_data(cfgs['root'])[:, :, cfgs['features']]  # [T,N,C]
            self.time_flag = generate_time_and_flags(start_data=cfgs['start_data'], end_data=cfgs['end_data'],
                                                     interval=cfgs['interval'], time_ranges=cfgs['time_ranges'])
        self.T, self.N, self.C = self.data.shape

        self.mean, self.std = compute_mean_std(self.data)  # [3,1]
        self.nor_base = [self.mean, self.std]
        self.data_nor = normalize(self.data, self.mean, self.std)  # [T,N,C]

    def get_flow_data(self, root):
        flow_file = osp.join(root, 'data.npy')
        data = np.load(flow_file)[:, :, np.newaxis]
        return data

    def time_transform(self, data):
        T, N, C = data.shape
        time = np.arange(0, T)
        time = time % (24 * 12)

        return time

    def rest_transform(self, data, root):
        T, N, C = data.shape
        time = np.arange(0, T)
        time = time // (24 * 12)
        restday = pd.read_excel(osp.join(root, 'restday.xlsx'))['restday'].to_numpy()
        time = restday[time]
        return time

    def get_adjacent_matrix(self, root, N, top_k):
        distance_file = root + '/G_distance.npy'
        adj_dis = np.load(distance_file)
        adj = np.eye(N, dtype=int)
        adj[adj_dis > 0] = 1

        non_zero_elements_index = np.flatnonzero(adj_dis)
        non_zero_elements = adj_dis.flat[non_zero_elements_index]
        std_dis = np.std(non_zero_elements)
        adj_dis = np.exp(-(adj_dis ** 2) / (std_dis ** 2))

        for i in range(N):
            adj_dis[i, i] = 1
        adj_dis = top_k_and_normalize(adj_dis, k=top_k)
        return adj, adj_dis

    def more_graph(self, data, top_k):
        # 相似图
        T, N, C = data.shape
        similarities = cosine_similarity(data.reshape(N, -1))
        g_sml = compute_graph_sml(data.reshape(N, -1))
        g_sml = similarities + g_sml
        g_sml = top_k_and_normalize(matrix=g_sml, k=top_k)

        # 时间图
        time_file = self.root + '/G_duration.npy'
        G_time = np.load(time_file)

        graph = np.stack([g_sml, G_time], axis=0)

        return graph

    def data_split(self, data, in_len, out_len, axis=0):
        input_list = []
        target_list = []
        if axis != 0:
            data = torch.transpose(data, 0, axis)
        T = data.shape[0]
        for i in range(T - in_len - out_len):
            start = i
            mid = start + in_len
            end = mid + out_len
            if len(data.shape) == 3:
                input = data[start:mid, :, :]
                target = data[mid:end, :, :]
            elif len(data.shape) == 2:
                input = data[start:mid, :]
                target = data[mid:end, :]
            else:
                input = data[start:mid]
                target = data[mid:end]
            if axis != 0:
                input = torch.transpose(input, 0, axis)
                target = torch.transpose(target, 0, axis)
            input_list.append(input)
            target_list.append(target)
        return input_list, target_list

    def weather(self, time, root):
        df = pd.read_excel(osp.join(root, 'weather.xlsx'))
        nearest_time_indices = np.abs(df['upTime'].values[:, np.newaxis] - time).argmin(axis=0)
        nearest_time_data = df.iloc[nearest_time_indices]
        upTime = nearest_time_data['upTime'].to_numpy()
        wtNm = nearest_time_data['wtNm'].to_numpy()
        wtTemp = nearest_time_data['wtTemp'].to_numpy()
        wtHumi = nearest_time_data['wtHumi'].to_numpy()
        wtWinp = nearest_time_data['wtWinp'].to_numpy()
        wtAqi = nearest_time_data['wtAqi'].to_numpy()

        return upTime, wtNm, wtTemp, wtHumi, wtWinp, wtAqi

    def get_temporal_characterisrics(self):
        day = self.time_transform(self.data)
        week = self.rest_transform(self.data, self.root)
        time = np.arange('2024-06-01', '2024-07-06', np.timedelta64(5, 'm'), dtype='datetime64[ns]')[
               :self.data.shape[0]]
        upTime, wtNm, wtTemp, wtHumi, wtWinp, wtAqi = self.weather(time, self.root)
        temporal_characterisrics = np.stack((day, week, wtNm, wtWinp, wtTemp, wtHumi, wtAqi))  # [2,T,1]

        return temporal_characterisrics

    def get_spatial_characterisrics(self, root):
        spatial_characterisrics = pd.read_excel(osp.join(root, 'spatial_factors.xlsx'), index_col=0)
        spatial_characterisrics = spatial_characterisrics.to_numpy()
        return spatial_characterisrics

    def get_predefine_graphs(self):
        graph_conn, graph_dis = self.get_adjacent_matrix(self.root, self.N, self.top_k)  # [N,N]
        eigenmaps = compute_eigenmaps(graph_conn, self.eigenmaps_k)
        mor_graph = self.more_graph(self.data, self.top_k)  # [1,N,N]
        graph_dis = graph_dis.reshape([1, self.N, self.N])
        transition_matrices = np.concatenate([graph_dis, mor_graph], axis=0)  # [2,N,N]
        return eigenmaps, transition_matrices

    def process(self, split):
        temporal_characterisrics = self.get_temporal_characterisrics()
        data, data_nor, nor_base, temporal_characterisrics, time_flag = totensor(
            [self.data, self.data_nor, self.nor_base, temporal_characterisrics, self.time_flag])

        if split == "train":
            start = 0
            end = self.ratio[0] * 288
        elif split == 'val':
            start = self.ratio[0] * 288
            end = (self.ratio[0] + self.ratio[1]) * 288
        elif split == 'test':
            start = (self.ratio[0] + self.ratio[1]) * 288
            end = (self.ratio[0] + self.ratio[1] + self.ratio[2]) * 288
        else:
            print('split error')

        data, data_nor, temporal_characterisrics, time_flag = data[
                                                              start:end], data_nor[
                                                                          start:end], temporal_characterisrics[:,
                                                                                      start:end], time_flag[
                                                                                                  start:end]

        input_norm, target_norm = self.data_split(data_nor, self.in_len, self.out_len, axis=0)
        _, target_unnorm = self.data_split(data, self.in_len, self.out_len, axis=0)
        input_temporal_characterisrics, target_temporal_characterisrics = self.data_split(temporal_characterisrics,
                                                                                          self.in_len, self.out_len,
                                                                                          axis=1)
        _, target_time_flag = self.data_split(time_flag, self.in_len, self.out_len, axis=0)

        data = {'input_norm': input_norm,
                'target_norm': target_norm,
                'target_unnorm': target_unnorm,
                'target_time_flag': target_time_flag,
                'input_temporal_characterisrics': input_temporal_characterisrics,
                'target_temporal_characterisrics': target_temporal_characterisrics}

        if split == 'train':
            eigenmaps, transition_matrices = self.get_predefine_graphs()
            spatial_characterisrics = self.get_spatial_characterisrics(self.root)
            eigenmaps, transition_matrices, spatial_characterisrics = totensor(
                [eigenmaps, transition_matrices, spatial_characterisrics])
            data['statics'] = {'nor_base': nor_base,
                               'eigenmaps': eigenmaps,
                               'transition_matrices': transition_matrices,
                               'spatial_characterisrics': spatial_characterisrics}

        folder_path = f'data/Preprocessing-data-sets/{self.cfgs["name"]}/{self.cfgs["debug"]}/{self.cfgs["in_len"]}-{self.cfgs["out_len"]}'  # 文件夹路径
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(data, osp.join(folder_path, f'{split}.pt'))
        return data

    def draw_net(self):
        adj, _ = self.get_adjacent_matrix(self.root, self.N, 8)
        for i in range(adj.shape[0]):
            adj[i, i] = 0.

        plt.figure(figsize=[18, 12])
        G = nx.Graph(adj)
        # 绘制无向图
        pos = nx.spring_layout(G, k=0.15)
        pos = nx.kamada_kawai_layout(G)
        # 绘制网络图
        nx.draw(G, pos, with_labels=True, node_color='skyblue')
        # 显示图形
        plt.show()
        plt.savefig(self.root + '/net.png')


def Update(cfgs, update=False):
    name = cfgs['name']
    # cfgs['root'] = '../data/' + name
    cfgs['root'] = 'data/' + name
    his_cfgs_add = f'data/Preprocessing-data-sets/{cfgs["name"]}/{cfgs["debug"]}/{cfgs["in_len"]}-{cfgs["out_len"]}/his_cfgs'
    if name == 'HZMetro' or name == 'SHMetro':
        data_class = Metro(cfgs)
    elif name == 'WHBT':
        data_class = CityBT(cfgs)
    else:
        data_class = Highway(cfgs)
    cfgs.pop('which_transition_matrices', None)
    cfgs.pop('temporal_num_embeddings', None)
    cfgs.pop('spatial_num_embeddings', None)
    try:
        his_cfgs = torch.load(his_cfgs_add)
        if cfgs != his_cfgs or update:
            data_class.process('train')
            data_class.process('val')
            data_class.process('test')
            torch.save(cfgs, his_cfgs_add)
    except:
        data_class.process('train')
        data_class.process('val')
        data_class.process('test')
        torch.save(cfgs, his_cfgs_add)


if __name__ == '__main__':
    name = 'PEMS08'
    cfgs = yaml.safe_load(open(f'../cfgs/datasets\{name}.yaml'))['dataset']
    cfgs['root'] = fr'D:\MGT\data\PEMS08'
    net = Highway(cfgs)
    net.draw_net()
    # data = net.process('train')
    # net.get_spatial_characterisrics(net.root)
    # net.process('train')
    # net.draw_net()
