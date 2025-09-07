import numpy as np
import torch
import csv
def get_adjacency_matrix1(distance_df_filename):
    AC_martix = np.load(distance_df_filename)
    #print(AC_martix)
    num_of_vertices=AC_martix.shape[0]
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i in range(num_of_vertices):
        for j in range(num_of_vertices):
            if AC_martix[i][j]>0:
                A[i][j]=1

    return A
def get_dtw_adj(dtw_path):
    dtw_adj=np.load(dtw_path)

    return dtw_adj

def get_adj_matrix_csv(adj_matrix_node,adj_path):
    adj_matrix = np.zeros([int(adj_matrix_node),int(adj_matrix_node)])
    data = []
    with open(adj_path) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        # header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到data中
            data.append(row)  # 选择某一列加入到data数组中

    node1 = []
    node2 = []
    value = []
    for i in range(1, len(data)):
        node1.append(data[i][0])
        node2.append(data[i][1])
        value.append(data[i][2])

    for j in range(len(node1)):
        adj_matrix[int(node1[j])-1][int(node2[j])-1]=1

    return adj_matrix

def process_graph(graph_data):  # 这个就是在原始的邻接矩阵之上，再次变换，也就是\hat A = D_{-1/2}*A*D_{-1/2}
    N = graph_data.shape[0]  # 获得节点的个数
    matrix_i = torch.eye(N, dtype=torch.float, device=graph_data.device)  # 定义[N, N]的单位矩阵
    graph_data += matrix_i  # [N, N]  ,就是 A+I

    degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)  # [N],计算度矩阵，塌陷成向量，其实就是将上面的A+I每行相加
    degree_matrix = degree_matrix.pow(-1)  # 计算度矩阵的逆，若为0，-1次方可能计算结果为无穷大的数
    degree_matrix[degree_matrix == float("inf")] = 0.  # 让无穷大的数为0

    degree_matrix = torch.diag(degree_matrix)  # 转换成对角矩阵

    return torch.mm(degree_matrix, graph_data)  # 返回 \hat A=D^(-1) * A ,这个等价于\hat A = D_{-1/2}*A*D_{-1/2}

def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

def MSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mse = np.square(y_pred - y_true)
        mse = np.nan_to_num(mse * mask)
        mse = np.mean(mse)
        return mse


def RMSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def MAE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae


def MAPE(y_true, y_pred, null_val=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def RMSE_MAE_MAPE(y_true, y_pred):
    return (
        RMSE(y_true, y_pred),
        MAE(y_true, y_pred),
        MAPE(y_true, y_pred),
    )


def MSE_RMSE_MAE_MAPE(y_true, y_pred):
    return (
        MSE(y_true, y_pred),
        RMSE(y_true, y_pred),
        MAE(y_true, y_pred),
        MAPE(y_true, y_pred),
    )
