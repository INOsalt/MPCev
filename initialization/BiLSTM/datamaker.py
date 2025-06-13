import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')


# 制作数据集和标签
import torch
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, MinMaxScaler




# 这些转换是为了将数据和标签转换为PyTorch可以处理的张量
def make_data_labels(x_data, y_label):
    '''
        返回 x_data: 数据集     torch.tensor
            y_label: 对应标签值  torch.tensor
    '''
    # 信号值
    x_data = torch.tensor(x_data).float()
    # 标签值
    y_label = torch.tensor(y_label).float()
    return x_data, y_label


# 使用滑动窗口处理时间序列数据
def data_window_maker(x_var, ylable_data, window_size):
    '''
        参数:
        x_var      : 输入 变量数据
        ylable_data: 对应y数据
        window_size: 滑动窗口大小

        返回:
        data_x: 特征数据
        data_y: 标签数据
    '''
    # 用来存放输入特征
    data_x = []
    # 用来存放输出标签
    data_y = []
    # 构建训练集和对应标签
    data_len = x_var.shape[0]  # 序列长度
    for i in range(data_len - window_size):
        data_x.append(x_var[i:i + window_size, :])  # 取前window_size个数据作为输入特征
        data_y.append(ylable_data[i + window_size])  # 取第window_size+1个数据作为输出标签
    # 将列表转换为单一的NumPy数组
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    # 转换为 tensor
    data_x, data_y = make_data_labels(data_x, data_y)
    return data_x, data_y


# 数据集制作
def make_dataset(var_data, ylable_data, window_size, split_rate=[0.8, 0.2]):
    '''
        参数:
        var_data   : 输入训练集  变量
        ylable_data : 输入y值标签  变量
        window_size: 滑动窗口大小
        split_rate : 训练集、测试集划分比例

        返回:
        train_set: 训练集数据
        train_label: 训练集标签
        test_set: 测试集数据
        test_label: 测试集标签
    '''
    # 第一步，划分数据集
    # 序列数组
    sample_len = var_data.shape[0]  # 样本总长度
    train_len = int(sample_len * split_rate[0])  # 向下取整
    # 变量数据 划分训练集、测试集
    train_var = var_data[:train_len, :]  # 训练集
    test_var = var_data[train_len:, :]  # 测试集
    # y标签 划分训练集、测试集
    train_y = ylable_data[:train_len]  # 训练集
    test_y = ylable_data[train_len:]  # 测试集

    # 第二步，制作数据集标签  滑动窗口
    train_set, train_label = data_window_maker(train_var, train_y, window_size)
    test_set, test_label = data_window_maker(test_var, test_y, window_size)

    return train_set, train_label, test_set, test_label

if __name__ == "__main__":
    # 读取数据
    original_data = pd.read_csv('feng2019.csv')
    print(original_data.shape)
    original_data.head()
    # 取风速数据
    winddata = original_data['Target'].tolist()
    winddata = np.array(winddata)  # 转换为numpy
    # 可视化
    plt.figure(figsize=(15, 5), dpi=100)
    plt.grid(True)
    plt.plot(winddata, color='red')
    plt.show()

    # 选取 数值 型 变量
    original_data = original_data[
        ['A', 'B', 'C', 'E',
         'F', 'Target']]
    # 1. 输入训练集  变量：
    var_data = original_data
    # 2. 对应y值标签为：
    ylable_data = original_data[['Target']]

    # 归一化处理
    # 使用标准化（z-score标准化）
    scaler = StandardScaler()
    var_data = scaler.fit_transform(var_data)
    ylable_data = scaler.fit_transform(ylable_data)
    # 保存 归一化 模型
    dump(scaler, 'scaler')

    # 定义滑动窗口大小
    window_size = 4
    # 制作数据集
    train_set, train_label, test_set, test_label = make_dataset(var_data, ylable_data, window_size)
    # 保存数据
    dump(train_set, 'train_set')
    dump(train_label, 'train_label')
    dump(test_set, 'test_set')
    dump(test_label, 'test_label')
    print('数据 形状：')
    print(train_set.size(), train_label.size())
    print(test_set.size(), test_label.size())

    # 解释：8个输入维度 去 滑动预测 一个风速
    # 变量 序列长度为 7 , 预测一个 值 （单步预测）

