# 加载数据
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from joblib import dump, load
import torch.utils.data as Data
import torch
import torch.nn as nn
import param_manage as pm
# 参数与配置
torch.manual_seed(100)  # 设置随机种子，以使实验结果具有可重复性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
def dataloader(batch_size, tag, workers=0):
    # 训练集
    train_set = load(pm.train_set_RTPV )
    train_label = load(pm.train_label_RTPV)
    # 测试集
    test_set = load(pm.test_set_RTPV)
    test_label = load(pm.test_label_RTPV)

    # 加载数据
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_set, train_label),
                                   batch_size=batch_size, num_workers=workers, drop_last=True)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_set, test_label),
                                  batch_size=batch_size, num_workers=workers, drop_last=True)
    return train_loader, test_loader


from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
class TransformerBiLSTM(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_layer_sizes,hidden_dim, num_layers, num_heads, output_dim, dropout_rate=0.5):
        """
        params:
        batch_size         : 批次量大小
        input_dim          : 输入数据的维度
        hidden_layer_sizes : bilstm 隐藏层的数目和维度
        hidden_dim          : 注意力维度
        num_layers          : Transformer编码器层数
        num_heads           : 多头注意力头数
        output_dim         : 输出维度
        dropout_rate        : 随机丢弃神经元的概率
        """
        super().__init__()
        # 参数
        self.batch_size = batch_size

        # Transformer编码器  Transformer layers
        self.hidden_dim = hidden_dim
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout=dropout_rate, batch_first=True),
            num_layers
        )
        # self.avgpool = nn.AdaptiveAvgPool1d(9)

        # BiLSTM参数
        self.num_layers = len(hidden_layer_sizes)  # bilstm层数
        self.bilstm_layers = nn.ModuleList()  # 用于保存BiLSTM层的列表
        # 定义第一层BiLSTM
        self.bilstm_layers.append(nn.LSTM(input_dim, hidden_layer_sizes[0], batch_first=True, bidirectional=True))
        # 定义后续的BiLSTM层
        for i in range(1, self.num_layers):
                self.bilstm_layers.append(nn.LSTM(hidden_layer_sizes[i-1]* 2, hidden_layer_sizes[i], batch_first=True, bidirectional=True))

        # 定义线性层
        self.linear  = nn.Linear(hidden_layer_sizes[-1] * 2, output_dim)


    def forward(self, input_seq):
        # Transformer 处理
        # 在PyTorch中，transformer模型的性能与batch_first参数的设置相关。
        # 当batch_first为True时，输入的形状应为(batch, sequence, feature)，这种设置在某些情况下可以提高推理性能。
        transformer_output = self.transformer(input_seq)  #  torch.Size([256, 7, 8])

        # 送入 BiLSTM 层
        #改变输入形状，bilstm 适应网络输入[batch, seq_length, H_in]
        bilstm_out = transformer_output
        for bilstm in self.bilstm_layers:
            bilstm_out, _= bilstm(bilstm_out)  ## 进行一次BiLSTM层的前向传播  # torch.Size([256, 7, 8])
        predict = self.linear(bilstm_out[:, -1, :]) # torch.Size([256, 1]  # 仅使用最后一个时间步的输出
        return predict


# 看下这个网络结构总共有多少个参数
def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')

# 训练模型
import time
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc("font", family='Microsoft YaHei')


def model_train(batch_size, model, epochs, loss_function, optimizer, train_loader, test_loader,tag,files_dir):
    model = model.to(device)
    # 样本长度
    train_size = len(train_loader) * batch_size
    test_size = len(test_loader) * batch_size

    # 最低MSE
    minimum_mse = 1000.
    # 最佳模型
    best_model = model

    train_mse = []  # 记录在训练集上每个epoch的 MSE 指标的变化情况   平均值
    test_mse = []  # 记录在测试集上每个epoch的 MSE 指标的变化情况   平均值

    # 计算模型运行时间
    start_time = time.time()
    for epoch in range(epochs):
        # 训练
        model.train()

        train_mse_loss = 0.  # 保存当前epoch的MSE loss和
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            # 每次更新参数前都梯度归零和初始化
            optimizer.zero_grad()
            # 前向传播
            y_pred = model(seq)  # torch.Size([16, 10])
            # 损失计算
            loss = loss_function(y_pred, labels)
            train_mse_loss += loss.item()  # 计算 MSE 损失
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()
        #     break
        # break
        # 计算总损失
        train_av_mseloss = train_mse_loss / train_size
        train_mse.append(train_av_mseloss)

        print(f'Epoch: {epoch + 1:2} train_MSE-Loss: {train_av_mseloss:10.8f}')
        # 每一个epoch结束后，在验证集上验证实验结果。
        with torch.no_grad():
            test_mse_loss = 0.  # 保存当前epoch的MSE loss和
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                pre = model(data)
                # 计算损失
                test_loss = loss_function(pre, label)
                test_mse_loss += test_loss.item()

            # 计算总损失
            test_av_mseloss = test_mse_loss / test_size
            test_mse.append(test_av_mseloss)
            print(f'Epoch: {epoch + 1:2} test_MSE_Loss:{test_av_mseloss:10.8f}')
            # 如果当前模型的 MSE 低于于之前的最佳准确率，则更新最佳模型
            # 保存当前最优模型参数
            if test_av_mseloss < minimum_mse:
                minimum_mse = test_av_mseloss
                best_model = model  # 更新最佳模型的参数

    # 保存最后的参数
    # torch.save(model, 'final_model_transformer_bilstm.pt')
    # 保存最好的参数
    torch.save(best_model, files_dir/f'transformer_bilstm_{tag}.pt')
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')

    # 可视化
    plt.plot(range(epochs), train_mse, color='b', label='train_MSE-loss')
    plt.plot(range(epochs), test_mse, color='y', label='test_MSE-loss')
    plt.legend()
    plt.show()  # 显示 lable
    print(f'min_MSE: {minimum_mse}')

def train_main(input_dim,num_heads,output_dim,tag,files_dir):
    batch_size = 128
    # 加载数据
    train_loader, test_loader = dataloader(batch_size,tag)
    # 定义模型参数
    batch_size = 128
    # input_dim = 6  # 输入维度
    hidden_layer_sizes = [64, 64]  # BiLSTM 层 结构
    hidden_dim = 128  # 注意力维度
    num_layers = 3  # 编码器层数
    # num_heads = 3  # 多头注意力头数 注意多头注意力的整除，选的时候注意
    # output_dim = 1  # 输出维度为 1

    model = TransformerBiLSTM(batch_size, input_dim, hidden_layer_sizes, hidden_dim, num_layers, num_heads, output_dim)

    # 定义损失函数和优化函数
    loss_function = nn.MSELoss(reduction='sum')  # loss
    learn_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)  # 优化器


    count_parameters(model)

    #  模型训练
    batch_size = 64
    epochs = 100
    model_train(batch_size, model, epochs, loss_function, optimizer, train_loader, test_loader,tag,files_dir)
    # 模型预测
    # 模型 测试集 验证
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(files_dir/f'transformer_bilstm_{tag}.pt')
    model = model.to(device)

    # 预测数据
    original_data = []
    pre_data = []
    with torch.no_grad():
            for data, label in test_loader:
                model.eval()  # 将模型设置为评估模式
                origin_lable = label.tolist()
                original_data += origin_lable
                # 预测
                data, label = data.to(device), label.to(device)
                test_pred = model(data)  # 对测试集进行预测
                # 使用 .squeeze() 将其转换为一维张量
                test_pred = test_pred.tolist()
                pre_data += test_pred
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    # 反归一化处理
    # 使用相同的均值和标准差对预测结果进行反归一化处理
    # 反标准化
    scaler  = load(files_dir/f'scaler_{tag}')
    original_data = scaler.inverse_transform(original_data)
    pre_data = scaler.inverse_transform(pre_data)
    # 可视化结果
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(original_data, label='真实值',color='red')  # 真实值
    plt.plot(pre_data, label='Transformer-BiLSTM预测值',color='blue')  # 预测值
    plt.legend()
    plt.show()
    # # 将真实值和预测值合并为一个 DataFrame
    # result_df = pd.DataFrame({'真实值': original_data.flatten(), '预测值': pre_data.flatten()})
    # # 保存 DataFrame 到一个 CSV 文件
    # result_df.to_csv('真实值与预测值.csv', index=False, encoding='utf-8')
    # 打印保存成功的消息
    # print('真实值和预测值已保存到真实值与预测值.csv文件中。')

    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # 模型分数
    original_data = np.array(original_data)
    pre_data = np.array(pre_data)
    score = r2_score(original_data, pre_data)
    print('*'*50)
    print('Transformer-BiLSTM 模型分数--R^2:', score)

    print('*'*50)
    # 测试集上的预测误差
    test_mse = mean_squared_error(original_data, pre_data)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(original_data, pre_data)
    print('测试数据集上的均方误差--MSE: ',test_mse)
    print('测试数据集上的均方根误差--RMSE: ',test_rmse)
    print('测试数据集上的平均绝对误差--MAE: ',test_mae)