# -*- coding:utf-8 -*-
# __author__ = 'Taited'

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import matplotlib.pyplot as plt


def gen_train(_is_noise=False, _step=0.1):
    _x = np.arange(-5, 5, _step)
    _y = np.sin(_x)
    if _is_noise:
        _y = gen_noise(_y)
    _x = torch.tensor(_x, dtype=torch.float32, requires_grad=True)
    _y = torch.tensor(_y, dtype=torch.float32, requires_grad=True)
    return _x, _y


def gen_noise(_y):
    if _y is None:
        print('shape is None')
        return None
    _begin = int(_y.shape[0] * 2 / 3)
    _noise = np.random.normal(0, 0.5, size=_y[_begin:-1].shape)
    _y[_begin:-1] = _y[_begin:-1] + _noise
    # _y[_begin:_y.shape[0]-1] = _y[_begin:_y.shape[0]-1] + _noise
    return _y


def gen_test(_step=0.4):
    _x = np.arange(-10, 10, _step)
    _y = np.sin(_x)
    _x = torch.tensor(_x, dtype=torch.float32)
    _y = torch.tensor(_y, dtype=torch.float32)
    return _x, _y


def plot_loss(_loss_list, _is_save=False):
    _x = np.arange(len(_loss_list))
    _y = np.array(_loss_list)
    plt.figure()
    plt.plot(_x, _y, label='loss')
    plt.grid()
    plt.legend()
    plt.show()
    if _is_save:
        plt.savefig('loss_pic.png')


def plot_uncertainty(_x_train, _y_train, _x_pre, _y_mean, _y_std, _y_var, _x_test, _y_test, _t):
    # Aleatoric uncertainty
    # _t: 训练次数
    plt.figure()
    plt.subplot(2, 1, 1)
    y_1 = (_y_mean - _y_std).reshape(-1)
    y_2 = (_y_mean + _y_std).reshape(-1)
    plt.scatter(_x_test, _y_test, color='gold')
    plt.scatter(_x_train, _y_train)
    plt.fill_between(_x_pre, y_1, y_2, color='peachpuff', alpha=0.5,
                     interpolate=True, label='aleatoric uncertainty')
    plt.plot(_x_pre, _y_mean, color='red', label='prediction on test data')
    plt.title('Aleatoric Uncertainty Training with Step of {}'.format(_t))
    plt.legend()
    plt.grid()

    # Epistemic uncertainty
    plt.subplot(2, 1, 2)
    y_1 = (_y_mean - _y_var).reshape(-1)
    y_2 = (_y_mean + _y_var).reshape(-1)
    plt.scatter(_x_test, _y_test, color='gold')
    plt.scatter(_x_train, _y_train)
    plt.fill_between(_x_pre, y_1, y_2, color='peachpuff', alpha=0.5,
                     interpolate=True, label='epistemic uncertainty')
    plt.plot(_x_pre, _y_mean, color='red', label='prediction on test data')
    plt.title('Epistemic Uncertainty Training with Step of {}'.format(_t))
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('{} Step.png'.format(_t))


class FC(nn.Module):
    def __init__(self, in_dim=1, n_hidden_1=150, n_hidden_2=80,
                 n_hidden_3=20, out_dim=1):
        super(FC, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1, bias=True)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2, bias=True)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3, bias=True)
        self.layer4 = nn.Linear(n_hidden_3, out_dim)

    def forward(self, _x, dp=0.1):
        _x = _x.view(_x.size(0), -1)  # 不加上这句使其变成一维向量，会有维度不匹配的错误
        _x = self.layer1(_x)
        _x = F.dropout(_x, p=dp)
        _x = F.relu(_x)
        _x = self.layer2(_x)
        _x = F.dropout(_x, p=dp)
        _x = F.relu(_x)
        _x = self.layer3(_x)
        _x = F.dropout(_x, p=dp)
        _x = F.relu(_x)
        _x = self.layer4(_x)
        return _x


class NN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, p=0.9):
        super(NN, self).__init__()
        self.p = p
        self.fc1 = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.Dropout(p=1 - p),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.Dropout(p=1 - p),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.Dropout(p=1 - p),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.Dropout(p=1 - p),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.Dropout(p=1 - p),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.Dropout(p=1 - p),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.Dropout(p=1 - p),
                                 nn.ReLU())

        self.out = nn.Linear(hidden_size, output_size)
        self.var_out = nn.Sequential(nn.Linear(hidden_size, 1))

    def forward(self, _x):
        features = self.fc1(_x)
        _output = self.out(features)
        _log_var = self.var_out(features)
        return _output, _log_var


class AleatoricLoss(nn.Module):
    def __init__(self):
        super(AleatoricLoss, self).__init__()

    def forward(self, _output, _gt, _log_variance):
        _loss = torch.sum(0.5 * (torch.exp(-1 * _log_variance)) * (_gt - _output) ** 2 + 0.5 * _log_variance)
        return _loss


def aleatoric_loss_func(_prediction, _label):
    # _prediction is a n*m matrix, 数据长度为m，总共进行了n次dropout
    # _mean作为预测值，_std作为偶然不确定性
    _mean, _std = cal_mean_std_in_column(_prediction)
    _loss_tensor = torch.zeros(_label.shape)
    for _i in range(len(_label)):
        _item_1 = _mean[_i] - _label[_i]
        # 避免出现除以0，而得到Nan
        if _std[_i] <= 0:
            _item_2 = _item_1 * _item_1 / (2 * 1e-4)
        else:
            _item_2 = _item_1 * _item_1 / (2 * _std[_i])
        _loss_tensor[_i] = _item_2 + 0.5 * torch.log(_std[_i])
    _loss = _loss_tensor.mean()
    return _loss


# 对一个二维tensor按列计算均值和方差
def cal_mean_std_in_column(_tensor):
    _mean = torch.zeros([_tensor.shape[1]])
    _std = torch.zeros([_tensor.shape[1]])
    for _i in range(_tensor.shape[1]):
        _mean[_i] = _tensor[:, _i].mean().reshape(1)
        _std[_i] = _tensor[:, _i].std().reshape(1)
    return _mean, _std


# 计算认知不确定性
def cal_epistemic(_tensor):
    # _mean为预测值, _std为偶然不确定性, _var为认知不确定性
    _mean, _std = cal_mean_std_in_column(_tensor)
    _var = torch.zeros(_mean.shape[0])
    for _i in range(_mean.shape[0]):
        _item1 = _std[_i] - _mean[_i] * _mean[_i]
        _item2 = torch.zeros(1)
        for _t in range(_tensor.shape[0]):
            _item2 += _tensor[_t, _i] * _tensor[_t, _i]
        _item2 = _item2 / _tensor.shape[0]
        _var[_i] = _item1 + _item2
    return torch.sqrt(_var), _mean, _std


# def train_origin():
#
#
#     while loss >= -0.8:
#
#         predict = torch.zeros([BATCHES, x_train.shape[0]])
#         # 进行batches次dropout
#         for batch in range(BATCHES):
#             prediction = net(x_train)
#             predict[batch, :] = prediction.reshape([1, prediction.shape[0]])
#         loss = aleatoric_loss_func(predict, y_train)
#         if torch.isnan(loss) or loss <= -0.8:  # -0.8是为了早停
#             break
#         loss_list.append(loss)
#         if (t % 50) == 0 or t == 0:
#             print('loss in iter {}: {}'.format(t, loss))
#         optimizer.zero_grad()  # 清空上一步的残余更新参数值
#         t += 1
#     plot_loss(loss_list, _is_save=True)
#
#     # 计算Aleatoric Uncertainty
#     x_test = torch.tensor(x_test, dtype=torch.float32)
#     predict_tensor = torch.zeros([BATCHES, x_test.shape[0]])
#     for batch in range(int(BATCHES)):
#         prediction = net(x_test)
#         predict_tensor[batch, :] = prediction.reshape([1, prediction.shape[0]])
#     var, mean, std = cal_epistemic(predict_tensor)


def MC_samples(_net, _x_data):
    T = 64
    weight_decay = 1e-4
    output_list = []
    aleatoric_uncertainty_list = []
    _l = 10
    for _id in range(len(_x_data)):
        _x = _x_data[_id].reshape(1)
        output_list.append([])
        aleatoric_uncertainty_list.append([])
        for t in range(T):
            _output, _log_var = net(_x)
            _output = _output.item()
            _log_var = _log_var.item()
            output_list[_id].append(_output)
            aleatoric_uncertainty_list[_id].append(_log_var)
    _output = np.array(output_list)
    _aleatoric_uncertainty = np.array(aleatoric_uncertainty_list)
    means = _output.mean(axis=1)
    aleatoric_uncertainty_means = _aleatoric_uncertainty.mean(axis=1)
    variances = np.var(_output, axis=1)
    tau = _l ** 2 * net.p / (2 * len(_x_data) * weight_decay)
    variances += tau**-1
    return means, aleatoric_uncertainty_means, variances


if __name__ == '__main__':
    EPOCH = int(1e3)  # int
    step = 0.3
    # 生成训练数据
    x_train, y_train = gen_train(_is_noise=True, _step=step)
    x_test, y_test = gen_test(_step=step)
    # x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True)
    # y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True)

    # 网络初始化
    net = NN()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001, weight_decay=1e-4)
    loss_func = AleatoricLoss()
    loss_list = []
    # train
    net.train()  # 设置成训练模式，打开dropout
    for _ in range(EPOCH):
        loss_in_turn = torch.zeros(1)
        for id in range(len(x_train)):
            x = x_train[id].reshape(1)
            y = y_train[id].reshape(1)
            net.zero_grad()
            output, log_var = net(x)
            loss = loss_func(output, y, log_var)
            loss.backward()  # 将误差返回给模型
            optimizer.step()  # 建模型的数据更新
            loss_in_turn += loss
        loss_in_turn /= len(x_train)
        loss_list.append(loss_in_turn.detach().numpy())
        if (_ % 50) == 0 or _ == 0 or _ == EPOCH-1:
            print('Loss in iter {}: {}'.format(_, loss_in_turn.detach().numpy()))
    plot_loss(loss_list)
    mean, aleatoric, variances = MC_samples(net, x_test)
    # 画图
    plot_uncertainty(x_train.detach().numpy(), y_train.detach().numpy(),
                     x_test.detach().numpy(), mean, aleatoric, variances,
                     x_test.detach().numpy(), y_test.detach().numpy(), step)
