# -*- coding:utf-8 -*-
# __author__ = 'Taited'

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import matplotlib.pyplot as plt


def gen_train(_is_noise=False):
    _x = np.arange(-5, 5, 0.3)
    _y = np.sin(_x)
    if _is_noise:
        _y = gen_noise(_y)
    return _x, _y


def gen_noise(_y):
    if _y is None:
        print('shape is None')
        return None
    _begin = int(_y.shape[0] * 2/3)
    _noise = np.random.normal(0, 0.5, size=_y[_begin:-1].shape)
    _y[_begin:-1] = _y[_begin:-1] + _noise
    # _y[_begin:_y.shape[0]-1] = _y[_begin:_y.shape[0]-1] + _noise
    return _y


def gen_test():
    _x = np.arange(-10, 10, 0.2)
    _y = np.sin(_x)
    return _x, _y


def my_plot(_x_train, _y_train, _x_pre, _y_pre, _x_test, _y_test):
    plt.figure()
    plt.scatter(_x_train, _y_train, label='train data')
    plt.plot(_x_pre, _y_pre, label='prediction', color='red')
    plt.plot(_x_test, _y_test, color='coral', label='true test')
    plt.legend()
    plt.grid()
    plt.show()


def plot_loss(_loss_list, _is_save=False):
    x = np.arange(len(_loss_list))
    y = np.array(_loss_list)
    plt.figure()
    plt.plot(x, y, label='loss')
    plt.grid()
    plt.legend()
    plt.show()
    if _is_save:
        plt.savefig('loss_pic.png')


class FC(nn.Module):
    def __init__(self, in_dim=1, n_hidden_1=150, n_hidden_2=80,
                 n_hidden_3=20, out_dim=2):
        super(FC, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1, bias=True)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2, bias=True)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3, bias=True)
        self.layer4 = nn.Linear(n_hidden_3, out_dim, bias=True)

    def forward(self, x, dp=0.5):
        x = x.view(x.size(0), -1)  # 不加上这句使其变成一维向量，会有维度不匹配的错误
        x = self.layer1(x)
        x = F.dropout(x, p=dp)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.dropout(x, p=dp)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.dropout(x, p=dp)
        x = F.relu(x)
        x = self.layer4(x)
        return x


def tensor_list_to_np(_prediction_list):
    _i_row = len(_prediction_list)
    _i_column = len(_prediction_list[0])
    _result = np.zeros([_i_row, _i_column])
    for _i in range(_i_row):
        for _j in range(_i_column):
            _result[_i][_j] = _prediction_list[_i][_j]
    return _result


def cal_epistemic(_np):
    _row = _np.shape[0]
    _col = _np.shape[1]
    _mean = np.zeros([_col, 1])
    _std = np.zeros([_col, 1])
    for _i in range(_col):
        _mean[_i] = _np[:, _i].mean()
        _std[_i] = _np[:, _i].std()
    return _mean, _std


def plot_cal_epistemic(_x_train, _y_train, _x_pre, _y_mean, _y_std, _x_test, _y_test):
    plt.figure()
    y_1 = (_y_mean - _y_std).reshape(-1)
    y_2 = (_y_mean + _y_std).reshape(-1)
    plt.fill_between(_x_pre, y_1, y_2, color='peachpuff', interpolate=True, label='epistemic uncertainty')
    plt.scatter(_x_test, _y_test, color='gold', label='test data')
    plt.scatter(_x_train, _y_train, label='train data')
    plt.plot(_x_pre, _y_mean, color='red', label='prediction on test data')
    plt.title('Epistemic Uncertainty')
    plt.legend()
    plt.grid()
    plt.show()


def aleatoric_loss_func(_prediction, _label):
    _loss_list = torch.zeros(_label.shape)
    for _i in range(len(_label)):
        _sigma_2 = _prediction[_i, 1] * _prediction[_i, 1]
        _item_1 = _prediction[_i, 0] - _label[_i]
        # 避免出现除以0，而得到Nan
        _item_2 = torch.exp(torch.log(_sigma_2))
        _item_3 = _item_1 * _item_1 / (2 * _item_2)
        _loss_list[_i] = _item_3 + 0.5 * torch.log(_sigma_2)
    _loss = _loss_list.mean()
    return _loss


if __name__ == '__main__':
    x_train, y_train = gen_train(_is_noise=True)
    x_test, y_test = gen_test()
    # my_plot(x_train, y_train, x_test, y_test)
    x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True)
    y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True)
    y_train = y_train.view(y_train.size(0), -1)
    net = FC()
    optimizer = optim.Adam(params=net.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_func = torch.nn.MSELoss()
    loss_list = []
    # train
    net.train()  # 设置成训练模式，打开dropout
    for t in range(int(1e3)):
        if t:
            loss.backward()  # 将误差返回给模型
            optimizer.step()  # 建模型的数据更新
        prediction = net(x_train)
        # loss = loss_func(prediction, y_train)
        loss = aleatoric_loss_func(prediction, y_train)
        if loss <= 0.04:  # 早停
            break
        loss_list.append(loss)
        if (t % 100) == 0 or t == 0:
            print('loss in iter {}: {}'.format(t, loss))
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
    plot_loss(loss_list, _is_save=True)

    # 计算认知不确定性
    net.train()
    T = 1e3
    prediction_list = []
    x_test = torch.tensor(x_test, dtype=torch.float32)
    for t in range(int(T)):
        prediction = net(x_test)
        prediction_list.append(prediction)
    prediction_np = tensor_list_to_np(prediction_list)
    mean, std = cal_epistemic(prediction_np)

    # 画图
    plot_cal_epistemic(x_train.detach().numpy(), y_train.detach().numpy(),
            x_test.detach().numpy(), mean, std,
            x_test.detach().numpy(), y_test)
    # my_plot(x_train.detach().numpy(), y_train.detach().numpy(),
    #         x_test.detach().numpy(), y_prediction.detach().numpy(),
    #         x_test.detach().numpy(), y_test)
