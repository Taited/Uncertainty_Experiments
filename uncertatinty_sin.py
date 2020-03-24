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
    _noise = np.random.normal(0, 0.2, size=_y[_begin:-1].shape)
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
                 n_hidden_3=20, out_dim=1):
        super(FC, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1, bias=True)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2, bias=True)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3, bias=True)
        self.layer4 = nn.Linear(n_hidden_3, out_dim)

    def forward(self, x, dp=0.1):
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
    # _prediction is a n*m matrix
    _mean, _std = cal_mean_std_in_column(_prediction)
    _loss_tensor = torch.zeros(_label.shape)
    for _i in range(len(_label)):
        _item_1 = _mean[_i] - _label[_i]
        # 避免出现除以0，而得到Nan
        _item_2 = torch.exp(-1 * torch.log(_std[_i]))
        if _item_2 == 0:
            stoper = 1
        _item_3 = _item_1 * _item_1 * _item_2 * 0.5
        _loss_tensor[_i] = _item_3 + 0.5 * torch.log(_std[_i])
    _loss = _loss_tensor.mean()
    return _loss


def cal_mean_std_in_column(_tensor):
    _mean = torch.zeros([_tensor.shape[1]])
    _std = torch.zeros([_tensor.shape[1]])
    for _i in range(_tensor.shape[1]):
        _mean[_i] = _tensor[:, _i].mean().reshape(1)
        _std[_i] = _tensor[:, _i].std().reshape(1)
    return _mean, _std


if __name__ == '__main__':
    BATCHES = 100
    x_train, y_train = gen_train(_is_noise=True)
    x_test, y_test = gen_test()
    x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True)
    y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True)
    y_train = y_train.view(y_train.size(0), -1)
    net = FC()
    optimizer = optim.Adam(params=net.parameters(), lr=0.0001, weight_decay=1e-4)
    loss_func = torch.nn.MSELoss()
    loss_list = []
    # train
    net.train()  # 设置成训练模式，打开dropout
    for t in range(10001):
        if t:
            loss.backward()  # 将误差返回给模型
            optimizer.step()  # 建模型的数据更新
        predict = torch.zeros([BATCHES, x_train.shape[0]])
        for batch in range(100):
            prediction = net(x_train)
            predict[batch, :] = prediction.reshape([1, prediction.shape[0]])
        loss = aleatoric_loss_func(predict, y_train)
        loss_list.append(loss)
        if (t % 100) == 0 or t == 0:
            print('loss in iter {}: {}'.format(t, loss))
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
    plot_loss(loss_list, _is_save=True)

    # 计算Aleatoric Uncertainty
    x_test = torch.tensor(x_test, dtype=torch.float32)
    predict_tensor = torch.zeros([BATCHES, x_test.shape[0]])
    for batch in range(int(BATCHES)):
        prediction = net(x_test)
        predict_tensor[batch, :] = prediction.reshape([1, prediction.shape[0]])
    mean, std = cal_mean_std_in_column(predict_tensor)

    # 画图
    plot_cal_epistemic(x_train.detach().numpy(), y_train.detach().numpy(),
            x_test.detach().numpy(), mean.detach().numpy(), std.detach().numpy(),
            x_test.detach().numpy(), y_test)
    # my_plot(x_train.detach().numpy(), y_train.detach().numpy(),
    #         x_test.detach().numpy(), y_prediction.detach().numpy(),
    #         x_test.detach().numpy(), y_test)
