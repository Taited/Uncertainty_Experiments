from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom

from torch.utils.data import DataLoader
from BagData import test_dataset, train_dataset
from FCN import VGGNet, UNet
from LossFunction import DiceLoss, MeanAccuracy
from UNet import UNet as unet
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

from time import time


# 使用预训练好的VGG作为UNet的backbone
def train(epo_num=50, show_vgg_params=False, is_save=False):
    vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 4
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = UNet(pretrained_net=vgg_model, n_class=2)
    fcn_model = fcn_model.to(device)
    criterion_BCE = nn.BCELoss().to(device)
    criterion_DICE = DiceLoss().to(device)
    optimizer = optim.Adam(fcn_model.parameters(), lr=1e-3, weight_decay=1e-4)

    all_train_iter_loss = []
    all_test_iter_loss = []

    # start timing
    prev_time = datetime.now()
    for epo in range(epo_num):

        train_loss = 0
        fcn_model.train()
        for index, (bag, bag_msk, image_id) in enumerate(train_data_loader):
            # bag.shape is torch.Size([4, 3, 160, 160])
            # bag_msk.shape is torch.Size([4, 2, 160, 160])

            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            optimizer.zero_grad()
            output = fcn_model(bag)
            output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
            loss_BCE = criterion_BCE(output, bag_msk)
            loss_Dice = criterion_DICE(output, bag_msk)
            loss = loss_BCE + loss_Dice
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
            output_np = np.argmin(output_np, axis=1)
            bag_msk_np = bag_msk.cpu().detach().numpy().copy()  # bag_msk_np.shape = (4, 2, 160, 160)
            bag_msk_np = np.argmin(bag_msk_np, axis=1)

            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_data_loader), iter_loss))
                # vis.close()
                vis.images(output_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction'))
                vis.images(bag_msk_np[:, None, :, :], win='train_label', opts=dict(title='label'))
                vis.line(all_train_iter_loss, win='train_iter_loss', opts=dict(title='train iter loss'))

        test_loss = 0
        fcn_model.eval()
        with torch.no_grad():
            for index, (bag, bag_msk) in enumerate(test_data_loader):

                bag = bag.to(device)
                bag_msk = bag_msk.to(device)

                optimizer.zero_grad()
                output = fcn_model(bag)
                output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
                loss = criterion_DICE(output, bag_msk)
                IOU = 1 - loss
                loss = IOU / (2 - IOU)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
                output_np = np.argmin(output_np, axis=1)
                bag_msk_np = bag_msk.cpu().detach().numpy().copy()  # bag_msk_np.shape = (4, 2, 160, 160)
                bag_msk_np = np.argmin(bag_msk_np, axis=1)

                if np.mod(index, 15) == 0:
                    print(r'Testing... Open http://localhost:8097/ to see test result.')
                    # vis.close()
                    vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction'))
                    vis.images(bag_msk_np[:, None, :, :], win='test_label', opts=dict(title='label'))
                    vis.line(all_test_iter_loss, win='test_IOU', opts=dict(title='test IOU'))

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch train loss = %f, epoch test IOU = %f, %s'
              % (train_loss / len(train_data_loader), test_loss / len(test_data_loader), time_str))

        if np.mod(epo, 5) == 0 and is_save:
            torch.save(fcn_model, 'checkpoints/fcn_model_{}.pt'.format(epo))
            print('saveing checkpoints/fcn_model_{}.pt'.format(epo))


# 带有Uncertainty的训练函数，不使用预训练好的VGG
def train_uncertainty(epo_num=50, is_save=False):
    vis = visdom.Visdom()

    BATCH_SIZE = 4
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_model = unet(n_class=2)
    net_model = net_model.to(device)
    criterion_BCE = nn.BCELoss().to(device)
    criterion_DICE = DiceLoss().to(device)
    optimizer = optim.Adam(net_model.parameters(), lr=1e-3, weight_decay=1e-4)

    all_train_iter_loss = []
    all_test_iter_loss = []

    # start timing
    prev_time = datetime.now()
    net_model.train()
    for epo in range(epo_num):

        train_loss = 0

        for index, (bag, bag_msk, image_id) in enumerate(train_dataloader):
            # bag.shape is torch.Size([4, 3, 160, 160])
            # bag_msk.shape is torch.Size([4, 2, 160, 160])

            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            optimizer.zero_grad()
            output = net_model(bag)
            output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
            loss_BCE = criterion_BCE(output, bag_msk)
            # loss_Dice = criterion_DICE(output, bag_msk)
            # loss = loss_BCE + loss_Dice
            loss = loss_BCE
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
            output_np = np.argmin(output_np, axis=1)
            bag_msk_np = bag_msk.cpu().detach().numpy().copy()  # bag_msk_np.shape = (4, 2, 160, 160)
            bag_msk_np = np.argmin(bag_msk_np, axis=1)

            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))
                # vis.close()
                vis.images(output_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction'))
                vis.images(bag_msk_np[:, None, :, :], win='train_label', opts=dict(title='label'))
                vis.line(all_train_iter_loss, win='train_iter_loss', opts=dict(title='train iter loss'))

        test_loss = 0
        with torch.no_grad():
            for index, (bag, bag_msk, image_id) in enumerate(test_dataloader):

                bag = bag.to(device)
                bag_msk = bag_msk.to(device)

                optimizer.zero_grad()
                output = net_model(bag)
                output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
                loss = criterion_DICE(output, bag_msk)
                IOU = 1 - loss
                loss = IOU / (2 - IOU)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
                output_np = np.argmin(output_np, axis=1)
                bag_msk_np = bag_msk.cpu().detach().numpy().copy()  # bag_msk_np.shape = (4, 2, 160, 160)
                bag_msk_np = np.argmin(bag_msk_np, axis=1)


                if np.mod(index, 15) == 0:
                    print(r'Testing... Open http://localhost:8097/ to see test result.')
                    # vis.close()
                    # MC Sample
                    Sample_Time = 50
                    uncertainty_out = torch.zeros(
                        [Sample_Time, output_np.shape[0], output_np.shape[1], output_np.shape[2]])
                    for _ in range(Sample_Time):
                        output = net_model(bag)
                        uncertainty_out[_, :, :, :] = output[:, 0, :, :].reshape(output_np.shape)
                    uncertainty_out = uncertainty_out.std(dim=0)
                    heat_map = _tensor_to_heat_map(uncertainty_out)
                    vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction'))
                    vis.images(bag_msk_np[:, None, :, :], win='test_label', opts=dict(title='label'))
                    vis.images(heat_map.transpose(2, 0, 1)[::-1, ...], win='epistemic uncertainty',
                               opts=dict(title='epistemic uncertainty'))
                    vis.line(all_test_iter_loss, win='test_IOU', opts=dict(title='test IOU'))

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch train loss = %f, epoch test IOU = %f, %s'
              % (train_loss / len(train_dataloader), test_loss / len(test_dataloader), time_str))

        if np.mod(epo, 5) == 0 and is_save:
            torch.save(net_model, 'checkpoints/bce_loss_{}.pt'.format(epo))
            print('saveing checkpoints/bce_loss_{}.pt'.format(epo))


# 通过加载预训练好的模型来进行评估，并进行uncertainty估计和矫正
def model_test(_model='bce_loss_{}.pt', _is_correct=False):
    BATCH_SIZE = 1
    T = 10
    FLAG = 0.46
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('dataset loaded!')
    criterion_DICE = DiceLoss().to(device)
    criterion_Accuracy = MeanAccuracy().to(device)
    model = load_model(_model=_model)
    print('model loaded!')
    with torch.no_grad():
        # 不同uncertainty下的损失
        _train_uncertainty_loss_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Dropout采样结果
        _train_prediction_list = []
        _train_std_list = []
        # 准确度调整后结果
        _train_accuracy_before_list = []
        _train_accuracy_after_list = []
        for index, (bag, bag_msk, image_id) in enumerate(train_dataloader):
            bag_msk = bag_msk.to(device)
            _mean, _std = monte_carlo_sample(model, bag.to(device), _t=T, _device=device)
            # 只取一个维度
            bag_msk = bag_msk[:, 1, :, :]
            _mean = _mean[:, 1, :, :]
            _std = _std[:, 1, :, :]
            _train_prediction_list.append(_mean)
            _train_std_list.append(_std)
            # 计算相关性
            _train_uncertainty_loss_list = select_uncertainty(_mean, _std, bag_msk,
                                                              _train_uncertainty_loss_list, criterion_Accuracy)
            # 计算loss
            loss = criterion_Accuracy(_mean, bag_msk)
            _train_accuracy_before_list.append(loss.item())
            if _is_correct:
                _mean = correct_with_uncertainty(_mean, _std, FLAG)
                loss_after = criterion_Accuracy(_mean, bag_msk)
                _train_accuracy_after_list.append(loss_after.item())
            # if index >= 30:
            #     break
            print('The Accuracy of Train data id:{} is {}. The corrected Accuracy is {}'.
                  format(index, loss.item(), loss_after.item()))
        _train_uncertainty_loss = np.array(_train_uncertainty_loss_list) / (index + 1)

        # uncertainty loss 结果
        _test_uncertainty_loss_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        _test_prediction_list = []
        _test_std_list = []
        _test_accuracy_before_list = []
        _test_accuracy_after_list = []
        for index, (bag, bag_msk, image_id) in enumerate(test_dataloader):
            bag_msk = bag_msk.to(device)
            _mean, _std = monte_carlo_sample(model, bag.to(device), _t=T, _device=device)
            bag_msk = bag_msk[:, 1, :, :]
            _mean = _mean[:, 1, :, :]
            _std = _std[:, 1, :, :]
            _test_uncertainty_loss_list = select_uncertainty(_mean, _std, bag_msk, _test_uncertainty_loss_list,
                                                             criterion_Accuracy)
            _test_prediction_list.append(_mean)
            _test_std_list.append(_std)
            loss = criterion_Accuracy(_mean, bag_msk)
            _test_accuracy_before_list.append(loss.item())
            if _is_correct:
                _mean = correct_with_uncertainty(_mean, _std, FLAG)
                loss_after = criterion_Accuracy(_mean, bag_msk)
                _test_accuracy_after_list.append(loss_after.item())
            print('The accuracy of Test data id:{} is {}. The corrected accuracy is {}'.
                  format(index, loss.item(), loss_after.item()))
        _test_uncertainty_loss = np.array(_test_uncertainty_loss_list) / (index + 1)
    return _train_prediction_list, _train_std_list, _test_prediction_list, _test_std_list, \
           _train_accuracy_before_list, _train_accuracy_after_list, \
           _test_accuracy_before_list, _test_accuracy_after_list, \
           _train_uncertainty_loss, _test_uncertainty_loss


# 通过MC Dropout得到不确定性
def monte_carlo_sample(_net, _input, _t: int, _device):
    # sample _t times
    with torch.no_grad():
        _net.train()  # 打开dropout
        uncertainty_result = torch.zeros([_t, _input.shape[0], 2, _input.shape[2], _input.shape[3]]).to(_device)

        for _ in range(_t):
            prediction = _net(_input)
            uncertainty_result[_, :, :, :, :] = prediction
    _mean = uncertainty_result.mean(dim=0)
    _std = uncertainty_result.std(dim=0)
    return _mean, _std


def load_model(_model='bce_loss_{}.pt'):
    PATH = './checkpoints/' + _model.format(85)
    model = torch.load(PATH)
    return model


def data_box_plot(_tensor_list, _name='Train Dataset'):
    # tensor转换成一维张量
    _shape = _tensor_list[0].view(_tensor_list[0].size(0), -1).shape[1]
    _np_array = np.zeros([len(_tensor_list), _shape])
    for _i in range(len(_tensor_list)):
        _tensor = _tensor_list[_i].view(_tensor_list[_i].size(0), -1)
        _np_array[_i] = _tensor.cpu().numpy()
    _np_array = _np_array.flatten()
    plt.figure()
    plt.boxplot(_np_array)
    plt.xticks([1], ["{} std".format(_name)])
    plt.grid(axis="y", ls=":", lw=1, color="gray", alpha=0.4)
    plt.savefig('./figure/{}.png'.format(_name))
    plt.show()
    return _np_array


def data_split(_np_array):
    my_dict = {'0~0.1': 0, '0.1~0.15': 0, '0.15~0.2': 0, '0.2~0.25': 0, '0.25~0.3': 0,
               '0.3~0.35': 0, '0.35~0.4': 0, '0.4~0.45': 0, '0.45~0.5': 0, '0.5~1': 0}
    for item in _np_array:
        if item <= 0.1:
            my_dict['0~0.1'] += 1
        elif item <= 0.15:
            my_dict['0.1~0.15'] += 1
        elif item <= 0.2:
            my_dict['0.15~0.2'] += 1
        elif item <= 0.25:
            my_dict['0.2~0.25'] += 1
        elif item <= 0.3:
            my_dict['0.25~0.3'] += 1
        elif item <= 0.35:
            my_dict['0.3~0.35'] += 1
        elif item <= 0.4:
            my_dict['0.35~0.4'] += 1
        elif item <= 0.45:
            my_dict['0.4~0.45'] += 1
        elif item <= 0.5:
            my_dict['0.45~0.5'] += 1
        else:
            my_dict['0.5~1'] += 1
    return my_dict


def draw_uncertainty_distribution(_train_std_list, _test_std_list):
    _train_std_np_array = data_box_plot(_train_std_list, _name='Train Dataset')
    _test_std_np_array = data_box_plot(_test_std_list, _name='Test Dataset')
    _train_std_dict = data_split(_train_std_np_array)
    _test_std_dict = data_split(_test_std_np_array)
    _train_height = np.array(list(_train_std_dict.values())[1:])
    _test_height = np.array(list(_test_std_dict.values())[1:])
    plot_distribution(_train_height, _test_height, _name='uncertainty_distribution')


def correct_with_uncertainty(_mean, _std, _flag=0.3):
    _the_id = torch.where(_std >= _flag)
    _mean[_the_id] = 1 - _mean[_the_id]
    return _mean


def select_uncertainty(_mean, _std, _mask, _list, _criterion):
    my_list = [-0.001, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1]
    for i in range(1, len(my_list)):
        _temp_x = _mean[torch.where((_std <= my_list[i]) & (_std > my_list[i - 1]))]
        _temp_y = _mask[torch.where((_std <= my_list[i]) & (_std > my_list[i - 1]))]
        try:
            loss = _criterion(_temp_x, _temp_y)
            _list[i - 1] += loss.item()
        except RuntimeError:
            pass
    return _list


def _tensor_to_heat_map(_image_tensor):
    # 先将输入的图片拆开
    _result = _image_tensor[0, :, :].reshape(_image_tensor.shape[1:3])
    sns.set()
    ax = sns.heatmap(_result.detach().numpy(), cbar=False,
                     xticklabels=[], yticklabels=[])
    plt.savefig('./figure/uncertainty.png')
    _image = cv2.imread('./figure/uncertainty.png')
    return _image


def plot_distribution(_train_height, _test_height, _name='my_picture'):
    my_dict = {'0~0.1': 0, '0.1~0.15': 0, '0.15~0.2': 0, '0.2~0.25': 0, '0.25~0.3': 0,
               '0.3~0.35': 0, '0.35~0.4': 0, '0.4~0.45': 0, '0.45~0.5': 0, '0.5~1': 0}
    # 从1开始表示不需要展示0~0.1区间的数据
    _x_label = list(my_dict.keys())[1:]
    # plot
    x = np.arange(1, 10)
    plt.figure(figsize=(7, 6.0))
    plt.title(_name)
    plt.subplot(2, 1, 1)
    plt.bar(x=x, height=_train_height, width=0.3)
    plt.xticks(x, _x_label, fontsize=10)
    plt.title('Train Dataset')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.bar(x=x, height=_test_height, width=0.3)
    plt.xticks(x, _x_label, fontsize=10)
    plt.title('Test Dataset')
    plt.grid()
    plt.savefig('./figure/{}.png'.format(_name))
    plt.show()


if __name__ == "__main__":
    begin_time = time()

    # 训练的函数
    # train(epo_num=35, show_vgg_params=False)
    # train_uncertainty(epo_num=100, is_save=True)

    # 模型测试
    train_prediction_list, train_std_list, test_prediction_list, test_std_list, \
        train_accuracy_before_list, train_accuracy_after_list, test_accuracy_before_list, test_accuracy_after_list, \
        train_uncertainty_loss, test_uncertainty_loss \
        = model_test(_model='bce_loss_{}.pt', _is_correct=True)

    # 绘图
    draw_uncertainty_distribution(train_std_list, test_std_list)
    plot_distribution(train_uncertainty_loss[1:], test_uncertainty_loss[1:], _name='mean_accurate_distribution')

    # 输出平均准确度
    train_mean_acc = np.array(train_accuracy_before_list).mean()
    train_correct_mean_acc = np.array(train_accuracy_after_list).mean()
    print('Train dataset mean accuracy is {}. After correction, mean accuracy is {}'.
          format(train_mean_acc, train_correct_mean_acc))
    test_mean_acc = np.array(test_accuracy_before_list).mean()
    test_correct_mean_acc = np.array(test_accuracy_after_list).mean()
    print('Test  dataset mean accuracy is {}. After correction, mean accuracy is {}'.
          format(test_mean_acc, test_correct_mean_acc))

    # 输出提升效果
    print('The enhancement of train data is {}'.format(train_correct_mean_acc - train_mean_acc))
    print('The enhancement of test  data is {}'.format(test_correct_mean_acc - test_mean_acc))

    # 输出时间
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time)
