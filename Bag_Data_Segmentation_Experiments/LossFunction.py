import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, _is_training=True):
        N = target.size(0)
        if _is_training:
            smooth = 1
        else:
            smooth = 1e-4

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + 2 * smooth)
        loss = 1 - loss.sum() / N

        return loss


class MultiClassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MultiClassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


class MeanAccuracy(nn.Module):
    def __init__(self):
        super(MeanAccuracy, self).__init__()

    def forward(self, _input, _target):
        # 拉成一维
        _input = _input.view(_input.size(0), -1)
        _target = _target.view(_target.size(0), -1)
        # 0/1二值化
        _input[torch.where(_input >= 0.5)] = 1
        _input[torch.where(_input != 1)] = 0
        # 相减取绝对值
        _temp = torch.abs(_target - _input)
        error = _temp.sum()
        if _input.shape[0] == 1:
            _accuracy = error / _input.shape[1]
        else:
            _accuracy = error / _input.shape[0]
        return 1 - _accuracy
