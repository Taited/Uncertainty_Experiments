import os
import os.path

import cv2
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from onehot import onehot

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class BagDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir('bag_data'))

    def __getitem__(self, idx):
        img_name = os.listdir('bag_data')[idx]
        imgA = cv2.imread('bag_data/' + img_name)
        imgA = cv2.resize(imgA, (160, 160))
        imgB = cv2.imread('bag_data_msk/' + img_name, 0)
        imgB = cv2.resize(imgB, (160, 160))
        imgB = imgB / 255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        imgB = imgB.transpose(2, 0, 1)
        imgB = torch.FloatTensor(imgB)
        # print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)

        return imgA, imgB, idx


bag = BagDataset(transform)

train_size = int(0.9 * len(bag))
test_size = len(bag) - train_size
train_dataset, test_dataset = random_split(bag, [train_size, test_size])


if __name__ == '__main__':
    for i in range(379, 600):
        img = cv2.imread('./bag_data/{}.png'.format(i))
        cv2.imwrite('./bag_data/{}.jpg'.format(i), img)
        os.remove('./bag_data/{}.png'.format(i))
        print('{} success!'.format(i))
