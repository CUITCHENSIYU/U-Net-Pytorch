from torch.utils.data import Dataset
import PIL.Image as Image
import os
import numpy as np
'''本方法的任务：对训练数据、测试数据的读取与矩阵化处理（在'main.py'被转成tensor）'''

def train_dataset(img_root, label_root):
    imgs = []
    n = len(os.listdir(img_root))#遍历root目录下的所有文件夹
    for i in range(n):
        img = os.path.join(img_root, "%d.png" % i)
        label = os.path.join(label_root, "%d_mask.png" % i)

        imgs.append((img, label))#组合标签和图片
    return imgs


def test_dataset(img_root):
    imgs = []
    n = len(os.listdir(img_root))
    for i in range(n):
        img = os.path.join(img_root, "%d.png" % i)
        imgs.append(img)
    return imgs


class TrainDataset(Dataset):
    def __init__(self, img_root, label_root, transform=None, target_transform=None):
        imgs = train_dataset(img_root, label_root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)


class TestDataset(Dataset):
    def __init__(self, img_root, transform=None, target_transform=None):
        imgs = test_dataset(img_root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        img_x = Image.open(x_path)#读取图像
        if self.transform is not None:
            img_x = self.transform(img_x)
        return img_x

    def __len__(self):
        return len(self.imgs)

#调色板
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

#按照需求将结果处理成更加容易辨识的结果，由于本次实验为图像分割，因此将num_class=2,即分为两种颜色；
def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255