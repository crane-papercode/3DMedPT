import glob
import os
import numpy as np
import torch
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from natsort import natsorted

BASE = os.path.dirname(os.path.abspath(__file__)) + "/data/IntrA3D/"


class Intra3D(Dataset):
    def __init__(self, train_mode='train', cls_state=True, npoints=2048, data_aug=True, choice=0):
        self.npoints = npoints  # 2048 pts
        self.data_augmentation = data_aug
        self.datapath = []
        self.label = {}
        self.cls_state = cls_state
        self.train_mode = train_mode

        choice_list = [i for i in range(5)]
        poped_val = choice_list.pop(choice)

        if self.cls_state:
            self.label[0] = glob.glob(BASE + "generated/vessel/ad/" + "*.ad")  # label 0: healthy; 1694 files; negSplit
            self.label[1] = glob.glob(BASE + "generated/aneurysm/ad/" + "*.ad") + \
                            glob.glob(BASE + "annotated/ad/" + "*.ad")  # label 1: unhealthy; 331 files

            train_test_set_ann = natsorted(glob.glob(BASE + "fileSplit/cls/ann_clsSplit_" + "*.txt"))  # label 1
            train_test_set_neg = natsorted(glob.glob(BASE + "fileSplit/cls/negSplit_" + "*.txt"))  # label 0
            train_set = [train_test_set_ann[i] for i in choice_list] + [train_test_set_neg[i] for i in choice_list]
            test_set = [train_test_set_ann[poped_val]] + [train_test_set_neg[poped_val]]
        else:
            train_test_set = natsorted(glob.glob(BASE + "fileSplit/seg/annSplit_" + "*.txt"))
            train_set = [train_test_set[i] for i in choice_list]
            test_set = [train_test_set[poped_val]]

        if self.train_mode == 'train':
            for file in train_set:
                with open(file, 'r') as f:
                    for line in f.readlines():
                        self.datapath.append(BASE + line.strip())
        elif self.train_mode == 'test':
            for file in test_set:
                with open(file, 'r') as f:
                    for line in f.readlines():
                        self.datapath.append(BASE + line.strip())
        elif self.train_mode == 'all':
            for file in (train_set + test_set):
                with open(file, 'r') as f:
                    for line in f.readlines():
                        self.datapath.append(BASE + line.strip())
        else:
            print("Error")
            raise Exception("training mode invalid")

    def __getitem__(self, index):
        curr_file = self.datapath[index]
        cls = None
        if self.cls_state:
            if curr_file in self.label[0]:
                cls = torch.from_numpy(np.array([0]).astype(np.int64))
            elif curr_file in self.label[1]:
                cls = torch.from_numpy(np.array([1]).astype(np.int64))
            else:
                print("Error found!!!")
                exit(-1)

        point_set = np.loadtxt(curr_file)[:, :-1].astype(np.float32)  # [x, y, z, norm_x, norm_y, norm_z]
        seg = np.loadtxt(curr_file)[:, -1].astype(np.int64)  # [seg_label]
        seg[np.where(seg == 2)] = 1  # making boundary lines (label 2) to A. (label 1)

        # random choice
        if point_set.shape[0] < self.npoints:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        else:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        point_set = point_set[choice, :]
        seg = seg[choice]

        # normalization to unit ball
        point_set[:, :3] = point_set[:, :3] - np.mean(point_set[:, :3], axis=0)  # x, y, z
        dist = np.max(np.sqrt(np.sum(point_set[:, :3] ** 2, axis=1)), 0)
        point_set[:, :3] = point_set[:, :3] / dist

        # data augmentation
        if self.data_augmentation:
            # jitter points (x,y,z)
            if self.train_mode == 'train':
                point_set[:, :3] = random_scale(point_set[:, :3])
                point_set[:, :3] = translate_pointcloud(point_set[:, :3])
                # point_set[:, :3] = point_set[:, :3]
            if self.train_mode == 'test':
                # point_set[:, :3] = random_scale(point_set[:, :3])
                # point_set[:, :3] = translate_pointcloud(point_set[:, :3])
                point_set[:, :3] = point_set[:, :3]

        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        return (point_set, seg) if not self.cls_state else (point_set, cls)

    def __len__(self):
        return len(self.datapath)


def get_train_valid_loader(num_workers=4, pin_memory=False, batch_size=4, npoints=2048, choice=0):
    train_dataset = Intra3D(train_mode='train', cls_state=True, npoints=npoints, data_aug=True, choice=choice)
    valid_dataset = Intra3D(train_mode='test', cls_state=True, npoints=npoints, data_aug=False, choice=choice)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, valid_loader, len(train_dataset)


def jitter(point_data, sigma=0.01, clip=0.05):
    N, C = point_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_data += point_data
    return jittered_data


def random_scale(point_data, scale_low=0.8, scale_high=1.2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original batch of point clouds
        Return:
            Nx3 array, scaled batch of point clouds
    """
    scale = np.random.uniform(low=scale_low, high=scale_high, size=[3])
    scaled_pointcloud = np.multiply(point_data, scale).astype('float32')
    return scaled_pointcloud


def translate_pointcloud(pointcloud):
    shift = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(pointcloud, shift).astype('float32')
    return translated_pointcloud


if __name__ == '__main__':
    dataset_test = Intra3D(cls_state=True, npoints=1024, data_aug=True, choice=0)
    for i, (point_set, labels) in enumerate(dataset_test):
        # print(point_set[:, :3].shape)
        print(point_set.shape, labels.shape)
        if i == 3:
            break
