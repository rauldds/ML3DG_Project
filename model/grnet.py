# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-09-06 11:35:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:20:36
# @Email:  cshzxie@gmail.com

import torch

from model.extensions.cubic_feature_sampling import CubicFeatureSampling
from model.extensions.gridding import Gridding, GriddingReverse


class RandomPointSampling(torch.nn.Module):
    def __init__(self, n_points):
        super(RandomPointSampling, self).__init__()
        self.n_points = n_points

    def forward(self, pred_cloud, partial_cloud=None):
        if partial_cloud is not None:
            pred_cloud = torch.cat([partial_cloud, pred_cloud], dim=1)

        _ptcloud = torch.split(pred_cloud, 1, dim=0)
        ptclouds = []
        for p in _ptcloud:
            non_zeros = torch.sum(p, dim=2).ne(0)
            p = p[non_zeros].unsqueeze(dim=0)
            n_pts = p.size(1)
            if n_pts < self.n_points:
                rnd_idx = torch.cat([torch.randint(0, n_pts, (self.n_points, ))])
            else:
                rnd_idx = torch.randperm(p.size(1))[:self.n_points]
            ptclouds.append(p[:, rnd_idx, :])

        return torch.cat(ptclouds, dim=0).contiguous()


class GRNet(torch.nn.Module):
    def __init__(self):
        super(GRNet, self).__init__()
        self.gridding = Gridding(scale=64)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.fc5 = torch.nn.Sequential(
            torch.nn.Linear(16384, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout()
        )
        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(2048, 16384),
            torch.nn.ReLU(),
            torch.nn.Dropout()
        )
        self.dconv7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.dconv8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.dconv9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.dconv10 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.ReLU()
        )
        self.gridding_rev = GriddingReverse(scale=64)
        self.point_sampling = RandomPointSampling(n_points=2048)
        self.feature_sampling = CubicFeatureSampling()
        self.fc11 = torch.nn.Sequential(
            torch.nn.Linear(1792, 1792),
            torch.nn.ReLU()
        )
        self.fc12 = torch.nn.Sequential(
            torch.nn.Linear(1792, 448),
            torch.nn.ReLU()
        )
        self.fc13 = torch.nn.Sequential(
            torch.nn.Linear(448, 112),
            torch.nn.ReLU()
        )
        self.fc14 = torch.nn.Linear(112, 10)
        self.fc15 = torch.nn.Sequential(
            torch.nn.Linear(20480, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 15),
        )

    def forward(self, data):
        #partial_cloud = data
        #print(partial_cloud.size())     # torch.Size([batch_size, 2048, 3])
        #pt_features_64_l = self.gridding(partial_cloud).view(-1, 1, 64, 64, 64)
        #print(pt_features_64_l.size())  # torch.Size([batch_size, 1, 64, 64, 64])
        pt_features_32_l = self.conv1(data)
        # print(pt_features_32_l.size())  # torch.Size([batch_size, 32, 32, 32, 32])
        pt_features_16_l = self.conv2(pt_features_32_l)
        # print(pt_features_16_l.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_8_l = self.conv3(pt_features_16_l)
        # print(pt_features_8_l.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_4_l = self.conv4(pt_features_8_l)
        # print(pt_features_4_l.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        features = self.fc5(pt_features_4_l.view(-1, 16384))
        # print(features.size())          # torch.Size([batch_size, 2048])
        pt_features_4_r = self.fc6(features).view(-1, 256, 4, 4, 4) + pt_features_4_l
        # print(pt_features_4_r.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        pt_features_8_r = self.dconv7(pt_features_4_r) + pt_features_8_l
        # print(pt_features_8_r.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_16_r = self.dconv8(pt_features_8_r) + pt_features_16_l
        # print(pt_features_16_r.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_32_r = self.dconv9(pt_features_16_r) + pt_features_32_l
        # print(pt_features_32_r.size())  # torch.Size([batch_size, 32, 32, 32, 32])
        completion_pred = self.dconv10(pt_features_32_r) + data
        # print(completion_pred.size())  # torch.Size([batch_size, 1, 64, 64, 64])
        sparse_cloud = self.gridding_rev(completion_pred.squeeze(dim=1))
        #print(sparse_cloud.size())      # torch.Size([batch_size, 262144, 3])
        sparse_cloud = self.point_sampling(sparse_cloud)
        # print(sparse_cloud.size())      # torch.Size([batch_size, 2048, 3])
        point_features_32 = self.feature_sampling(sparse_cloud, pt_features_32_r).view(-1, 2048, 256)
        # print(point_features_32.size()) # torch.Size([batch_size, 2048, 256])
        point_features_16 = self.feature_sampling(sparse_cloud, pt_features_16_r).view(-1, 2048, 512)
        # print(point_features_16.size()) # torch.Size([batch_size, 2048, 512])
        point_features_8 = self.feature_sampling(sparse_cloud, pt_features_8_r).view(-1, 2048, 1024)
        # print(point_features_8.size())  # torch.Size([batch_size, 2048, 1024])
        point_features = torch.cat([point_features_32, point_features_16, point_features_8], dim=2)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 1792])
        point_features = self.fc11(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 1792])
        point_features = self.fc12(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 448])
        point_features = self.fc13(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 112])
        point_features = self.fc14(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 10])
        class_pred = self.fc15(point_features.flatten(start_dim=1))
        # print(class_pred.size())    # torch.Size([batch_size, 15])

        return completion_pred, class_pred

class GRNet_comp(torch.nn.Module):
    def __init__(self):
        super(GRNet_comp, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.fc5 = torch.nn.Sequential(
            torch.nn.Linear(16384, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout()
        )
        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(2048, 16384),
            torch.nn.ReLU(),
            torch.nn.Dropout()
        )
        self.dconv7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.dconv8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.dconv9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.dconv10 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.ReLU()
        )

    def forward(self, data):
        #partial_cloud = data
        #print(partial_cloud.size())     # torch.Size([batch_size, 2048, 3])
        #pt_features_64_l = self.gridding(partial_cloud).view(-1, 1, 64, 64, 64)
        #print(pt_features_64_l.size())  # torch.Size([batch_size, 1, 64, 64, 64])
        pt_features_32_l = self.conv1(data)
        # print(pt_features_32_l.size())  # torch.Size([batch_size, 32, 32, 32, 32])
        pt_features_16_l = self.conv2(pt_features_32_l)
        # print(pt_features_16_l.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_8_l = self.conv3(pt_features_16_l)
        # print(pt_features_8_l.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_4_l = self.conv4(pt_features_8_l)
        # print(pt_features_4_l.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        features = self.fc5(pt_features_4_l.view(-1, 16384))
        # print(features.size())          # torch.Size([batch_size, 2048])
        pt_features_4_r = self.fc6(features).view(-1, 256, 4, 4, 4) + pt_features_4_l
        # print(pt_features_4_r.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        pt_features_8_r = self.dconv7(pt_features_4_r) + pt_features_8_l
        # print(pt_features_8_r.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_16_r = self.dconv8(pt_features_8_r) + pt_features_16_l
        # print(pt_features_16_r.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_32_r = self.dconv9(pt_features_16_r) + pt_features_32_l
        # print(pt_features_32_r.size())  # torch.Size([batch_size, 32, 32, 32, 32])
        completion_pred = self.dconv10(pt_features_32_r) + data
        # print(completion_pred.size())  # torch.Size([batch_size, 1, 64, 64, 64])

        skip = {
            '32_r': pt_features_32_r,
            '16_r': pt_features_16_r,
            '8_r': pt_features_8_r
        }
        return completion_pred, skip

class GRNet_clas(torch.nn.Module):
    def __init__(self):
        super(GRNet_clas, self).__init__()
        self.gridding_rev = GriddingReverse(scale=64)
        self.maxpooling = torch.nn.MaxPool3d(kernel_size=4)
        self.point_sampling = RandomPointSampling(n_points=2048)
        self.feature_sampling = CubicFeatureSampling()
        self.fc11 = torch.nn.Sequential(
            torch.nn.Linear(1792, 1792),
            torch.nn.ReLU()
        )
        self.fc12 = torch.nn.Sequential(
            torch.nn.Linear(1792, 448),
            torch.nn.ReLU()
        )
        self.fc13 = torch.nn.Sequential(
            torch.nn.Linear(448, 100),
            torch.nn.ReLU()
        )
        self.fc14 = torch.nn.Sequential(
            torch.nn.Conv1d(2048, 1024, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU()
        )
        self.fc15 = torch.nn.Sequential(
            torch.nn.Linear(3200, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,15)
        )

    def forward(self, data, skip):
        # Modification to hihglight points close to the object surface
        # and reduce the importance of the rest of the grid
        completion_pred = (abs(data) <= 1).float()
        # print(completion_pred.size())  # torch.Size([batch_size, 1, 64, 64, 64])
        sparse_cloud = self.gridding_rev(completion_pred.squeeze(dim=1))
        #print(sparse_cloud.size())      # torch.Size([batch_size, 262144, 3])
        sparse_cloud = self.maxpooling(sparse_cloud.reshape([sparse_cloud.shape[0],3,64,64,64]))
        #print(sparse_cloud.size())      # torch.Size([batch_size, 3, 16, 16, 16])
        sparse_cloud = self.point_sampling(sparse_cloud.reshape([sparse_cloud.shape[0],4096,3]))
        # print(sparse_cloud.size())      # torch.Size([batch_size, 2048, 3])
        point_features_32 = self.feature_sampling(sparse_cloud, skip["32_r"]).view(-1, 2048, 256)
        # print(point_features_32.size()) # torch.Size([batch_size, 2048, 256])
        point_features_16 = self.feature_sampling(sparse_cloud, skip["16_r"]).view(-1, 2048, 512)
        # print(point_features_16.size()) # torch.Size([batch_size, 2048, 512])
        point_features_8 = self.feature_sampling(sparse_cloud, skip["8_r"]).view(-1, 2048, 1024)
        # print(point_features_8.size())  # torch.Size([batch_size, 2048, 1024])
        point_features = torch.cat([point_features_32, point_features_16, point_features_8], dim=2)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 1792])
        point_features = self.fc11(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 1792])
        point_features = self.fc12(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 448])
        point_features = self.fc13(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 100])
        point_features = self.fc14(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 32, 100])
        class_pred = self.fc15(point_features.flatten(start_dim=1))
        # print(class_pred.size())    # torch.Size([batch_size, 15])

        return class_pred
