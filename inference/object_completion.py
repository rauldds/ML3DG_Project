from pathlib import Path
from statistics import mode
import numpy as np
import torch
from model.grnet import GRNet
import skimage
import trimesh
import open3d as o3d

input_sdf_path = "/media/rauldds/TOSHIBA EXT/ML3G/Davids targets/DATASET_test/InputData/SDFs/pillow/scene0577_00_00016/0_scene0577_00_00016.npz"


model = GRNet()
#print(GRNet())
model = model.cuda()

train_parameters = []
train_parameters += list(model.gridding.parameters())
train_parameters += list(model.conv1.parameters())
train_parameters += list(model.conv2.parameters())
train_parameters += list(model.conv3.parameters())
train_parameters += list(model.conv4.parameters())
train_parameters += list(model.fc5.parameters())
train_parameters += list(model.fc6.parameters())
train_parameters += list(model.dconv7.parameters())
train_parameters += list(model.dconv8.parameters())
train_parameters += list(model.dconv9.parameters())
train_parameters += list(model.dconv10.parameters())
train_parameters += list(model.gridding_rev.parameters())

optimizer = torch.optim.Adam(train_parameters)

checkpoint = torch.load("./ckpts/ckpt-best.pth")

model.load_state_dict(checkpoint['model_state_dict'])

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print("success loading")

data = np.load(input_sdf_path, allow_pickle=True)

voxels = data["arr_0"]*32

voxels = np.clip(voxels,-3,3).reshape((1, 1, 64, 64, 64))

print(voxels.shape)

voxels = torch.from_numpy(voxels).cuda()

model.eval()

recon, class_pred = model.forward(data=voxels)

print(recon.shape)
print(class_pred.shape)

recon = recon.detach().cpu().numpy()

recon = recon.reshape((64, 64, 64))

vertices, faces, normals, _ = skimage.measure.marching_cubes(recon, level=0)

mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

mesh = mesh.as_open3d

o3d.visualization.draw_geometries([mesh],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[0, 0, 0],
                                  up=[0.2304, -0.8825, 0.4101])