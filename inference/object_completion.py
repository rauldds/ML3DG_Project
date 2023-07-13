from pathlib import Path
from statistics import mode
import numpy as np
import torch
from model.grnet import GRNet_comp
import skimage
import trimesh
import open3d as o3d

input_sdf_path = "/home/rauldds/Documents/InputSamples/SDFs/table/scene0507_00_00005/4_scene0507_00_00005.npz"


model = GRNet_comp()
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters())

checkpoint = torch.load("./ckpts/ScanObjectNN/ckpt-best-completion.pth")

model.load_state_dict(checkpoint['model_comp'])

optimizer.load_state_dict(checkpoint['cmp_optim'])

print("success loading")

data = np.load(input_sdf_path, allow_pickle=True)

voxels = data["arr_0"]*32

voxels = np.clip(voxels,-3,3).reshape((1, 1, 64, 64, 64))

print(voxels.shape)

voxels = torch.from_numpy(voxels).cuda()

model.eval()

recon, skip = model.forward(data=voxels)

recon = torch.exp(recon) - 1.0

print(recon.shape)

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