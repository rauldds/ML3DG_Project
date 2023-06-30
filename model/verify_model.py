from pathlib import Path
from statistics import mode
import numpy as np
import torch
from grnet import GRNet
import skimage
import trimesh
import open3d as o3d

model = GRNet()
print(GRNet())
model = model.cuda()

'''optimizer = torch.optim.Adam(model.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(Path)

model.load_state_dict(checkpoint['model_state_dict'])

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

x = torch.ones((1,1,64,64,64))
x = x.cuda()
#print(x.is_cuda)

model.eval()

recon, class_pred = model.forward(data=x)

print(recon.shape)
print(class_pred.shape)

vertices, faces, normals, _ = skimage.measure.marching_cubes(recon, level=0)

mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

mesh = mesh.as_open3d

o3d.visualization.draw_geometries([mesh],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[0, 0, 0],
                                  up=[0.2304, -0.8825, 0.4101])'''