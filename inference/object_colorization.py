import numpy as np
import torch
from model.color_net import EncoderDecoder
from utils.class_map import CLASS_MAP
import trimesh
input_obj_path = "/data/SampleDataset/CompleteDataset/InputSamples/colorized_meshes/bag/038_00032/2_038_00032.obj"
target_obj_path = "/data/SampleDataset/CompleteDataset/GT/colorized_meshes/bag/038_00032.obj"

model = EncoderDecoder()

model = model.to(torch.double)

checkpoint = torch.load("./ckpts/colored/ckpt-epoch-1500-only_skip_L1_loss_lr_0.0001.pth")

model.load_state_dict(checkpoint['model'])

print("success loading")

mesh_complete = trimesh.load(target_obj_path)

mesh_incomplete = trimesh.load(input_obj_path)

mesh_incomplete.visual = mesh_incomplete.visual.to_texture()

mesh_complete.visual = mesh_incomplete.visual.copy()

v_idx = mesh_incomplete.kdtree.query(mesh_complete.vertices.copy())[1]

mesh_complete.visual.uv = mesh_incomplete.visual.uv[v_idx]

mesh_complete.visual = mesh_complete.visual.to_color()

vertices = np.asarray(mesh_complete.vertices).T

colors =np.asarray(mesh_complete.visual.vertex_colors).T[0:3]

save_colors = colors

colors = colors[np.newaxis, ...]

colors = torch.from_numpy(colors).to(torch.double)

num_points = colors.shape[2]

colors = colors[:, :, :num_points - num_points % 16]

mesh_target = trimesh.load(target_obj_path)

mesh_target.show()

colors_target = np.asarray(mesh_target.visual.vertex_colors).T[0:3]

colors_target = colors_target[np.newaxis, ...]

colors_target = torch.from_numpy(colors_target).to(torch.double)

print("###target color")

print(colors_target)

print(colors_target.shape)

model.eval()

predicted_color = model(colors)

predicted_color = predicted_color.detach().numpy()

print("###predicted color")

print(predicted_color)

print(predicted_color.shape)

predicted_color = predicted_color.squeeze()

save_colors[:, :predicted_color.shape[1]] = predicted_color

mesh_target.visual.vertex_colors = save_colors.T.astype(np.uint8)

mesh_target.show()

