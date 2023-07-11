import numpy as np
import torch
from model.color_net import EncoderDecoder
from utils.class_map import CLASS_MAP
import trimesh
input_obj_path = "/data/SampleDataset/CompleteDataset/InputSamples/colorized_meshes/bag/036_00019/0_036_00019.obj"
target_obj_path = "/data/SampleDataset/CompleteDataset/GT/colorized_meshes/bag/036_00019.obj"

model = EncoderDecoder()
model = model.to(torch.double)

optimizer = torch.optim.Adam(model.parameters())

checkpoint = torch.load("./ckpts/colored/ckpt-epoch-001-.pth")

model.load_state_dict(checkpoint['model'])

optimizer.load_state_dict(checkpoint['optim'])

print("success loading")

mesh = trimesh.load(input_obj_path)
mesh.show()
vertices = np.asarray(mesh.vertices).T
colors = np.asarray(mesh.visual.vertex_colors).T[0:3]
colors = colors[np.newaxis, ...]
colors = torch.from_numpy(colors).to(torch.double)
print("###incomplete color")
print(colors.shape)
print(colors)
model.eval()
predicted_color = model(colors)
predicted_color = predicted_color.detach().numpy()
print("###predicted color")
print(predicted_color)
print(predicted_color.shape)
mesh_target = trimesh.load(target_obj_path)
colors_target = np.asarray(mesh_target.visual.vertex_colors).T[0:3]
colors_target = colors_target[np.newaxis, ...]
colors_target = torch.from_numpy(colors_target).to(torch.double)
print("###target color")
print(colors_target)
predicted_color = predicted_color.squeeze()
mesh.visual.vertex_colors = predicted_color.T.astype(np.uint8)
mesh.show()

