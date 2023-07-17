import random
from pathlib import Path
import numpy as np
import torch
from model.grnet import GRNet_clas, GRNet_comp
import skimage
import trimesh
import open3d as o3d
from utils.class_map import CLASS_MAP

# Define the path to the directory containing all the samples
samples_directory = '/media/davidg-dl/Second SSD/CompleteDataset/InputSamples/SDFs/'

# Define the path to the text file containing the sample information
sample_info_file = './utils/test.txt'

# Specify the number of samples you want to visualize
num_samples_to_visualize = 10

# Load the sample information from the text file
sample_info = np.genfromtxt(sample_info_file, dtype=str)

# Randomly select sample indices without replacement
sample_indices = random.sample(range(len(sample_info)), num_samples_to_visualize)

# Create a list to store the paths of the samples
sample_paths = []
class_names = []

# Iterate through the randomly selected sample indices and construct the sample paths
for index in sample_indices:
    _, sample_name, class_name = sample_info[index]
    class_name = class_name.lower()
    sample_path = Path(samples_directory) / class_name / sample_name / f'2_{sample_name}.npz'
    sample_paths.append(sample_path)
    class_names.append(class_name)

# Instantiate the GRNet model and move it to the GPU
model_comp = GRNet_comp()
model_comp = model_comp.cuda()
model_cls = GRNet_clas()
model_cls = model_cls.cuda()


# Load the model checkpoint
optimizer_comp = torch.optim.Adam(model_comp.parameters())
optimizer_cls = torch.optim.Adam(model_cls.parameters())
checkpoint = torch.load("./ckpts/ScanObjectNN/ckpt-best-completion.pth")
model_comp.load_state_dict(checkpoint['model_comp'])
optimizer_comp.load_state_dict(checkpoint['cmp_optim'])
checkpoint = torch.load("./ckpts/ScanObjectNN/ckpt-best-classification.pth")
model_cls.load_state_dict(checkpoint['model_cls'])
optimizer_cls.load_state_dict(checkpoint['cls_optim'])
print("Success loading model checkpoint.")

# Loop through the sample paths and visualize each sample
for count, sample_path in enumerate(sample_paths):
    print(f"[INFO] Sample: {sample_path}")
    # Load the SDF data from the sample
    data = np.load(sample_path, allow_pickle=True)
    voxels = data["arr_0"] * 32
    voxels = np.clip(voxels, -3, 3).reshape((1, 1, 64, 64, 64))
    print(voxels.shape)
    voxels = torch.from_numpy(voxels).cuda()
    model_comp.eval()
    model_cls.eval()

    # Perform the inference
    with torch.no_grad():
        cls_recon, skip = model_comp.forward(data=voxels)
        cls_recon = torch.exp(cls_recon) - 1.0
        recon = cls_recon.detach().cpu().numpy()
        skip_detached = {key: value.detach() for key, value in skip.items()}
        cls = model_cls.forward(data=cls_recon.detach(),skip=skip_detached)
        cls = cls.detach().cpu().numpy()
        cls = np.argmax(cls)
        recon = recon.reshape((64, 64, 64))
        for key, value in CLASS_MAP.items():
            if value == cls:
                print(f'GT: {class_names[count]}')
                print(f'detected class: {key}')
                break

    # Extract the mesh from the reconstructed volume
    vertices, faces, normals, _ = skimage.measure.marching_cubes(recon, level=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

    # Convert the mesh to Open3D mesh
    mesh_o3d = mesh.as_open3d

    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh_o3d],
                                      zoom=0.664,
                                      front=[-0.4761, -0.4698, -0.7434],
                                      lookat=[0, 0, 0],
                                      up=[0.2304, -0.8825, 0.4101])