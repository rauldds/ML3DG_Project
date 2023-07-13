from pathlib import Path
import numpy as np
import torch
from model.grnet import GRNet_comp
import skimage
import trimesh
import open3d as o3d

# Define the path to the directory containing all the samples
samples_directory = '/media/davidg-dl/Second SSD/CompleteDataset/InputSamples/SDFs/'

# Define the path to the text file containing the sample information
sample_info_file = './utils/val.txt'

# Specify the number of samples you want to visualize
num_samples_to_visualize = 5

# Load the sample information from the text file
sample_info = np.genfromtxt(sample_info_file, dtype=str)

# Create a list to store the paths of the samples
sample_paths = []

# Iterate through the sample information and construct the sample paths
for i in range(num_samples_to_visualize):
    _, sample_name, class_name = sample_info[i]
    class_name = class_name.lower()
    sample_path = Path(samples_directory) / class_name / sample_name / f'2_{sample_name}.npz'
    sample_paths.append(sample_path)

print(sample_paths)
# Instantiate the GRNet model and move it to the GPU
model = GRNet_comp()
model = model.cuda()

# # Load the model checkpoint
optimizer = torch.optim.Adam(model.parameters())
checkpoint = torch.load("./ckpts/ScanObjectNN/ckpt-best-completion.pth")
model.load_state_dict(checkpoint['model_comp'])
optimizer.load_state_dict(checkpoint['cmp_optim'])
print("success loading")


# # Set the model to evaluation mode
# model.eval()
#
# Loop through the sample paths and visualize each sample
for sample_path in sample_paths:
    print(f"[INFO] Sample: {sample_path}")
    # Load the SDF data from the sample
    data = np.load(sample_path, allow_pickle=True)
    voxels = data["arr_0"] * 32
    voxels = np.clip(voxels, -3, 3).reshape((1, 1, 64, 64, 64))
    print(voxels.shape)
    voxels = torch.from_numpy(voxels).cuda()
    model.eval()

#
#     # Perform the inference
    with torch.no_grad():
        recon, skip = model.forward(data=voxels)
        recon = torch.exp(recon) - 1.0
        recon = recon.detach().cpu().numpy()
        recon = recon.reshape((64, 64, 64))
        print(recon.shape)
#
#     # Extract the mesh from the reconstructed volume
    vertices, faces, normals, _ = skimage.measure.marching_cubes(recon, level=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
#
#     # Convert the mesh to Open3D CUDA mesh
    mesh = mesh.as_open3d
#
    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh],
                                      zoom=0.664,
                                      front=[-0.4761, -0.4698, -0.7434],
                                      lookat=[0, 0, 0],
                                      up=[0.2304, -0.8825, 0.4101])


