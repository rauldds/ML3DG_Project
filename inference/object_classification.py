from pathlib import Path
import random
import numpy as np
import torch
from model.grnet import GRNet_clas
from utils.class_map import CLASS_MAP

samples_directory = '/media/rauldds/TOSHIBA EXT/ML3G/Full_Project_Dataset/GT/SDFs/'

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
    sample_path = Path(samples_directory) / class_name / f'{sample_name}.bin.npz'
    sample_paths.append(sample_path)
    class_names.append(class_name)


model = GRNet_clas()
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters())

checkpoint = torch.load("./ckpts/ScanObjectNN/ckpt-best-classification.pth")

model.load_state_dict(checkpoint['model_clas'])

optimizer.load_state_dict(checkpoint['cls_optim'])

print("success loading")


skip = {
            '32_r': torch.ones([1, 32, 32, 32, 32],device="cuda:0"),
            '16_r': torch.ones([1, 64, 16, 16, 16],device="cuda:0"),
            '8_r': torch.ones([1, 128, 8, 8, 8],device="cuda:0")
        }

# Loop through the sample paths and visualize each sample
for count, sample_path in enumerate(sample_paths):
    # Load the SDF data from the sample
    data = np.load(sample_path, allow_pickle=True)
    voxels = data["arr_0"] * 32
    voxels = np.clip(voxels, -3, 3).reshape((1, 1, 64, 64, 64))
    voxels = torch.from_numpy(voxels).cuda()
    model.eval()

    # Perform the inference
    cls = model.forward(data=voxels,skip=skip)
    cls = cls.detach().cpu().numpy()
    cls = np.argmax(cls)
    for key, value in CLASS_MAP.items():
        if value == cls:
            print(f'GT: {class_names[count]}')
            print(f'detected class: {key}')
            break
