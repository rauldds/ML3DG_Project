from unittest import skip
import numpy as np
import torch
from model.grnet import GRNet_clas
from utils.class_map import CLASS_MAP

input_sdf_path = "/media/rauldds/TOSHIBA EXT/ML3G/Full_Project_Dataset/GT/SDFs/desk/005_00005.bin.npz"


model = GRNet_clas()
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters())

checkpoint = torch.load("./ckpts/ScanObjectNN/ckpt-best-classification.pth")

model.load_state_dict(checkpoint['model_clas'])

optimizer.load_state_dict(checkpoint['cls_optim'])

print("success loading")

data = np.load(input_sdf_path, allow_pickle=True)

voxels = data["arr_0"]*32

voxels = np.clip(voxels,-3,3).reshape((1, 64, 64, 64))

print(voxels.shape)

voxels = torch.from_numpy(voxels).cuda()

model.eval()

skip = {
            '32_r': torch.ones([1, 32, 32, 32, 32],device="cuda:0"),
            '16_r': torch.ones([1, 64, 16, 16, 16],device="cuda:0"),
            '8_r': torch.ones([1, 128, 8, 8, 8],device="cuda:0")
        }

cls = model.forward(data=voxels,skip=skip)
sftmx = torch.nn.Softmax(0)
cls = sftmx(cls)

print(cls.shape)

cls = cls.detach().cpu().numpy()
cls = np.argmax(cls)

print(cls)

for key, value in CLASS_MAP.items():
    if value == cls:
        print(f'detected class: {key}')
        break
