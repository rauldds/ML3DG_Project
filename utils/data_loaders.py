import json
import logging
import numpy as np
from pathlib import Path
import torch.utils.data.dataset
from enum import Enum, unique
from tqdm import tqdm

@unique
class DatasetSubset(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2

class ScanObjectNNDataset(torch.utils.data.dataset.Dataset):
    DATASET_PATH = '/media/davidg-dl/Second SSD/DATASET_test/'

    #TODO: how many of the original implementations do we need? need to figure it out when testing in the train loop
    def __init__(self, split, options, file_list, transforms=None):
        assert split in ['train', 'val', 'overfit']

        # Read the lines from the split or overfit file and separate view, sample, and class elements
        with open(f"{split}.txt", "r") as file:
            lines = [line.strip().split() for line in file]

        # Unpack the elements into separate variables using list comprehension
        self.items = [(view, sample, class_name) for view, sample, class_name in lines]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        view_id, shape_id, class_name = self.items[index]
        # TODO find a way to correctly load the input SDF and all the views from the sample name

        # Get the target SDF
        target_sdf = ScanObjectNNDataset.get_target_sdf(dataset_path=ScanObjectNNDataset.DATASET_PATH,
                                                        class_name=class_name,
                                                        shape_id=shape_id)

        incomplete_view = ScanObjectNNDataset.get_incomplete_view(dataset_path=ScanObjectNNDataset.DATASET_PATH,
                                                                  class_name=class_name,
                                                                  shape_id=shape_id,
                                                                  view_id=view_id)

        return {
            "target_sdf": target_sdf,
            "incomplete_view": incomplete_view,
            "class": class_name
        }

    @staticmethod
    def get_target_sdf(dataset_path, class_name, shape_id):
        # Loading the SDFs stored as npz
        data = np.load(dataset_path+ 'GT/SDFs/' + class_name + f"/{shape_id}.bin.npz")
        target_sdf = data["arr_0"]
        target_sdf = target_sdf[np.newaxis, ...]
        return target_sdf

    @staticmethod
    def get_incomplete_view(dataset_path, class_name, shape_id, view_id):
        # Load the corresponding incomplete view for the current shape_id
        path = (dataset_path + "InputData/SDFs/" + class_name + f"/{shape_id}/" + f"{view_id}.npz")
        data = np.load(path)
        incomplete_view = data["arr_0"]
        incomplete_view = incomplete_view[np.newaxis,  ...]
        return incomplete_view

    # TODO: Check the truncation function, and conclude the input shape to the NET
    def truncate_sdf(sdf):
        # Apply truncation to SDF to be +/- 3
        input_sdf = np.clip(sdf, a_min=-3.0, a_max=3.0)
        sdf_signs = np.sign(input_sdf)
        input_sdf = np.absolute(input_sdf)
        concatenated_array = np.concatenate((input_sdf, sdf_signs))
        print(f"The truncated array shape is: {concatenated_array.shape}print @ data_loaders.py")
        return concatenated_array

# Usage example in case modifications are needed

# output: # Target SDFs shape: torch.Size([4, 1, 64, 64, 64])
# Incomplete views shape: torch.Size([4, 1, 64, 64, 64])
# Classes ['table', 'table', 'table', 'table']
# Target SDFs shape: torch.Size([4, 1, 64, 64, 64])

# from torch.utils.data import DataLoader
# dataset = ScanObjectNNDataset(split='overfit', options=None, file_list=None, transforms=None)
# test_sample = dataset[10]
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#
# for batch in dataloader:
#     # Extract the batch elements
#     target_sdfs = batch['target_sdf']
#     incomplete_views = batch['incomplete_view']
#     classes = batch['class']
#
#     # Print the shapes of the batched tensors as an example
#     print(f"Target SDFs shape: {target_sdfs.shape}")
#     print(f"Incomplete views shape: {incomplete_views.shape}")
#     print(f"Classes {classes}")