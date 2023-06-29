import json
import logging
import numpy as np
# from torch.utils.data import DataLoader, Dataset
import torch.utils.data.dataset

from enum import Enum, unique
from tqdm import tqdm

@unique
class DatasetSubset(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2

# understand collate_fn, not clear yet how many of the original functions are suited for our implementation.

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms =transforms
        self.items = None
        self.path_to_sdf = ""
        self.sample_id = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        self.incomplete_views = []
        # TODO find a way to correctly load the input SDF and the all the views from the sample name
        target_sdf, input_incomplete_sdf = (None, None)

        # load SDF stored as npz preliminary version, still need to loop for all the samples having the
        # idea: sample names would be usefully for example to use it as a reference to loop over all the views.
        # input_sdf.shape = (64, 64, 64)

        data = np.load(self.path_to_sdf)
        input_sdf = data["arr_0"]

        # apply truncation to SDF to be +/- 3
        input_sdf = np.clip(input_sdf, a_min=-3.0, a_max=3.0)
        sdf_signs = np.sign(input_sdf)
        input_sdf = np.absolute(input_sdf)
        concatenated_array = np.concatenate((input_sdf, sdf_signs))

        return {
            "sample_name": f"{self.sample_id}",
            "input_sdf": input_sdf,
            "incomplete_views" : self.incomplete_views
        }

# import numpy as np
#
# array = np.load('/media/davidg-dl/Second SSD/ScanObjectNNDataset_SDF_Colorize_Meshes/SDFs/bag/005_00020.bin.npz')
# loaded_array = array["arr_0"]
# print(loaded_array.shape)

