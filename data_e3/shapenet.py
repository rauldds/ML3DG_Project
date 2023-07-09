from pathlib import Path
import json

import numpy as np
import torch


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8
    dataset_sdf_path = Path("./data_e3/shapenet_dim32_sdf")  # path to voxel data
    dataset_df_path = Path("./data_e3/shapenet_dim32_df")  # path to voxel data
    class_name_mapping = json.loads(Path("./data_e3/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.truncation_distance = 3

        self.items = Path(f"./data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        sdf_id, df_id = self.items[index].split(' ')

        input_sdf = ShapeNet.get_shape_sdf(sdf_id)
        target_df = ShapeNet.get_shape_df(df_id)

        # TODO Apply truncation to sdf and df
        # TODO Stack (distances, sdf sign) for the input sdf
        # TODO Log-scale target df

        #####################################################
        #     RESPECT THE KEY NAMES OF THE DICT BELOW       #
        #     ADDITIONALLY, IF CLASSIFICATION WILL BE       #
        #     TESTED, CLASS KEY AND VALUE SHOULD ALSO       #
        #                 BE RETRIEVABLE                    #
        #####################################################
        return {
            'name': f'{sdf_id}-{df_id}',
            'incomplete_view': input_sdf,
            'target_sdf': target_df
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        # TODO add code to move batch to device
        pass

    @staticmethod
    def get_shape_sdf(shapenet_id):
        sdf = None
        # TODO implement sdf data loading
        # WHAT THE NET WILL EXPECT: torch.Size([batch_size, 1, 32, 32, 32])
        return sdf

    @staticmethod
    def get_shape_df(shapenet_id):
        df = None
        # TODO implement df data loading
        return df
