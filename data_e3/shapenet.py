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

        self.items = Path(f"./data_e3/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        sdf_id, df_id = self.items[index].split(' ')

        input_sdf = ShapeNet.get_shape_sdf(sdf_id)
        target_df = ShapeNet.get_shape_df(df_id)

        # TODO Apply truncation to sdf and df
        # TODO Stack (distances, sdf sign) for the input sdf
        # TODO Log-scale target df
        input_sdf = np.clip(input_sdf, -3, 3)
        target_df = np.clip(target_df, -3, 3)
        #input_sdf = input_sdf.astype(np.float32)
        #target_df = target_df.astype(np.float32)


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
        batch['incomplete_view'] = batch['incomplete_view'].to(device)
        batch['target_sdf'] = batch['target_sdf'].to(device)

    @staticmethod
    def get_shape_sdf(shapenet_id):
        shapenet_path = ShapeNet.dataset_sdf_path / (shapenet_id + '.sdf')
        dimX, dimY, dimZ = np.fromfile(shapenet_path, dtype=np.uint64, count=3)
        with open(shapenet_path, 'rb') as f:
            f.seek(24)
            sdf = np.fromfile(f, dtype=np.float32, count=-1)
        sdf = sdf.reshape((dimX, dimY, dimZ))
        # TODO implement df data loading
        return sdf

    @staticmethod
    def get_shape_df(shapenet_id):
        shapenet_path = ShapeNet.dataset_df_path / (shapenet_id + '.df')
        dimX, dimY, dimZ = np.fromfile(shapenet_path, dtype=np.uint64, count=3)
        with open(shapenet_path, 'rb') as f:
            f.seek(24)
            df = np.fromfile(f, dtype=np.float32, count=-1)
        df = df.reshape((dimX, dimY, dimZ))
        # TODO implement df data loading
        return df
