import enum
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import numpy as np
import trimesh

# Custom dataset class
class ColoredMeshesDataset(Dataset):
    def __init__(self, split, dataset_path):
        assert split in ['train', 'val', 'overfit']

        # Read the lines from the split or overfit file and separate view, sample, and class elements
        with open(f"./utils/{split}.txt", "r") as file:
            lines = [line.strip().split() for line in file]

        # Unpack the elements into separate variables using list comprehension
        self.items = [(view, sample, class_name) for view, sample, class_name in lines]
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        view_id, shape_id, class_name = self.items[idx]
        
        complete_mesh = ColoredMeshesDataset.get_target_mesh(dataset_path=self.dataset_path,
                                                        class_name=class_name,
                                                        shape_id=shape_id)

        incomplete_mesh = ColoredMeshesDataset.get_incomplete_mesh(dataset_path=self.dataset_path,
                                                                  class_name=class_name,
                                                                  shape_id=shape_id,
                                                                  view_id=view_id)
        return {
            "target_mesh": complete_mesh,
            "incomplete_mesh": incomplete_mesh
        }
    @staticmethod
    def send_data_to_device(batch_val, device):
        batch_val["target_mesh"] = batch_val["target_mesh"].to(device)
        batch_val["incomplete_mesh"] = batch_val["incomplete_mesh"].to(device)
    @staticmethod
    def get_target_mesh(dataset_path, class_name, shape_id):
        target_path = dataset_path+ '/GT/colorized_meshes/' + class_name + f"/{shape_id}.obj"
        mesh = trimesh.load(target_path)
        vertices = np.asarray(mesh.vertices).T
        colors = np.asarray(mesh.visual.vertex_colors).T[0:3]
        full_mesh = np.concatenate((vertices,colors[0:3]),axis=0)
        num_points = colors.shape[1]
        num_points = num_points - num_points % 16
        # Ensure that the number of points in the point clouds is a multiple of 16, and ensure that the shapes remain
        # consistent after passing through four layers of convolution and convolution transpose
        return colors[:, :num_points]

    @staticmethod
    def get_incomplete_mesh(dataset_path, class_name, shape_id, view_id):
        target_path = dataset_path+ '/GT/colorized_meshes/' + class_name + f"/{shape_id}.obj"
        incomplete_path = (dataset_path + "/InputSamples/colorized_meshes/" + class_name + f"/{shape_id}/" + f"{view_id}.obj")
        mesh_complete = trimesh.load(target_path)
        mesh_incomplete = trimesh.load(incomplete_path)
        mesh_incomplete.visual = mesh_incomplete.visual.to_texture()
        mesh_complete.visual = mesh_incomplete.visual.copy()
        v_idx = mesh_incomplete.kdtree.query(mesh_complete.vertices.copy())[1]
        mesh_complete.visual.uv = mesh_incomplete.visual.uv[v_idx]
        mesh_complete.visual = mesh_complete.visual.to_color()
        vertices = np.asarray(mesh_complete.vertices).T
        colors =np.asarray(mesh_complete.visual.vertex_colors).T[0:3]
        full_mesh = np.concatenate((vertices,colors[0:3]),axis=0)
        num_points = colors.shape[1]
        num_points = num_points - num_points % 16
        # Ensure that the number of points in the point clouds is a multiple of 16, and ensure that the shapes remain
        # consistent after passing through four layers of convolution and convolution transpose
        return colors[:, :num_points]

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x['target_mesh']), reverse=True)
    targets = [torch.from_numpy(item['target_mesh']) for item in batch]
    inputs = [torch.from_numpy(item['incomplete_mesh']) for item in batch]
    max_length = max(seq.shape[1] for seq in targets)
    target_padded_batch = []
    input_padded_batch = []
    for i, target in enumerate(targets):
        padding_length = max_length - target.shape[1]
        if padding_length>0:
            padded_target = torch.cat((target, torch.zeros([3,padding_length])),dim=1)
            padded_input = torch.cat((inputs[i], torch.zeros([3,padding_length])),dim=1)
            target_padded_batch.append(padded_target)
            input_padded_batch.append(padded_input)
        else:
            target_padded_batch.append(target)
            input_padded_batch.append(inputs[i])
        

    return {
            "target_mesh": torch.stack(target_padded_batch),
            "incomplete_mesh": torch.stack(input_padded_batch)
        }

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = ColoredMeshesDataset("overfit","/media/rauldds/TOSHIBA EXT/ML3G/Full_Project_Dataset")
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True,collate_fn=collate_fn)
    for batch in train_dataloader:
        print(len(batch['target_mesh']))
        for i, target in enumerate(batch["target_mesh"]):
            print(target.shape)
            print(batch['incomplete_mesh'][i].shape)
