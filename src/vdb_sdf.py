from typing import Tuple

import numpy as np
import open3d as o3d
import pyopenvdb as vdb
import torch
import vdbfusion
from torch.utils.data.dataloader import default_collate
from vdb_to_numpy import LeafNodeGrid, vdb_to_triangle_mesh



# FIRST 4 FUNCTIONS OBTAINED FROM: 
# https://github.com/PRBonn/make_it_dense/blob/main/src/make_it_dense/utils/vdb_utils.py
def get_occ_percentage(nodes_a):
    '''Get the percentage of the grid that is occupied'''
    return np.count_nonzero((np.abs(nodes_a) < 1.0)) / nodes_a.ravel().shape[0]

def vdb_to_torch(grid, origin_xyz, shape: Tuple = None, empty_th: float = 0.0):
    '''Convert float grid to tensor'''
    shape = get_shape(grid.transform.voxelSize()[0]) if not shape else shape
    grid_origin = grid.transform.worldToIndexCellCentered(origin_xyz)
    x = np.empty(shape, dtype=np.float32)
    grid.copyToArray(x, grid_origin)
    if get_occ_percentage(x) <= empty_th:
        return None
    return {
        "nodes": torch.as_tensor(x).unsqueeze(0),
        "origin": torch.as_tensor(grid_origin, dtype=torch.int32),
    }

def torch_to_vdb_np(coords_ijk_t, leaf_nodes_t, voxel_size, sdf_trunc, normalize=True):
    """Convert torch arrays to VDB grids."""
    if normalize:
        leaf_nodes_t = np.float32(sdf_trunc) * leaf_nodes_t
    vdb_grid = vdb.FloatGrid()
    vdb_grid.background = np.float32(sdf_trunc)
    vdb_grid.gridClass = vdb.GridClass.LEVEL_SET
    vdb_grid.transform = vdb.createLinearTransform(voxelSize=voxel_size)
    coords_ijk_a = coords_ijk_t.detach().cpu().numpy()
    leaf_nodes_a = leaf_nodes_t.detach().cpu().numpy()
    # Network predicts bigger values than sdf_trunc due the scaled tanh
    tolerance = np.float32(leaf_nodes_a.max() - sdf_trunc)
    for i, ijk in enumerate(coords_ijk_a):
        #print((leaf_nodes_a[i]).shape)
        vdb_grid.copyFromArray(leaf_nodes_a[i], ijk, tolerance=tolerance)
    return vdb_grid

def get_shape(voxel_size: float, max_voxel_size: float = 0.4) -> Tuple:
    """Obtain volume shape from voxel size.

    These are some examples of the output of this function:

        voxel_size==0.4 -> (8,   8,  8)
        voxel_size==0.2 -> (16, 16, 16)
        voxel_size==0.1 -> (32, 32, 32)
    """
    nlog2 = round(np.log2(max_voxel_size / voxel_size))
    return 3 * (int(2 ** (nlog2 + 3)),)

# FUNCTION OBTAINED FROM: 
# https://github.com/PRBonn/make_it_dense/blob/main/src/make_it_dense/evaluation/scan_complete.py
def get_input_tensors(grid, coords_xyz,shape):
    """Conversion of TSDF Grids (pyopenvdb float grid) to Tensors"""
    inputs = []
    #print(grid)
    for origin_xyz in coords_xyz:
        input_dict = vdb_to_torch(grid, origin_xyz, shape=shape,empty_th=0.1)
        #print(input_dict)
        inputs.append(input_dict) if input_dict else None
    print(f"inputs len: {len(inputs)}")
    return default_collate(inputs)

# Path to input cloud
pc_path = "/media/rauldds/TOSHIBA EXT/ML3G/object_dataset/desk/005_00001.bin"
scan = np.fromfile(pc_path, dtype=np.float32)
# Removing the first value as it corresponds to the number of points that the cloud contains
scan = scan[1:]
# According to scan object cnn documentation each point contains 11 values
# https://github.com/hkust-vgd/scanobjectnn
scan = scan.reshape((-1,11))
# Only keep the position information
scan = scan[:,:3]
# Change format to the one needed by vdbfusion
scan = scan.astype(np.float64)
#Default the camera position. 
pose = np.eye(4)
#IMPORTANT: HAD TO DEFINE THIS SIZES BECAUSE LOWER RESOLUTIONS RESULT IN VERY POOR RECONSTRUCTIONS
#           Try modifying this values to check that out. Decreasing the voxel size to 0.01 
#           made my PC(Raul) crash.
#Definition of truncation distance in meters
sdf_truncation = .1
#Voxel size in meters (necessary to compute the grid)
tamano_vox=0.025

#Generating TSDF Grid
vdb_volume = vdbfusion.VDBVolume(tamano_vox,sdf_truncation)
vdb_volume.integrate(scan, pose)
# Generated TSDF
in_grid = vdb_volume.tsdf

# Reconstruction of TSDF to mesh
mesh = vdb_to_triangle_mesh(in_grid)
print(mesh)
# Writing file in ply format
o3d.io.write_triangle_mesh("cloud_recon_vdb.ply", mesh)

# TODO: This is still a hack, run tsdf with a higher resolution just to obtain the target
# coordinates values
#BEGIN: STILL NEED TO UNDERSTAND
base_volume = vdbfusion.VDBVolume(4*tamano_vox, 4*sdf_truncation)
base_volume.integrate(scan, pose)
target_coords_ijk_a, _ = LeafNodeGrid(base_volume.tsdf).numpy()
coords_xyz = base_volume.voxel_size * target_coords_ijk_a
#END: STILL NEED TO UNDERSTAND

# Returns the defined voxel size for the grid
voxel_size = in_grid.transform.voxelSize()[0]
# Returns the distance truncation value
sdf_trunc = in_grid.background
# Print grid size
print(f"grid shape: {get_shape(voxel_size)}")
# Conversion of Float Grids to tensors
# BEGIN: Still need to understand what the nodes and origins are.
#        It kinda looks like various grids with different origin (maybe it was just an
#        augmentation of make it dense)
inputs = get_input_tensors(in_grid, coords_xyz,get_shape(voxel_size))
nodes = inputs["nodes"]
origins = inputs["origin"]
print(f"tensors: {nodes.shape}")
print(f"origins: {origins.shape}")
# END: Still need to unders
# Conversion back to VDB Float Grid
x = torch_to_vdb_np(inputs["origin"], nodes.squeeze(1), voxel_size, sdf_trunc, normalize=True)
# Generation of mesh based on tensors
mesh = vdb_to_triangle_mesh(x)
o3d.io.write_triangle_mesh("cloud_recon_tensor.ply", mesh)