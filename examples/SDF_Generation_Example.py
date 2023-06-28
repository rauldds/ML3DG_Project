import argparse
import open3d as o3d
import numpy as np
import subprocess
import trimesh
import skimage
from mesh_to_sdf import mesh_to_voxels as mtv
from utils.SDF_Visualizer import visualize_sdf
from utils.recolorize_mesh import colorizing

argParser = argparse.ArgumentParser()
argParser.add_argument("-pc", "--point_cloud", help="point cloud path", type=str)
args = argParser.parse_args()
print("Point Cloud Path: %s" % args.point_cloud)

pc_path = args.point_cloud

#DEFINITION OF RELATIVE PATHS
mesh_input_pointcloud_path = "./results/mesh_input_cloud.obj"
watertight_mesh_path = "./results/mesh_input_cloud_watertight.obj"
sdf_visualization_path = "./results/sdf_visualization.ply"
recons_path ="./results/mesh_input_cloud_recon_marching_cubes.obj"

#Extraction of point cloud data from binary file
scan = np.fromfile(pc_path, dtype=np.float32)
scan = scan[1:]
scan = scan.reshape((-1,11))

#Conversion of point cloud to o3d format
pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(scan[:,0:3])
pcl.normals = o3d.utility.Vector3dVector(scan[:,3:6])
pcl.colors = o3d.utility.Vector3dVector(scan[:,6:9].astype(np.uint8)/255)

#Implementation of Poisson Surface Reconstruction to create mesh from cloud
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcl, depth=9)
print("Creating Mesh from Input Point Cloud")
print(f'Exported to: {mesh_input_pointcloud_path}')
o3d.io.write_triangle_mesh(mesh_input_pointcloud_path, mesh)

#Visualization of the generated mesh
o3d.visualization.draw_geometries([mesh],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[0, 0, 0],
                                  up=[0.2304, -0.8825, 0.4101])

# Conversion of non-watertight mesh to watertight (0 holes)
# Used scripts from "Robust Watertight Manifold Surface Generation Method for ShapeNet Models"
# Link: https://arxiv.org/pdf/1802.01698.pdf
process = subprocess.Popen(['./additional_repos/Manifold/build/manifold', mesh_input_pointcloud_path, watertight_mesh_path], 
                           stdout=subprocess.PIPE,
                           universal_newlines=True)
while True:
    #output = process.stdout.readline()
    #print(output.strip())
    # Do something else
    return_code = process.poll()
    if return_code is not None:
        print('RETURN CODE', return_code)
        break

#COLORIZING MANIFOLD MESH
mesh_manifold = trimesh.load(watertight_mesh_path)
mesh = colorizing(trimesh.load(mesh_input_pointcloud_path),mesh_manifold)
print("Creating Colored Watertight Mesh")
print(f'Exported to: {watertight_mesh_path}')
trimesh.exchange.export.export_mesh(mesh,watertight_mesh_path,"obj")

# Generation of SDF given the watertight mesh
voxels = mtv(mesh, 64, pad=False)
visualize_sdf(voxels,sdf_visualization_path)

# Implementation of marching cubes to visualize reconstruction based on SDF
print("Creating mesh reconstruction from SDF")
print(f'Exported to: {recons_path}')
vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
trimesh.exchange.export.export_mesh(mesh,recons_path,"obj")