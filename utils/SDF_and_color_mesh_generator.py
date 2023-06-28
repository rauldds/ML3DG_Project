import os
import open3d as o3d
import numpy as np
import subprocess
import trimesh
import skimage
from mesh_to_sdf import mesh_to_voxels as mtv

import config.dataset_generation_GT_config
from examples.recolorize_mesh import colorizing
import time

DEPTH = config.dataset_generation_GT_config.DEPTH
# TODO: Make sure that you have the txt file with the paths of all the point clouds.
#       paths_txt_file_generator.py

mesh_input_pointcloud_path = "/home/davidg-dl/Desktop/ML3DG_Project-main/results/mesh_input_cloud.obj"
watertight_mesh_path = "/home/davidg-dl/Desktop/ML3DG_Project-main/results/mesh_input_cloud_watertight.obj"
sdf_visualization_path = "/home/davidg-dl/Desktop/ML3DG_Project-main/results/sdf_visualization.ply"
recons_path ="/home/davidg-dl/Desktop/ML3DG_Project-main/results/mesh_input_cloud_recon_marching_cubes.obj"
example_npz_sdf = "/home/davidg-dl/Desktop/ML3DG_Project-main/results/example_npz_sdf.npz"


def generate_incomplete_view(scan):

    # Reading number of points in the cloud
    num_points = scan.shape[0]
    print(f"number of points in the cloud: {num_points}")

    # Sampling the points
    su = int(np.random.uniform(0, num_points, 1))
    # sn = int(np.random.randint(0,num_points))
    print(f"uniform sample from points: {su}")
    # print(f"normal sample: {sn}")

    # Probably would be a good idea to modify the elimination percentage to up to 75 percent
    elimination_percentage = np.random.random_sample() * 0.5 + 0.45
    print(f"percentage of the cloud that'll be eliminated: {elimination_percentage}")

    # Get coordinates of the sampled point
    center = (scan[su, :]).reshape((1, 3))

    # Eliminate points based on K-NN. Where K is the number of points to be eliminated
    # First obtain euclidean distance (other distance could also be used)
    distances = (scan - center) ** 2
    distances = distances.sum(1)
    distances = (distances ** (1 / 2)).reshape((num_points))

    # Calculate the number of points that'll be kept
    num_eliminations = int(elimination_percentage * num_points)
    kept_points = num_points - num_eliminations
    print(f"number of points to be kept: {kept_points}")

    # Obtain the indices of the points that'll be kept.
    indices = np.argpartition(distances, -kept_points)[-kept_points:]
    # incomplete_scan = scan[indices, :]
    # print(f"shape of incomple scan: {incomplete_scan.shape}")
    #
    # # Visualize incomplete scan
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(incomplete_scan)
    #
    # o3d.visualization.draw_geometries([pcd],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[0, 0, 0],
    #                                   up=[-0.0694, -0.9768, 0.2024])

    return indices


def colorize_sdf_mesh(sdf_generated_mesh,
                      class_name: str,
                      file_name: str,
                      view_number: int,
                      GENERATE_TARGET=True):

    # MESH OBTAINED AFTER APPLYING MANIFOLD
    original_size_mesh = trimesh.load(
        "/home/davidg-dl/Desktop/ML3DG_Project-main/results/mesh_input_cloud_watertight.obj")

    maxpc = ((np.max(original_size_mesh.vertices, 0)))
    minpc = ((np.min(original_size_mesh.vertices, 0)))

    # Mesh obtained after applying marching cubes to SDF
    # marching_cubes_mesh = trimesh.load(
    #     "/home/davidg-dl/Desktop/ML3DG_Project-main/results/mesh_input_cloud_recon_marching_cubes.obj")

    marching_cubes_mesh = sdf_generated_mesh

    # TRANSLATING and SCALING marching cubes mesh to orginal X,Y,Z space
    mesh_size = np.max(marching_cubes_mesh.vertices, 0) - np.min(marching_cubes_mesh.vertices, 0)
    scaling_factor = (maxpc - minpc) / mesh_size
    marching_cubes_mesh.vertices *= scaling_factor
    centering_factor = (np.min(marching_cubes_mesh.vertices, 0) - minpc)
    marching_cubes_mesh.vertices -= centering_factor
    # MESH DIRECTLY OBTAINED FROM POINT CLOUD
    colored_mesh = trimesh.load("/home/davidg-dl/Desktop/ML3DG_Project-main/results/mesh_input_cloud_watertight.obj")

    # TRANSFERING TEXTURE FROM ORIGINAL COLORED MESH TO SCALED/TRANSLATED MARCHING CUBES MESH
    # print(colored_mesh.visual.vertex_colors.shape)
    colored_mesh.visual = colored_mesh.visual.to_texture()
    marching_cubes_mesh.visual = colored_mesh.visual.copy()
    v_idx = colored_mesh.kdtree.query(marching_cubes_mesh.vertices.copy())[1]
    marching_cubes_mesh.visual.uv = colored_mesh.visual.uv[v_idx]
    marching_cubes_mesh.visual = marching_cubes_mesh.visual.to_color()
    # print(marching_cubes_mesh.visual.vertex_colors.shape)
    # trimesh.exchange.export.export_mesh(marching_cubes_mesh,
    #                                     "/home/davidg-dl/Desktop/ML3DG_Project-main/results/mesh_input_cloud_recon_marching_cubes_scaled.obj",
    #                                     "obj")

    # Returning marching cubes mesh to its original space
    marching_cubes_mesh.vertices += centering_factor
    marching_cubes_mesh.vertices = marching_cubes_mesh.vertices / scaling_factor

    # Writing colored marching cubes mesh
    # trimesh.exchange.export.export_mesh(marching_cubes_mesh,
    #                                     "/home/davidg-dl/Desktop/ML3DG_Project-main/results/mesh_input_cloud_recon_marching_cubes_colored.obj",
    #                                     "obj")

    if GENERATE_TARGET:
        # print(f"THIS IS THE SHORTED PATH FOR COLORIZATION: /media/davidg-dl/Second SSD/ScanObjectNNDataset_SDF/colorized_meshes/" + f"{class_name}")
        colorization_mesh_output_path = "/media/davidg-dl/Second SSD/ScanObjectNNDataset_SDF_Colorize_Meshes/colorized_meshes/" + f"{class_name}/"
        print(colorization_mesh_output_path)
        os.makedirs(colorization_mesh_output_path, exist_ok=True)
        trimesh.exchange.export.export_mesh(marching_cubes_mesh,
                                            "/media/davidg-dl/Second SSD/ScanObjectNNDataset_SDF_Colorize_Meshes/colorized_meshes/"
                                            + f"{class_name}/" + file_name +".obj",
                                            "obj")


def generate_sdf_npz_and_colorized_mesh(path_to_point_cloud: str,
                                        COLORIZE=True,
                                        GENERATE_TARGET=True,
                                        view_number=None,
                                        VISUALIZE=False):


    file_name = os.path.splitext(os.path.basename(path_to_point_cloud))[0]
    class_name = os.path.basename(os.path.dirname(path_to_point_cloud))
    start = time.time()
    # Extraction of point cloud data from binary file
    scan = np.fromfile(path_to_point_cloud, dtype=np.float32)
    scan = scan[1:]
    scan = scan.reshape((-1, 11))

    if not GENERATE_TARGET:
        scan_prep = scan[:, :3]
        scan_prep = scan_prep.astype(np.float64)
        indices = generate_incomplete_view(scan_prep)
        # Conversion of point cloud to o3d format
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(scan[indices, 0:3])
        pcl.normals = o3d.utility.Vector3dVector(scan[indices, 3:6])
        pcl.colors = o3d.utility.Vector3dVector(scan[indices, 6:9].astype(np.uint8) / 255)
    else:
        # Conversion of point cloud to o3d format
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(scan[:, 0:3])
        pcl.normals = o3d.utility.Vector3dVector(scan[:, 3:6])
        pcl.colors = o3d.utility.Vector3dVector(scan[:, 6:9].astype(np.uint8) / 255)


    # Implementation of Poisson Surface Reconstruction to create mesh from cloud
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcl, depth=DEPTH)
    print("\n[INFO] Creating Mesh from Input Point Cloud...")
    print(f'\n[INFO] Exported to: {mesh_input_pointcloud_path}...')
    o3d.io.write_triangle_mesh(mesh_input_pointcloud_path, mesh)
    end_from_point_cloud_to_od3mesh_time = time.time()
    print(
        f'\n[TIME-INFO] Time for reconstruction from point cloud to od3mesh '
        f'{end_from_point_cloud_to_od3mesh_time - start} seconds', )

    if VISUALIZE:
        # Visualization of the generated mesh
        o3d.visualization.draw_geometries([mesh],
                                          zoom=0.664,
                                          front=[-0.4761, -0.4698, -0.7434],
                                          lookat=[0, 0, 0],
                                          up=[0.2304, -0.8825, 0.4101])

    start_generation_of_waterlight_mesh = time.time()


    # This obj is generated by default @ ./mesh_input_cloud_watertight.obj and
    # needed for colorization of the mesh generated with the SDF
    # Conversion of non-watertight mesh to watertight (0 holes)
    # Used scripts from "Robust Watertight Manifold Surface Generation Method for ShapeNet Models"
    # Link: https://arxiv.org/pdf/1802.01698.pdf
    process = subprocess.Popen(['/home/davidg-dl/Desktop/ML3DG_Project-main/additional_repos/Manifold/build/manifold'
                                   , mesh_input_pointcloud_path, watertight_mesh_path],
                               stdout=subprocess.PIPE,
                               universal_newlines=True)
    end_generation_of_waterlight_mesh = time.time()
    print(
        f'\n[TIME-INFO] Time for waterlight mesh generation '
        f'{end_generation_of_waterlight_mesh - start_generation_of_waterlight_mesh} seconds')

    while True:
        # output = process.stdout.readline()
        # print(output.strip())
        # Do something else
        return_code = process.poll()
        if return_code is not None:
            print('\nRETURN CODE', return_code)
            break

    start_generation_colorized_manifold_mesh = time.time()

    # COLORIZING MANIFOLD MESH
    mesh_manifold = trimesh.load(watertight_mesh_path)
    mesh = colorizing(trimesh.load(mesh_input_pointcloud_path), mesh_manifold)
    print("\n[INFO] Creating Colored Watertight Mesh...")
    print(f'\n[INFO] Exported to: {watertight_mesh_path}...')
    trimesh.exchange.export.export_mesh(mesh, watertight_mesh_path, "obj")

    end_generation_colorized_manifold_mesh = time.time()
    print(f'\n[TIME-INFO] Time for colorizing the manifold mesh '
          f'{end_generation_colorized_manifold_mesh - start_generation_colorized_manifold_mesh} seconds')

    start_generation_sdf = time.time()

    # Generation of SDF given the watertight mesh
    voxels = mtv(mesh, 64, pad=False)
    print(f"\n[INFO] voxels-sdf shape {voxels.shape}")
    #visualize_sdf(voxels, sdf_visualization_path)

    if GENERATE_TARGET:
        print("\n[INFO] creating npz file for the sdf")
        output_path = path_to_point_cloud.replace("object_dataset", "ScanObjectNNDataset_SDF_Colorize_Meshes/SDFs")
        shorted_path = output_path.rsplit("/", 1)[0]
        print(f"THIS IS THE SHORTED PATH: {shorted_path}")
        os.makedirs(shorted_path, exist_ok=True)
        np.savez(output_path, voxels)

        end_generation_sdf = time.time()
        print(f'\n[TIME-INFO] Time for generating the sdf {end_generation_sdf - start_generation_sdf} seconds')

        # No need to store this mesh
        # Implementation of marching cubes to visualize reconstruction based on SDF
        print("\n[INFO] Creating mesh reconstruction from SDF...")
        print(f'\n[INFO] Exported to: {recons_path}...')
        vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
        sdf_generated_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        # trimesh.exchange.export.export_mesh(mesh, recons_path, "obj")
    else:
        print("\n[INFO] generating npz file for one of the uncompleted views")
        #/ media / davidg - dl / SecondSSD / object_dataset / table / scene0300_00_00003.bin

        # output_path = path_to_point_cloud.replace("object_dataset", f"InputSamples/class/{file_name}/" + f"{view_number}_{file_name}")
        output_path = "/media/davidg-dl/Second SSD/InputSamples/" + f"{class_name}/{file_name}/" + f"{view_number}_{file_name}"
        shorted_path  = "/media/davidg-dl/Second SSD/InputSamples/" + f"{class_name}/{file_name}/"
        os.makedirs(shorted_path, exist_ok=True)
        np.savez(output_path, voxels)

    # IMPORTANT: end of the generation of the SDF file of the point cloud and start of the colorization of its sdf
    # generated mesh in the previous step

    if COLORIZE:
        colorize_sdf_mesh(sdf_generated_mesh,
                          class_name=class_name,
                          file_name=file_name,
                          view_number=view_number)
