import numpy as np
import trimesh

# MESH OBTAINED AFTER APPLYING MANIFOLD
original_size_mesh = trimesh.load('./results/mesh_input_cloud_watertight.obj')

maxpc=((np.max(original_size_mesh.vertices,0)))
minpc=((np.min(original_size_mesh.vertices,0)))

# Mesh obtained after applying marching cubes to SDF
marching_cubes_mesh = trimesh.load('./results/mesh_input_cloud_recon_marching_cubes.obj')

# TRANSLATING and SCALING marching cubes mesh to orginal X,Y,Z space
mesh_size = np.max(marching_cubes_mesh.vertices,0)-np.min(marching_cubes_mesh.vertices,0)
scaling_factor = (maxpc-minpc)/mesh_size
marching_cubes_mesh.vertices *= scaling_factor
centering_factor = (np.min(marching_cubes_mesh.vertices,0)-minpc)
marching_cubes_mesh.vertices -= centering_factor
# MESH DIRECTLY OBTAINED FROM POINT CLOUD
colored_mesh = trimesh.load('./results/mesh_input_cloud_watertight.obj')

# TRANSFERING TEXTURE FROM ORIGINAL COLORED MESH TO SCALED/TRANSLATED MARCHING CUBES MESH
#print(colored_mesh.visual.vertex_colors.shape)
colored_mesh.visual = colored_mesh.visual.to_texture()
marching_cubes_mesh.visual = colored_mesh.visual.copy()
v_idx = colored_mesh.kdtree.query(marching_cubes_mesh.vertices.copy())[1]
marching_cubes_mesh.visual.uv = colored_mesh.visual.uv[v_idx]
marching_cubes_mesh.visual = marching_cubes_mesh.visual.to_color()
#print(marching_cubes_mesh.visual.vertex_colors.shape)
trimesh.exchange.export.export_mesh(marching_cubes_mesh,
                                    "./results/mesh_input_cloud_recon_marching_cubes_scaled.obj",
                                    "obj")

# Returning marching cubes mesh to its original space
marching_cubes_mesh.vertices += centering_factor
marching_cubes_mesh.vertices = marching_cubes_mesh.vertices / scaling_factor

# Writing colored marching cubes mesh
trimesh.exchange.export.export_mesh(marching_cubes_mesh,
                                    "./results/mesh_input_cloud_recon_marching_cubes_colored.obj",
                                    "obj")