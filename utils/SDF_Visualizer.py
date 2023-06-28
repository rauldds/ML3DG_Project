import numpy as np
import trimesh
from pathlib import Path
from matplotlib import cm, colors

def visualize_sdf(sdf: np.array, filename: Path) -> None:
    assert sdf.shape[0] == sdf.shape[1] == sdf.shape[2], "SDF grid has to be of cubic shape"
    print(f"Creating SDF visualization for {sdf.shape[0]}^3 grid ...")

    voxels = np.stack(np.meshgrid(range(sdf.shape[0]), range(sdf.shape[1]), range(sdf.shape[2]))).reshape(3, -1).T

    sdf[sdf < 0] /= np.abs(sdf[sdf < 0]).max() if np.sum(sdf < 0) > 0 else 1.
    sdf[sdf > 0] /= sdf[sdf > 0].max() if np.sum(sdf < 0) > 0 else 1.
    sdf /= -2.

    corners = np.array([
        [-.25, -.25, -.25],
        [.25, -.25, -.25],
        [-.25, .25, -.25],
        [.25, .25, -.25],
        [-.25, -.25, .25],
        [.25, -.25, .25],
        [-.25, .25, .25],
        [.25, .25, .25]
    ])[np.newaxis, :].repeat(voxels.shape[0], axis=0).reshape(-1, 3)

    scale_factors = sdf[tuple(voxels.T)].repeat(8, axis=0)
    cube_vertices = voxels.repeat(8, axis=0) + corners * scale_factors[:, np.newaxis]
    cube_vertex_colors = cm.get_cmap('seismic')(colors.Normalize(vmin=-1, vmax=1)(scale_factors))[:, :3]

    faces = np.array([
        [1, 0, 2], [2, 3, 1], [5, 1, 3], [3, 7, 5], [4, 5, 7], [7, 6, 4],
        [0, 4, 6], [6, 2, 0], [3, 2, 6], [6, 7, 3], [5, 4, 0], [0, 1, 5]
    ])[np.newaxis, :].repeat(voxels.shape[0], axis=0).reshape(-1, 3)
    cube_faces = faces + (np.arange(0, voxels.shape[0]) * 8)[np.newaxis, :].repeat(12, axis=0).T.flatten()[:, np.newaxis]

    mesh = trimesh.Trimesh(vertices=cube_vertices, faces=cube_faces, vertex_colors=cube_vertex_colors, process=False)
    mesh.export(str(filename))
    print(f"Exported to {filename}")
