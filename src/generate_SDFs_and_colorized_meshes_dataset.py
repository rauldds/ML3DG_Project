# USAGE python generate_SDFs_and_colorized_meshes_dataset.py
"""
IMPORTANT: requires a .txt file with the paths of all point clouds in the following structure:
/media/davidg-dl/Second SSD/object_dataset/pillow/scene0577_00_00016.bin
/media/davidg-dl/Second SSD/object_dataset/table/scene0646_00_00017.bin

"""

from utils.SDF_and_color_mesh_generator import generate_sdf_npz_and_colorized_mesh

PATH_TO_SAMPLES_TXT = '/home/davidg-dl/Desktop/ML3DG_Project-main/Utils/paths_to_point_clouds.txt'
sample_paths = open(PATH_TO_SAMPLES_TXT).read().splitlines()
sample_paths = sample_paths[606:]
for i, sample in enumerate(sample_paths):
    generate_sdf_npz_and_colorized_mesh(sample,
                                        COLORIZE=True,
                                        GENERATE_TARGET=True)
    print(f"\n[INFO] Process completed for sample {i+1},\n[INFO] Remaining Samples: {len(sample_paths) - (i+1)}")
    print("-------------------------------------------------------------------------------------------------------------")