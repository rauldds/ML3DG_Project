from utils.SDF_and_color_mesh_generator import generate_sdf_npz_and_colorized_mesh

PATH_TO_SAMPLES_TXT = '/home/davidg-dl/Desktop/ML3DG_Project-main/Utils/experiment_dataset.txt'
VIEWS_TO_GENERATE = 1

def generate_individual_npz_and_colorized_mesh(view_number: int):
    sample_paths = open(PATH_TO_SAMPLES_TXT).read().splitlines()
    for i, sample in enumerate(sample_paths):
        generate_sdf_npz_and_colorized_mesh(sample,
                                            COLORIZE=False,
                                            GENERATE_TARGET=False,
                                            view_number=view_number)
        print(
            f"\n[INFO] Process completed for sample {i + 1},\n[INFO] Remaining Samples: {len(sample_paths) - (i + 1)}")
        print(
            "-------------------------------------------------------------------------------------------------------------")


for i, view in enumerate(range(VIEWS_TO_GENERATE)):
    generate_individual_npz_and_colorized_mesh(view_number=i)
    print(
        f"\n[INFO] Process completed for view {i + 1},\n[INFO] Remaining views: {VIEWS_TO_GENERATE - (i + 1)}\n")
    print(
        "-------------------------------------------------------------------------------------------------------------")