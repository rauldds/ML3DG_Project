from utils.SDF_and_color_mesh_generator import generate_sdf_npz_and_colorized_mesh
from progress.bar import IncrementalBar as Bar

PATH_TO_SAMPLES_TXT = './utils/paths_to_point_clouds.txt'
VIEWS_TO_GENERATE = 1

def generate_individual_npz_and_colorized_mesh(view_number: int):
    sample_paths = open(PATH_TO_SAMPLES_TXT).read().splitlines()
    suffix = '%(index)d/%(max)d [elapsed: %(elapsed_td)s / eta: %(eta_td)s]'
    with Bar('Processed Clouds: ',max=len(sample_paths),suffix=suffix) as bar:
        for i, sample in enumerate(sample_paths):
            generate_sdf_npz_and_colorized_mesh(sample,
                                                COLORIZE=True,
                                                GENERATE_TARGET=False,
                                                view_number=view_number)
            bar.next()
            #print(
            #    f"\n[INFO] Process completed for sample {i + 1},\n[INFO] Remaining Samples: {len(sample_paths) - (i + 1)}")
            #print(
            #   "-------------------------------------------------------------------------------------------------------------")


for i, view in enumerate(range(VIEWS_TO_GENERATE)):
    # TODO: MODIFY VIEW NUMBER IF NECESSARY
    generate_individual_npz_and_colorized_mesh(view_number=5)
    print(
        f"\n[INFO] Process completed for view {i + 1},\n[INFO] Remaining views: {VIEWS_TO_GENERATE - (i + 1)}\n")
    print(
        "-------------------------------------------------------------------------------------------------------------")