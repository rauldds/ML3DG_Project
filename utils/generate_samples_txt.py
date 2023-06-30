"""
Generate Output File with View-Sample-Class Triplets

This script generates an output file containing view-sample-class triplets based on the
directory structure of the GT (Ground Truth) and InputData directories. It iterates over
each class directory in GT, retrieves the sample files, and constructs the corresponding
scene directory path in InputData. It then retrieves the sorted list of view files in each scene
directory and writes the view-sample-class triplets to the output file.

Usage:
1. Specify the paths to the GT and InputData directories in the 'gt_path' and 'input_data_path' variables, respectively.
2. Set the desired path and name of the output file in the 'output_file_path' variable.
3. Run the script to generate the output file with view-sample-class triplets.

output example:
0_scene0646_00_00017 scene0646_00_00017 table
1_scene0646_00_00017 scene0646_00_00017 table
...

Note: Make sure the directory structures of the GT and InputData directories match
the expected format for correct results.

"""

from pathlib import Path

# Specify the paths to GT and InputData directories
gt_path = Path('/media/davidg-dl/Second SSD/DATASET_test/GT/SDFs')
input_data_path = Path("/media/davidg-dl/Second SSD/DATASET_test/InputData/SDFs")

# Open the output file in write mode
output_file_path = Path("output.txt")
with open(output_file_path, "w") as output_file:
    # Iterate over each class directory in GT
    for class_dir in gt_path.iterdir():
        # Get the class name
        class_name = class_dir.name

        # Print class name for debugging
        print(f"Processing class: {class_name}")

        # Iterate over each sample file in the class directory
        for sample_file in class_dir.glob("*.bin.npz"):
            # Get the sample name
            sample_name = sample_file.stem

            # Print sample name for debugging
            print(f"Processing sample: {sample_name}")

            # Construct the corresponding scene directory path in InputData
            scene_dir_path = input_data_path / class_name / sample_name.split(".")[0]

            # Print scene directory path for debugging
            print(f"Scene directory path: {scene_dir_path}")

            # Get the sorted list of view files in the scene directory
            view_files = sorted(scene_dir_path.glob("*.npz"))

            # Iterate over each sorted view file
            for view_file in view_files:
                # Get the view number
                view_number = view_file.stem.split("_")[0]

                # Write the view-sample-class triplet to the output file
                output_file.write(f"{view_number}_{sample_name.split('.')[0]} {sample_file.stem.split('.')[0]} {class_name}\n")

print("Processing completed.")
