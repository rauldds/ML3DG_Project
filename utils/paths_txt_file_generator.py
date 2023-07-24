import os
import argparse

# this script generates the paths file (containes paths to all the point clouds), 
# just need to provide the path to the dataset

def txt_file_generator(dataset_path):
    paths = []
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if "indices.bin" in file_name:
                pass
            elif "part.bin" in file_name:
                pass
            elif ".xml" in file_name:
                pass
            elif "txt" in file_name:
                pass
            elif "_part_bug.bin" in file_name:
                pass
            else:
                if ".bin" in file_name:
                    file_path = os.path.join(root, file_name)
                    paths.append(file_path)
                    # print("\nFile path:", file_path)
    with open("paths_to_point_clouds.txt", "w") as f:
        for path in paths:
            f.write("%s\n" % path)

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--dataset", help="dataset path", type=str)
    args = argParser.parse_args()
    pc_path = args.dataset

    if pc_path !=None:
        txt_file_generator(pc_path)
    else:
        print("No dataset path provided")