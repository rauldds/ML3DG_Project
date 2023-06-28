import os

# this script generates the paths file (containes paths to all the point clouds), 
# just need to provide the path to the dataset


experiment_dataset_path = '/media/davidg-dl/Second SSD/object_dataset'
paths = []
for root, dirs, files in os.walk(experiment_dataset_path):
    for file_name in files:
        if "indices.bin" in file_name:
            pass
        elif "part.bin" in file_name:
            pass
        elif "part.xml" in file_name:
            pass
        elif "txt" in file_name:
            pass
        else:
            file_path = os.path.join(root, file_name)
            paths.append(file_path)
            # print("\nFile path:", file_path)
with open("paths_to_point_clouds.txt", "w") as f:
    for path in paths:
        f.write("%s\n" % path)
        f.write("%s\n" % path)
