# ML3DG_Project
Reimplementation of [GRNet](https://github.com/hzxie/GRNet) with a few modifications as part of a University Project.

![alt text](https://github.com/rauldds/ML3DG_Project/blob/main/imgs/structure_grnet.png)

## First Steps:
When you clone this repository do it recursively, to make sure that the manifold package in additional_repos is also getting installed.

After cloning the repository, from root go to /additional_repos/Manifold and execute the commands below:
```sh
cd Manifold
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
Once done that, return to root, go to /model/extensions and execute the following commands:
```sh
# Chamfer Distance
cd chamfer_dist
python3 setup.py install --user
cd ..

# Cubic Feature Sampling
cd cubic_feature_sampling
python3 setup.py install --user
cd ..

# Gridding & Gridding Reverse
cd gridding
python3 setup.py install --user
cd ..

# Gridding Loss
cd gridding_loss
python3 setup.py install --user
cd ..
```

## Examples
In /examples you can find a few scripts that show how we generate incomplete versions of point clouds (incomplete_cloud_generator.py), how we convert point clouds to SDFs (), and how we recolorize a mesh generated using marching cubes based on a poisson reconstructed mesh (). 

## Dataset 
To train the model we use TSDF representations of [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/). 
To generate our version of the dataset simply follow the next steps: 
1. _Generate a .txt file containing all the samples paths in the dataset obtained as described above:_ 
```sh
# USAGE
python3 utils/paths_txt_file_generator.py -d "PATH/TO/DOWNLOADED/DATASET"
# This should generate a .txt file containing samples as the following examples:
/media/davidg-dl/Second SSD/object_dataset/table/scene0646_00_00017.bin
/media/davidg-dl/Second SSD/object_dataset/table/252_00004.bin
...
``` 
2. _Dataset Generation_: for training, validation, and testing we generate **_two_**  types of files; SDFs, and objs. Its important to mention that for our implementation we generated 6 _"incomplete views"_ for each of the samples in the dataset to use as Input of our Model. That is for each scan in our Dataset we should have [0-5] incomplete views as well as  [0-5] obj colorized meshes. Similar preprocessing applies for the Ground Truth (GT), for which we also generate one SDF and one obj per scan to generate all this files we provide a scrip that handles the preprocessing of our data with the following steps:
```sh
# a) Generating the Ground Truth (GT)

# IMPORTANT  for a) and b)  we need to provide the output path 
# where the preprocesed dataset will be stored in.
config/dataset_generation_GT_config.py 
  OUTPUT_PATH = "/path/to/output/directory"

# USAGE
python3 -m src.generate_SDFs_and_colorized_meshes_dataset
# To avoid errors make sure the "generate_sdf_npz_and_colorized_mesh" 
# function inside the provided script has the following parameters:
  generate_sdf_npz_and_colorized_mesh(sample,
                                      COLORIZE=True,
                                      GENERATE_TARGET=True)
# For this scrip its only necessary to provide  the path to the previously 
# generated .txt file containing the path to the samples.
  PATH_TO_SAMPLES_TXT = 'paths_to_point_clouds.txt'
  
 # b) Generating the 6 incomplete SDF incomplete input views 
 # and obj colored meshes
 
 # USAGE
 python3 -m src.generate_incomplete_SDFs_input_views
 # In this scrip we han manually select how many 
 # views we want to generate with:
  VIEWS_TO_GENERATE = 1
# We recommend generating one view individually, the re-run the scrip 
# multiple times to generate the remaining views as this process 
# can take up to one week depending on the computer used 
``` 
3. _Generating an auxiliary .txt for mapping all the incomplete views to its corresponding scan, and class as well as easy access to data in the **Dataset Class**:_  
```sh
# USAGE
python3 utils.generate_samples_txt.py
# 1. Specify the paths to the GT and InputData directories in the 'gt_path' and 'input_data_path' variables, respectively.
  gt_path = Path('/media/davidg-dl/Second SSD/CompleteDataset/GT/SDFs')
  input_data_path = Path('/media/davidg-dl/Second SSD/CompleteDataset/InputData/SDFs')
# 2. Set the desired path and name of the output file in the 'output_file_path' variable; Ex:"output-final.txt".
  output_file_path = Path("output-final.txt")
# 3. Run the script to generate the output file with view-sample-class triplets.

# Output example:
0_scene0646_00_00017 scene0646_00_00017 table
1_scene0646_00_00017 scene0646_00_00017 table
...
``` 
4. _Finally, Generate .txt files for the Dataset splits: **[train, val, test]**, and also an overfit file. This last part randomly samples elements in the auxiliary .txt file and allows to specify a Dataset split portion._

```sh
# USAGE
python3 -m utils.generate_data_splits
# Specify the paths to GT and InputData directories
# gt_path = Path('/media/davidg-dl/Second SSD/DATASET_test/GT/SDFs')
  gt_path = Path('/media/davidg-dl/Second SSD/CompleteDataset/GT/SDFs')
  input_data_path = Path('/media/davidg-dl/Second SSD/CompleteDataset/InputData/SDFs')

# Dataset split example 
  train_split = 0.7  # 70% for training
  val_split = 0.20  # 20% for validation
  test_split = 0.10  # 10% for testing
```
5. **Once all those steps are done we are ready to test our overfit split or train the model**


## Training

### Completion and Classification
To start training the completion part of the model simply run:
```
python3 -m src.train_runner -tr completion
```

To start training the classification part of the model simply run:
```
python3 -m src.train_runner -tr classification
```
To start training the whole model simply run:
```
python3 -m src.train_runner -tr all
```

### Color completion
To start training the color net of the model simply run:
```sh
python3 -m model.train_runner -dp <DATASET PATH>
```

Notes:
- In the dataloader modify the split that is being used accordingly.
- Create a folder /{PATH_TO_ML3DG_PROJECT}/ckpts/colored to store the checkpoints.
- Configuration can be set in /{PATH_TO_ML3DG_PROJECT}/config/color_net_config.py

## Inference
Once the completion model is trained, you can visualize how well it performs by simply running: 
```
python3 -m inference.object_completion
```
Once the classification model is trained, you can check how well it performs by simply running: 
```
python3 -m inference.object_classification
```

Once the colorization model is trained, you can check how well it performs by simply running: 
```
python3 -m inference.completion_and_classification

```
Notes: You have to modify the path of the dataset in this script. Probably would be a good idea to pass the dataset as an argument.



## Acknowledgements
- GRNET
- Manifold
