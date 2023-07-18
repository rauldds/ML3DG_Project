# ML3DG_Project
Reimplementation of [GRNet](https://github.com/hzxie/GRNet) with a few modifications as part of a University Project

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
To train the model we use TSDF representations of [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/). To generate our version of the dataset simply follow the next steps: (DAVID DESCRIBES HERE ALL THE STEPS TO GENERATE THE DATASET)

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

Notes: 
- In the dataloader (line 19) you have to modify the dataset path with its location. PROBABLY WOULD BE A GOOD IDEA TO PASS THE DATASET PATH AS AN ARGUMENT FOR THE DATALOADER.
- If you want to also resume from a checkpoint simply add the flag `--resume True` when executing any of the commands above.
- If you want to try overfitting/train completion with shapenet the `shapenet.py` script in data_e3 has to be updated with complete functions. Additionally, the dataset sign and distance value doesn't have to be separated and probably we wouldn't have to use the LOG SCALING. Once that is done, training completion with shapenet should be possible with `--dataset Shapenet`
- SHOULD WE ALSO DO THE LOG SCALING AS IN EXERCISE 3?

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
python3 -m inference.object_colorization
```
Notes: You have to modify the path of the dataset in this script. Probably would be a good idea to pass the dataset as an argument.



## Acknowledgements
- GRNET
- Manifold