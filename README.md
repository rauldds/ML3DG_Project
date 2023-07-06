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
To start training the completion part of the model simply run:
```
python3 -m src.train_runner -tr completion
```

To start training the classification part of the model simply run:
```
python3 -m src.train_runner -tr classification
```

Notes: In the dataloader (line 19) you have to modify the dataset path with its location. PROBABLY WOULD BE A GOOD IDEA TO PASS THE DATASET PATH AS AN ARGUMENT FOR THE DATALOADER


## Inference
Once the completion model is trained, you can visualize how well it performs by simply running: 
```
python3 -m inference.object_completion
```
Notes: You have to modify the path of the dataset in this script. Probably would be a good idea to pass the dataset as an argument.



## Acknowledgements
- GRNET
- Manifold