# Examples Scripts
- SDF_Generation_Example: Script that takes as input a point cloud, generates its corresponding SDF and then uses marching cubes to return to a mesh representation.
- resizing_colorizing_marching_cubes_example: Script that takes as input the colored watertight mesh and transfers its colors/texture to the mesh generated using marching cubes based on a SDF.
- incomplete_cloud_generator: Script to generate incomplete point clouds by selecting a random point from the cloud and then eliminating points around it using KNN.

# How to run

SDF_Generation_Example: from the root folder run:
```
python3 ./src/SDF_Generation_Example.py -pc  /PATH/TO/POINT_CLOUD.bin
```
resizing_colorizing_marching_cubes_example: from the root folder run:
```
python3 ./src/resizing_colorizing_marching_cubes_example.py
```
# Notes
- No conversion from SDF to TSDF has been done yet. Therefore, it is important to make sure that marching cubes still works after clamping.