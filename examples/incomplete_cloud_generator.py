from math import dist
from turtle import distance
import numpy as np
import open3d as o3d

#Reading Cloud
pc_path = "/media/rauldds/TOSHIBA EXT/ML3G/object_dataset/desk/005_00001.bin"
scan = np.fromfile(pc_path, dtype=np.float32)
scan = scan[1:]
scan = scan.reshape((-1,11))
scan = scan[:,:3]
scan = scan.astype(np.float64)

#Reading number of points in the cloud
num_points = scan.shape[0]
print(f"number of points in the cloud: {num_points}")

#Sampling the points
su = int(np.random.uniform(0,num_points,1))
#sn = int(np.random.randint(0,num_points))
print(f"uniform sample from points: {su}")
#print(f"normal sample: {sn}")

# Probably would be a good idea to modify the elimination percentage to up to 75 percent
elimination_percentage = np.random.random_sample()
print(f"percentage of the cloud that'll be eliminated: {elimination_percentage}")

#Get coordinates of the sampled point
center = (scan[su,:]).reshape((1,3))

#Eliminate points based on K-NN. Where K is the number of points to be eliminated
# First obtain euclidean distance (other distance could also be used)
distances = (scan-center)**2
distances = distances.sum(1)
distances = (distances**(1/2)).reshape((num_points))

#Calculate the number of points that'll be kept
num_eliminations = int(elimination_percentage*num_points)
kept_points = num_points - num_eliminations
print(f"number of points to be kept: {kept_points}")

# Obtain the indices of the points that'll be kept.
indices = np.argpartition(distances,-kept_points)[-kept_points:]
incomplete_scan = scan[indices,:]
print(f"shape of incomple scan: {incomplete_scan.shape}")

#Visualize incomplete scan
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(incomplete_scan)

o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[0, 0, 0],
                                  up=[-0.0694, -0.9768, 0.2024])
