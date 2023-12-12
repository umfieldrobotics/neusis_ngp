"""
This script is used to prepare the data for training: simulation data, (scene_aerial_01, with pickle file).

A few things to note:
1. The raw data is using NED-FRD convention, we need to convert it to ENU-FLU convention. This includes the pose and the FLS images (fliplr).
2. The point cloud used for training (which could be from altimeter), PC.npy, is E,N,Z, thus the implicit neural represention (h) is also E,N,Z, z= h[e,n]
3. When visualizing the heightmap using pyplot, h.T would be north up, east right.

Specifically for this dataset:
3. The data from pose_labels.csv has an offset in the X and Y, thus we use the XY from piclke file instead.
4. The raw FLS images are in very high frame rate, we only pick every 500th frame for training (85 images in total).
5. We flipud the FLS images so that when the idx is 0, the pixel has r_min

In theory, all simulation data could use this script to prepare the data for training. If we have the pickle file or we know the global offset somehow.
And make sure passing the right configuration to the parser.
"""

import os
#import cv2
import pickle 
import json 
import math
from scipy.io import savemat
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from numpy import genfromtxt
import transform
import numpy as np
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import glob 
from pyhocon import ConfigFactory
import imageio.v3 as iio
from PIL import Image
import sys
import argparse 

def bilinear_interpolate(heightmap, sigma=10):
    known_points = np.array(np.where(heightmap!=0)).T
    known_values = heightmap[heightmap!=0]

    # Create a meshgrid for interpolation
    grid_x, grid_y = np.mgrid[0:heightmap.shape[0]:1, 0:heightmap.shape[1]:1]

    # Perform linear interpolation
    interpolated_arr = griddata(known_points, known_values, (grid_x, grid_y), method='linear')

    # Perform nearest-neighbor fill for any NaNs left
    # interpolated_arr[np.isnan(interpolated_arr)] = griddata(known_points, known_values, (grid_x, grid_y), method='nearest')[np.isnan(interpolated_arr)]
    if sigma>0:
        return gaussian_filter(interpolated_arr, sigma=sigma)
    else:
        return interpolated_arr





parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_base_dir', type=str, default="/root/Data/Sim", help="where we keep the raw data")
parser.add_argument('--base_dir', type=str, default=".", help="where we keep the processed data ready for training")

parser.add_argument('--z_offset_stonefish', type=float, default=9.03, help="z_offset for the heightmap, in meters")
parser.add_argument('--scale_x', type=float, default=0.09856 , help="scale_x for the heightmap, in meters")
parser.add_argument('--scale_y', type=float, default=0.09856 , help="scale_x for the heightmap, in meters")
parser.add_argument('--height', type=float, default=5, help="max height for the heightmap, in meters")
parser.add_argument('--res', type=float, default=0.09856 , help="res for the heightmap, in meters, ideally the same as scale_x and scale_y")
parser.add_argument('--uint16_max', type=int, default=255 , help="quantization for the heightmap, 255 (8bit) or 65535 (16bit)")

parser.add_argument('--uint8_fls', type=int, default=255 , help="quantization for the FLS intensity, 255 (8bit) or 65535 (16bit)")
parser.add_argument('--skip_step', type=int, default=500 , help="for downsample the high frame rate FLS images")


args = parser.parse_args()

z_offset_stonefish = args.z_offset_stonefish # m
scale_x = args.scale_x  # m 
scale_y = args.scale_y  # m
height = args.height # m
uint16_max = args.uint16_max 
res = args.res # m


z_min = -z_offset_stonefish
z_max = height-z_offset_stonefish

uint8_fls = args.uint8_fls
skip_step = args.skip_step

data_dir = args.raw_data_base_dir + os.sep + "20231010_fls_multibeam_survey-20231012T190750Z-002/20231010_fls_multibeam_survey"
file_path = data_dir+os.sep+"maps/aerial_01.png"

heightmap_gt = iio.imread(file_path).astype(np.float32)[:,:,0]# 1024 
heightmap_gt = (heightmap_gt/uint16_max)*height -z_offset_stonefish
heightmap_gt = np.flipud(heightmap_gt) #(rows/height/Y/Easting) x 1024 (columns/width/X/Northing) 
# NOTE: we need h[e,n] for the NN
width = heightmap_gt.shape[1]

x_min = -width/2*scale_x
y_min = -width/2*scale_y
x_max = width/2*scale_x
y_max = width/2*scale_y
print("x_min, y_min res ",x_min, y_min, res)


print(heightmap_gt.max(), heightmap_gt.min())

size_x, size_y = heightmap_gt.shape 
full_size_x = (size_x-1) * scale_x 
full_size_y = (size_y-1) * scale_y
size_x_, size_y_ = math.ceil(full_size_x/res), math.ceil(full_size_y/res)
heightmap_gt_ = np.array(Image.fromarray(heightmap_gt).resize((size_x_, size_y_)))
plt.figure()
plt.imshow(heightmap_gt.T, vmax=z_max,vmin=z_min,cmap="turbo",origin="lower",extent=[-size_x*scale_x/2,size_x*scale_x/2,-size_y*scale_y/2,size_y*scale_y/2])
plt.colorbar()
plt.xlabel("Easting")
plt.ylabel("Northing")
# plt.savefig("figs/heightmap_gt.png", dpi=300)

plt.figure()
plt.imshow(heightmap_gt_.T, vmax=z_max,vmin=z_min,cmap="turbo",origin="lower",extent=[-size_x*scale_x/2,size_x*scale_x/2,-size_y*scale_y/2,size_y*scale_y/2])
plt.colorbar()
plt.xlabel("Easting")
plt.ylabel("Northing")
# plt.savefig("figs/heightmap_gt_.png", dpi=300)

plt.show()

my_data = genfromtxt(data_dir+os.sep+"pose_labels.csv", delimiter=',')
print("nbr of poses ", my_data.shape)
offset_n,offset_e = -17.613202159268365, -15.52896763447695

X = my_data[:,0] # Northing
Y = my_data[:,1] # Easting
Z = my_data[:,2] # Down
print(Z.max(),Z.min())

R = my_data[:,3]
P = my_data[:,4]
H = my_data[:,5]
UTIME =  my_data[:,8].astype(np.int64)# I think it is ms?

filename = data_dir+ os.sep+ "filtered_lcmlog_MB_FLS_heading.pkl"
with open(filename, 'rb') as f:
    state = pickle.load(f)

# dict_keys(['DETECT_BAD_FILTER', 'DETECT_BAD_MANUAL', 'DETECT_BAD_SONAR', 'DETECT_OK', 'header', 'n_beams', 'flags', 'range', 'intensity', 'lcm_timestamp'])
SIM_MOLA_SEN_MB = state["SIM_MOLA_SEN_MB"]
beams_x = SIM_MOLA_SEN_MB["range"]["x"]
beams_y = SIM_MOLA_SEN_MB["range"]["y"]
beams_z = SIM_MOLA_SEN_MB["range"]["z"]

beams_frd = np.concatenate((np.array(beams_x).reshape(-1,1),np.array(beams_y).reshape(-1,1),np.array(beams_z).reshape(-1,1)),axis=1) # N,3

# dict_keys(['header', 'pose', 'twist', 'lcm_timestamp'])
SIM_MOLA_NAV_STATE = state["SIM_MOLA_NAV_STATE"]

# dict_keys(['header', 'rph', 'angRate', 'lcm_timestamp'])
SIM_MOLA_NAV_STATE_RPH = state["SIM_MOLA_NAV_STATE_RPH"]


x_NED=np.array(SIM_MOLA_NAV_STATE["pose"]["pose"]["position"]["x"])
y_NED=np.array(SIM_MOLA_NAV_STATE["pose"]["pose"]["position"]["y"])
z_NED=np.array(SIM_MOLA_NAV_STATE["pose"]["pose"]["position"]["z"])

rph_NED = np.array(SIM_MOLA_NAV_STATE_RPH['rph']) # N,3

X1=np.zeros_like(X)
Y1=np.zeros_like(Y)
Z1=np.zeros_like(Z)
R1=np.zeros_like(R)
P1=np.zeros_like(P)
H1=np.zeros_like(H)

NAV_ts = SIM_MOLA_NAV_STATE["header"]["timestamp"] #N

# # sycronization
for (i, ts) in enumerate(UTIME):
    j = NAV_ts.index(ts)
    assert (ts-NAV_ts[j])<1e-6
    X1[i] = x_NED[j]
    Y1[i] = y_NED[j]
    Z1[i] = z_NED[j]
    R1[i] = rph_NED[j][0]
    P1[i] = rph_NED[j][1]
    H1[i] = rph_NED[j][2]

plt.figure()
plt.plot(R/math.pi*180,label="FLS")
plt.plot(R1/math.pi*180, label="SIM_MOLA_NAV_STATE")
plt.title("roll (deg)")
plt.legend()

plt.figure()
plt.plot(P/math.pi*180,label="FLS")
plt.plot(P1/math.pi*180, label="SIM_MOLA_NAV_STATE")
plt.title("pitch (deg)")
plt.legend()

plt.figure()
plt.plot(H/math.pi*180,label="FLS")
plt.plot(H1/math.pi*180, label="SIM_MOLA_NAV_STATE")
plt.title("heading (deg)")
plt.legend()

plt.show()


# X_ENU = np.copy(Y)+offset_e # Easting
# Y_ENU = np.copy(X)+offset_n # Northing
# Z_ENU = -np.copy(Z)  # Up

# R_ENU = np.copy(R)
# P_ENU = -np.copy(P)
# H_ENU = math.pi/2 - np.copy(H)

## Here we use XYZ from pickle file since the XY from pose_labels.csv have an offset
X_ENU = np.copy(Y1) # Easting
Y_ENU = np.copy(X1) # Northing
Z_ENU = -np.copy(Z1)  # Up

R_ENU = np.copy(R)
P_ENU = -np.copy(P)
H_ENU = math.pi/2 - np.copy(H)


plt.imshow(heightmap_gt.T, vmax=-4,vmin=-10,cmap="turbo",origin="lower", extent=[-size_x*scale_x/2,size_x*scale_x/2,-size_y*scale_y/2,size_y*scale_y/2])
plt.colorbar()
plt.xlabel("Easting")
plt.ylabel("Northing")
plt.scatter(X_ENU,Y_ENU,s=1)
# plt.xlim(-20,20)
# plt.ylim(-20,20)
# plt.savefig("heightmap_gt_50x50.png")



#   h[e,n]
PC_heightmap = []
for row in range(heightmap_gt.shape[0]):
    for col in range(heightmap_gt.shape[1]):
        z = heightmap_gt[row][col]
        e= row*res + x_min
        n = col*res + y_min
        
        PC_heightmap.append([e,n,z])

PC_heightmap = np.array(PC_heightmap)

print("PC_heightmap.shape ", PC_heightmap.shape)
PC = []
heightmap=np.zeros_like(heightmap_gt)
for i in range(my_data.shape[0]):
    n = Y_ENU[i]
    e = X_ENU[i]
    idx_n = math.floor((n-x_min)/res)
    idx_e = math.floor((e-y_min)/res)
    z = heightmap_gt[idx_e][idx_n]
    heightmap[idx_e,idx_n]=z
    PC.append([e,n,z])
PC = np.array(PC)

heightmap_plot = heightmap.copy()
heightmap_plot[heightmap_plot==0]=np.nan
heightmap_interpolated = bilinear_interpolate(heightmap, sigma=1)
heightmap_interpolated_plot = heightmap_interpolated.copy()
heightmap_interpolated_plot[heightmap_interpolated_plot==0]=np.nan
plt.figure()
plt.imshow(heightmap_interpolated_plot.T, origin="lower",extent=[-size_x*scale_x/2,size_x*scale_x/2,-size_y*scale_y/2,size_y*scale_y/2],vmax=z_max,vmin=z_min,cmap="turbo")

print(heightmap_interpolated[heightmap_interpolated!=0].max(), heightmap_interpolated[heightmap_interpolated!=0].min())
plt.xlabel("Easting")
plt.ylabel("Northing")
# plt.xlim(-20,20)
# plt.ylim(-20,20)
plt.colorbar()
plt.show()





to_folder = args.base_dir + os.sep +"data/scene_aerial_01"
to_folder_img = to_folder+os.sep+"Data"

if not os.path.exists(to_folder):
    os.makedirs(to_folder)
if not os.path.exists(to_folder_img):
    os.makedirs(to_folder_img)



np.save(to_folder+os.sep+"PC_heightmap.npy", np.array(PC_heightmap).astype(np.float32))
np.save(to_folder+os.sep+"PC.npy", np.array(PC).astype(np.float32))
np.save(to_folder+os.sep+"heightmap_gt_.npy", heightmap_gt_.astype(np.float32))
np.save(to_folder+os.sep+"heightmap_gt.npy", heightmap_gt.astype(np.float32))

files_list = sorted(os.listdir(data_dir+os.sep+"scene_aerial_01/images"))
print("len(files_list) ", len(files_list)) # 42355


PC_traj = []
for i, utime in enumerate(UTIME):

    filename = "{}/{}.{}".format(data_dir+os.sep+"scene_aerial_01/images", utime, "png")
    if not filename.endswith('.png'):
        continue
    # 85 imgs are used for training
    if i%skip_step==0:
        state = {}
        image = iio.imread(filename).astype(np.float32)/uint8_fls
        pose = np.array(transform.build_se3_transform([X_ENU[i],Y_ENU[i],Z_ENU[i],R_ENU[i],P_ENU[i],H_ENU[i]])) # 

        # NOTE We always need to fliplr since with training we use the ENU-FLU(ISO 8855), yaw positive left, but the image is in NED-FRD, yaw positive right
        # sometimes, I need to flipud, to make sure when idx is 0, it is the r_min
        state["ImagingSonar"] = np.fliplr(np.flipud(image.astype(np.float32)))

        state["PoseSensor"] = pose.astype(np.float32) 
        n = Y_ENU[i]
        e = X_ENU[i]
        idx_n = math.floor((n-x_min)/res)
        idx_e = math.floor((e-y_min)/res)
        z = heightmap_gt[idx_e][idx_n]
        PC_traj.append([e,n,z])
            
        file = open(to_folder_img+os.sep+str(utime)+'.pkl', 'wb')
        pickle.dump(state, file)
        file.close()

PC_traj = np.array(PC_traj)
np.save(to_folder+os.sep+"PC_traj.npy", np.array(PC_traj).astype(np.float32))
