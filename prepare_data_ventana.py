"""
This script is used to prepare the data for training :real data, (Ventana, 2022-11-14v1, scene_132429).

A few things to note:
1. The raw data is using NED-FRD convention, we need to convert it to ENU-FLU convention. This includes the pose and the FLS images (fliplr).
2. The point cloud used for training (which could be from altimeter), PC.npy, is E,N,Z, thus the implicit neural represention (h) is also E,N,Z, z= h[e,n]
3. When visualizing the heightmap using pyplot, h.T would be north up, east right.

Specifically for this dataset:
4. We have a prior map, but there is a global offset on X and Y. We have an initial estimation: (nav_offset_easting, nav_offset_northing).
We need to align the prior map with sonar using altimeter, using the flags coarse_search=1 and fine_search=1.
5. The FLS poses in ImageData.csv need to be adjusted as well. The roll has been added a constant 15 deg, and the sign of pitch is somehow flipped.
6. For other data from Ventana, we need to use the nav_raw data to double check the poses.
7. The altimeter data is not in the sonar frame, we need to figure it out a way to convert it to the sonar frame.
"""
import os
import cv2
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
import argparse 
import shutil

def bilinear_interpolate(heightmap, sigma=10):
    known_points = np.array(np.where(heightmap!=0)).T
    known_values = heightmap[heightmap!=0]

    # Create a meshgrid for interpolation
    grid_x, grid_y = np.mgrid[0:heightmap.shape[0]:1, 0:heightmap.shape[1]:1]

    # Perform linear interpolation
    interpolated_arr = griddata(known_points, known_values, (grid_x, grid_y), method='linear')

    # Perform nearest-neighbor fill for any NaNs left
    interpolated_arr[np.isnan(interpolated_arr)] = griddata(known_points, known_values, (grid_x, grid_y), method='nearest')[np.isnan(interpolated_arr)]
    if sigma>0:
        return gaussian_filter(interpolated_arr, sigma=sigma)
    else:
        return interpolated_arr



parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_base_dir', type=str, default="/root/Data/Gemini/Ventana", help="where we keep the raw data")
parser.add_argument('--base_dir', type=str, default=".", help="where we keep the processed data ready for training")

parser.add_argument('--x_min', type=float, default=-80, help="x_min for the heightmap, in meters")
parser.add_argument('--y_min', type=float, default=-80, help="y_min for the heightmap, in meters")
parser.add_argument('--x_max', type=float, default=40, help="x_max for the heightmap, in meters")
parser.add_argument('--y_max', type=float, default=40, help="x_max for the heightmap, in meters")
parser.add_argument('--nav_offset_easting', type=float, default=584690 , help="navigation offset (easting) initial estimation for the heightmap, in meters")
parser.add_argument('--nav_offset_northing', type=float, default=4067258 , help="navigation offset (nothing) initial estimation for the heightmap, in meters")
parser.add_argument('--nav_offset_northing_best', type=float, default=4067466.4 , help="navigation offset (easting) after alignment for the heightmap, in meters")
parser.add_argument('--nav_offset_easting_best', type=float, default=584687.5 , help="navigation offset (nothing) after alignment for the heightmap, in meters")
parser.add_argument('--res', type=float, default=0.1 , help="res for the heightmap, in meters")
parser.add_argument('--res_gt', type=float, default=1.0 , help="res for the gt heightmap, in meters")

parser.add_argument('--skip_step', type=int, default=50, help="for downsample the high frame rate FLS images")


args = parser.parse_args()

raw_data_base_dir = args.raw_data_base_dir
base_dir = args.base_dir

x_min = args.x_min
y_min = args.y_min
x_max = args.x_max
y_max = args.y_max
res = args.res
res_gt = args.res_gt
print("x_min y_min ", x_min, y_min)
print("x_max y_max ", x_max, y_max)

skip_step = args.skip_step


# NED Convention
#X,Y,Z,R,P,H,SEQUENCE_N,IMG_N
# The heading is changing
# my_data = genfromtxt(raw_data_base_dir + os.sep + 'scene_132429/ImageData.csv', delimiter=',')
my_data = genfromtxt(raw_data_base_dir + os.sep + 'scene_132429/ImageData_clean.csv', delimiter=',')
#X,Y,Z,R,P,H,ALTITUDE,UTIME,IMG_FILENAME

map_gt = np.load(raw_data_base_dir + os.sep +"SpongeRidge_MAUV_Topo1m_UTM_ROI.npz", allow_pickle=True)
nav_raw = np.load(raw_data_base_dir + os.sep +"nav_raw_20221114.npz", allow_pickle=True)["nav_raw"].item()
# nav_raw.keys() = t lat lon dep alt hdg utm_x utm_y


heightmap_gt = np.flipud(map_gt["map"])
geoformat = map_gt["geoformat"]
print("geoformat ", geoformat)

map_offset_easting = geoformat[0]
map_offset_northing= geoformat[3]
nav_offset_easting = args.nav_offset_easting
nav_offset_northing = args.nav_offset_northing + heightmap_gt.shape[1]



X = my_data[:,0] # Northing
Y = my_data[:,1] # Easting
Z = my_data[:,2] # Down

R = my_data[:,3] + math.radians(15) # there is a 15 deg offset Bastian added to roll
P = -my_data[:,4] # there is a 180 deg offset Bastian added to pitch?
H = my_data[:,5]

ALTITUDE = my_data[:,6]
UTIME = (my_data[:,7]).astype(np.int64)

# align timestamps
UTIME_float =  UTIME/1e6
idx_start = np.searchsorted(nav_raw["t"], UTIME_float[0])
idx_end = np.searchsorted(nav_raw["t"], UTIME_float[-1])

print("t", nav_raw["t"][idx_start-1], nav_raw["t"][idx_start],nav_raw["t"][idx_start+1], UTIME_float[0])
print("dep",nav_raw["dep"][idx_start-1], nav_raw["dep"][idx_start],nav_raw["dep"][idx_start+1], Z[0])
print("alt",nav_raw["alt"][idx_start-1],nav_raw["alt"][idx_start],nav_raw["alt"][idx_start+1], ALTITUDE[0])
print("hdg",nav_raw["hdg"][idx_start-1],nav_raw["hdg"][idx_start],nav_raw["hdg"][idx_start+1], H[0]/math.pi*180)
print("utm_x",nav_raw["utm_e"][idx_start-1],nav_raw["utm_e"][idx_start],nav_raw["utm_e"][idx_start+1], Y[0])
print("utm_n",nav_raw["utm_n"][idx_start-1],nav_raw["utm_n"][idx_start],nav_raw["utm_n"][idx_start+1], X[0])
print("pit",nav_raw["pit"][idx_start-1],nav_raw["pit"][idx_start],nav_raw["pit"][idx_start+1], X[0])
print("rol",nav_raw["rol"][idx_start-1],nav_raw["rol"][idx_start],nav_raw["rol"][idx_start+1], X[0])

plt.figure()
plt.plot(nav_raw["t"][idx_start-1:idx_end], nav_raw["pit"][idx_start-1:idx_end], label="nav_raw")
# plt.plot(nav_raw["t"][idx_start-1:idx_end], -nav_raw["pit"][idx_start-1:idx_end], label="-nav_raw")

plt.title("pitch (deg) ")
plt.plot(UTIME_float, P/math.pi*180, label="pose_label")
plt.legend()
plt.xlabel("timestamp")
plt.figure()
plt.plot(nav_raw["t"][idx_start-1:idx_end], nav_raw["rol"][idx_start-1:idx_end], label="nav_raw")
plt.title("rol (deg)")
plt.plot(UTIME_float, R/math.pi*180, label="pose_label")
plt.legend()
plt.xlabel("timestamp")


plt.figure()
plt.plot(nav_raw["t"][idx_start-1:idx_end], nav_raw["hdg"][idx_start-1:idx_end], label="nav_raw")
plt.title("heading (deg)")
plt.plot(UTIME_float, H/math.pi*180, label="pose_label")
plt.legend()
plt.xlabel("timestamp")

plt.figure()
plt.plot(nav_raw["t"][idx_start-1:idx_end], nav_raw["utm_n"][idx_start-1:idx_end], label="nav_raw")
plt.title("Northing ")
plt.plot(UTIME_float, X+nav_offset_northing- heightmap_gt.shape[1], label="pose_label")
plt.legend()
plt.xlabel("timestamp")

plt.figure()
plt.plot(nav_raw["t"][idx_start-1:idx_end], nav_raw["utm_e"][idx_start-1:idx_end], label="nav_raw")
plt.title("Easting ")
plt.plot(UTIME_float, Y+nav_offset_easting, label="pose_label")
plt.legend()
plt.xlabel("timestamp")


plt.figure()
plt.plot(nav_raw["t"][idx_start-1:idx_end], nav_raw["dep"][idx_start-1:idx_end], label="nav_raw")
plt.title("depth ")
plt.plot(UTIME_float, Z, label="pose_label")
plt.legend()
plt.xlabel("timestamp")


plt.figure()
plt.plot(nav_raw["t"][idx_start-1:idx_end], nav_raw["alt"][idx_start-1:idx_end], label="nav_raw")
plt.title("altitude ")
plt.plot(UTIME_float, ALTITUDE, label="pose_label")
plt.legend()
plt.xlabel("timestamp")

sonar_z_offset = -0.8 # NED convention

plt.figure()
plt.plot(nav_raw["t"][idx_start-1:idx_end], nav_raw["alt"][idx_start-1:idx_end]+nav_raw["dep"][idx_start-1:idx_end], label="nav_raw")
plt.title("altitude + depth")
plt.plot(UTIME_float, ALTITUDE+Z+sonar_z_offset, label="pose_label")
plt.legend()
plt.xlabel("timestamp")


plt.show()

plt.figure()
plt.plot(Y+nav_offset_easting, X+nav_offset_northing)
plt.scatter(Y[0]+nav_offset_easting, X[0]+nav_offset_northing, c='r')
idx=1200
plt.scatter(Y[idx]+nav_offset_easting, X[idx]+nav_offset_northing, c='blue')
title = " Heading red: "+str(H[0]/math.pi*180) + " Heading blue: "+str(H[idx]/math.pi*180)
plt.title(title)

plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")

plt.figure()
plt.imshow(heightmap_gt, origin="lower")

plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")

plt.figure()
plt.imshow(heightmap_gt, origin="lower", extent=[map_offset_easting,map_offset_easting+ heightmap_gt.shape[1], map_offset_northing, heightmap_gt.shape[0]+map_offset_northing])
plt.plot(Y+nav_offset_easting, X+nav_offset_northing)
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")

plt.show()

coarse_search = 0
fine_search = 0
if coarse_search:
    ### coarse search
    nav_offset_northing_init = 4067449.7#nav_offset_northing
    nav_offset_easting_init = 584690.9#nav_offset_easting
    error_min = np.inf
    ERROR = {}
    nav_offset_northing_best = None
    nav_offset_easting_best = None
    for nav_offset_northing in np.arange(nav_offset_northing_init-50, nav_offset_northing_init+50, 1):
        for nav_offset_easting in np.arange(nav_offset_easting_init-20, nav_offset_easting_init+20, 1):
            error =0
            ERROR[(nav_offset_northing, nav_offset_easting)]=[]
            # check if the xy alignmnet is correct
            for i in range(0,X.shape[0]):
                n = X[i]+nav_offset_northing-map_offset_northing
                e = Y[i]+nav_offset_easting-map_offset_easting
                assert n>=0 
                assert e>=0
                idx_n = int(n/res_gt)
                idx_e = int(e/res_gt)
                z = -Z[i]-ALTITUDE[i]
                z_heightmap = heightmap_gt[idx_n,idx_e]
                error += abs(z-z_heightmap)
                ERROR[(nav_offset_northing, nav_offset_easting)].append(abs(z-z_heightmap))
            error/=X.shape[0]
            if error<error_min:
                error_min = error
                nav_offset_northing_best = nav_offset_northing
                nav_offset_easting_best = nav_offset_easting

    STD = {}
    for (n,e) in ERROR.keys():
        STD[(n,e)]=np.std(ERROR[n,e])
    std_all = []
    mean_all = []
    for (n,e) in STD.keys():
        std_all.append(STD[(n,e)])
        mean_all.append(np.mean(ERROR[(n,e)]))
    plt.figure()
    plt.scatter(np.arange(len(std_all)), std_all,s=1, label="std")
    plt.title("std")

    plt.scatter(np.arange(len(mean_all)), mean_all,s=1, label="mean")
    plt.title("mean")
    plt.legend()
    plt.show()
    print("best northing, easting based on mean, error: ", nav_offset_northing_best, nav_offset_easting_best, error_min) #4067449.7 584690.9
    
    least_std_idx = np.where(std_all==np.min(std_all))[0][0]
    print(list(STD.keys())[least_std_idx], mean_all[least_std_idx], std_all[least_std_idx])
    nav_offset_northing_best,  nav_offset_easting_best = list(STD.keys())[least_std_idx] # 4067466.7 584687.9 
    tide_offset = mean_all[least_std_idx]
    print("tide_offset", tide_offset)

    print("best northing, easting based on std, std: ", nav_offset_northing_best, nav_offset_easting_best, std_all[least_std_idx])
    np.save("ERROR.npy", ERROR)
    print("saved ERROR.npy")


if fine_search:

    ## fine search
    nav_offset_northing_init, nav_offset_easting_init =  4067466.7, 584687.9
    error_min = np.inf
    ERROR = {}
    nav_offset_northing_best = None
    nav_offset_easting_best = None
    for nav_offset_northing in np.arange(nav_offset_northing_init-2, nav_offset_northing_init+2, 0.1):
        for nav_offset_easting in np.arange(nav_offset_easting_init-2, nav_offset_easting_init+2, 0.1):
            error =0
            ERROR[(nav_offset_northing, nav_offset_easting)]=[]
            # check if the xy alignmnet is correct
            for i in range(0,X.shape[0]):
                n = X[i]+nav_offset_northing-map_offset_northing
                e = Y[i]+nav_offset_easting-map_offset_easting
                assert n>=0 
                assert e>=0
                idx_n = int(n/res_gt)
                idx_e = int(e/res_gt)
                z = -Z[i]-ALTITUDE[i]
                z_heightmap = heightmap_gt[idx_n,idx_e]
                error += abs(z-z_heightmap)
                ERROR[(nav_offset_northing, nav_offset_easting)].append(abs(z-z_heightmap))
            error/=X.shape[0]
            if error<error_min:
                error_min = error
                nav_offset_northing_best = nav_offset_northing
                nav_offset_easting_best = nav_offset_easting

    STD = {}
    for (n,e) in ERROR.keys():
        STD[(n,e)]=np.std(ERROR[n,e])
    std_all = []
    mean_all = []
    for (n,e) in STD.keys():
        std_all.append(STD[(n,e)])
        mean_all.append(np.mean(ERROR[(n,e)]))
    plt.figure()
    plt.scatter(np.arange(len(std_all)), std_all,s=1, label="std")
    plt.title("std")

    plt.scatter(np.arange(len(mean_all)), mean_all,s=1, label="mean")
    plt.title("mean")
    plt.legend()
    plt.show()
    print("best northing, easting based on mean, error: ", nav_offset_northing_best, nav_offset_easting_best, error_min)# 4067464.7 584689.8

    least_std_idx = np.where(std_all==np.min(std_all))[0][0]
    print(list(STD.keys())[least_std_idx], mean_all[least_std_idx], std_all[least_std_idx])
    nav_offset_northing_best,  nav_offset_easting_best = list(STD.keys())[least_std_idx]
    tide_offset = mean_all[least_std_idx]
    print("tide_offset", tide_offset)
    print("best northing, easting based on std, std: ", nav_offset_northing_best, nav_offset_easting_best, std_all[least_std_idx]) # 4067466.4 584687.5
    np.save("ERROR.npy", ERROR)
    print("saved ERROR.npy")



else:
    nav_offset_northing_best =args.nav_offset_northing_best
    nav_offset_easting_best = args.nav_offset_easting_best
PC = np.array((Y,X,-Z-ALTITUDE-sonar_z_offset)).T

diff = []
Z_heightmap_gt = []
for (e,n,z) in PC:
    col=int((e+nav_offset_easting_best-map_offset_easting)/res_gt)
    row=int((n+nav_offset_northing_best-map_offset_northing)/res_gt)
    z_gt = heightmap_gt[row][col]
    if not np.isnan(z_gt):
        diff.append(z_gt-z)
        Z_heightmap_gt.append(z_gt)
diff = np.array(diff)
Z_heightmap_gt = np.array(Z_heightmap_gt)

tide_offset = np.mean(diff)
print("tide_offset", tide_offset)
tide_offset = -2.76 # this is actually for tide and altimeter offset


plt.figure()
plt.plot(Z_heightmap_gt, label="Z_heightmap_gt")
plt.plot(PC[:,2]+tide_offset, label="Z_altimeter")
plt.legend()
plt.show()





plt.figure()
plt.imshow(heightmap_gt, origin="lower", extent=[map_offset_easting,map_offset_easting+ heightmap_gt.shape[1], map_offset_northing, heightmap_gt.shape[0]+map_offset_northing])

plt.plot(Y+nav_offset_easting_best, X+nav_offset_northing_best)
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.xlim(x_min+nav_offset_easting_best, x_max+nav_offset_easting_best)
plt.ylim(y_min+nav_offset_northing_best, y_max+nav_offset_northing_best)

plt.show()
X_ENU = np.copy(Y) # Easting
Y_ENU = np.copy(X) # Northing
Z_ENU = -np.copy(Z) #- offset_z # Up

R_ENU = np.copy(R)
P_ENU = -np.copy(P)
H_ENU = math.pi/2 - np.copy(H)

plt.figure()
plt.scatter(X_ENU,Y_ENU,s=1,label="data")
plt.xlabel("Easting")
plt.ylabel("Northing")
plt.scatter(X_ENU[0],Y_ENU[0],c='r')
plt.figure()
plt.scatter(np.arange(H_ENU.shape[0]),H_ENU/math.pi*180,s=1,label="data")

plt.show()
# assert 1==0

# PC = np.array((X_ENU,Y_ENU,Z_ENU-ALTITUDE+tide_offset)).T
PC = np.array((X_ENU,Y_ENU,Z_ENU-ALTITUDE-sonar_z_offset)).T
# should we trust the bathymetry? or should we trust the altimeter?

print(PC[0],PC[-1])
print(PC.shape)
print(np.max(PC,axis=0),np.min(PC,axis=0))

print("x_max-x_min, y_max-y_min",x_max-x_min, y_max-y_min)

sigma = 1.0 # for Gaussian blur


heightmap_size = max(math.ceil((x_max-x_min)/res), math.ceil((y_max-y_min)/res))

heightmap = np.zeros((heightmap_size, heightmap_size))




# #   h[e,n]
PC_heightmap = []
for row in range(heightmap_gt.shape[0]):
    for col in range(heightmap_gt.shape[1]):
        n= row*1.0 + map_offset_northing
        e = col*1.0 + map_offset_easting
        if e-nav_offset_easting_best >=x_min and e-nav_offset_easting_best <=x_max and n-nav_offset_northing_best >=y_min and n-nav_offset_northing_best <=y_max:
            z = heightmap_gt[row][col]-tide_offset
            PC_heightmap.append([e-nav_offset_easting_best,n-nav_offset_northing_best,z])

PC_heightmap = np.array(PC_heightmap)

plt.figure()
plt.scatter(X_ENU,Y_ENU,s=1,label="data")
plt.xlabel("Easting")
plt.ylabel("Northing")



for (i, p) in enumerate(PC):
    x,y,z = p
    idx_x = math.floor((x-x_min)/res)
    idx_y = math.floor((y-y_min)/res)
    heightmap[idx_x][idx_y]=z


heightmap_interpolated = bilinear_interpolate(heightmap, sigma=sigma)
heightmap_interpolated[heightmap_interpolated==0]=np.nan


start_num = 0
end_num = X_ENU.shape[0] 

to_folder = base_dir + os.sep + "data/Gemini_132429"
if not os.path.exists(to_folder):
    os.makedirs(to_folder)


to_folder_img = to_folder+os.sep+"Data"
if not os.path.exists(to_folder_img):
    os.makedirs(to_folder_img)

vis_folder  = base_dir + os.sep + "Gemini_132429_images/"
if not os.path.exists(vis_folder):
    os.makedirs(vis_folder)

np.save(to_folder + os.sep + "heightmap_init.npy", heightmap_interpolated.astype(np.float32))
np.save(to_folder + os.sep + "PC.npy", PC.astype(np.float32))
np.save(to_folder + os.sep + "PC_heightmap.npy", PC_heightmap.astype(np.float32))

with open(raw_data_base_dir + os.sep +'scene_132429/ImageData_clean.csv', 'r') as file:
    data = file.readlines()




cal_mask_flag =1
valid_mask_heightmap = np.zeros((heightmap_interpolated.shape[0]//1, heightmap_interpolated.shape[1]//1))
valid_mask_heightmap_cnt = np.zeros((heightmap_interpolated.shape[0]//1, heightmap_interpolated.shape[1]//1))

for i in range(start_num, end_num):
    state = {}
    step =int(1)
    filename = raw_data_base_dir + os.sep + "scene_132429/" + data[i+1].split(",")[-1].split()[0]
    assert abs(float(data[i+1].split(",")[0]) - Y_ENU[i]) < 1e-6
    assert abs(float(data[i+1].split(",")[1]) - X_ENU[i]) < 1e-6
    assert abs(float(data[i+1].split(",")[2]) - Z[i]) < 1e-6
    assert abs(float(data[i+1].split(",")[5]) - H[i]) < 1e-6



    img_n = int(data[i+1].split("_")[-1].split(".")[0])
    # discard list
    if img_n in  [1804, 1954,3154,4610,5060,6226,6426,6676,8726]:
        continue
    if i%skip_step==0 or img_n in [1805,1956, 3156,4610,5068,6229,6429,6678,8730]:
        
        image = imread(filename)
        
        pose = np.array(transform.build_se3_transform([X_ENU[step*i],Y_ENU[step*i],Z_ENU[step*i],R_ENU[step*i],P_ENU[step*i],H_ENU[step*i]])) # TODO check this using GTSAM? Done!
        
        if cal_mask_flag:
            
            img = cv2.resize(np.fliplr(image), (image.shape[1]//1, image.shape[0]//1))
            valid_mask = img> 5e-3

            valid_idx_x, valid_idx_y = np.where(valid_mask)
            # valid_idx_x=valid_idx_x[valid_idx_x>3]
            # valid_idx_y=valid_idx_y[valid_idx_x>3]

            r_tmp = valid_idx_x*30/img.shape[0]
            theta_tmp = -math.radians(120/2) + valid_idx_y*math.radians(120)/img.shape[1] 
            phi_tmp = -np.arcsin( ALTITUDE[step*i]/r_tmp)

            pts = np.stack((r_tmp, theta_tmp, phi_tmp), axis=-1).reshape(-1, 3)
            X_r_rand = pts[:,0]*np.cos(pts[:,1])*np.cos(pts[:,2])
            Y_r_rand = pts[:,0]*np.sin(pts[:,1])*np.cos(pts[:,2])
            Z_r_rand = pts[:,0]*np.sin(pts[:,2])
            pts_r_rand = np.stack((X_r_rand, Y_r_rand, Z_r_rand, np.ones_like(X_r_rand)))
            pts_r_rand = np.matmul(pose, pts_r_rand)

            pts_r_rand = np.stack((pts_r_rand[0,:], pts_r_rand[1,:], pts_r_rand[2,:])).T
            valid_mask_heightmap[((pts_r_rand[:,0]-x_min)/res).astype(np.int16),((pts_r_rand[:,1]-y_min)/res).astype(np.int16)]=1
            valid_mask_heightmap_cnt[((pts_r_rand[:,0]-x_min)/res).astype(np.int16),((pts_r_rand[:,1]-y_min)/res).astype(np.int16)]+=1





        





        state["ImagingSonar"] = np.fliplr(image.astype(np.float32))
        state["PoseSensor"] = pose.astype(np.float32) 
        file = open(to_folder_img+os.sep+str(img_n)+'.pkl', 'wb')
        pickle.dump(state, file)
        file.close()
        shutil.copy2(filename, vis_folder)
    
valid_mask_heightmap = cv2.resize(valid_mask_heightmap, heightmap.shape)
np.save(to_folder + os.sep +"valid_mask_heightmap.npy", valid_mask_heightmap.astype(np.bool_))
np.save(to_folder + os.sep +"valid_mask_heightmap_cnt.npy", valid_mask_heightmap_cnt.astype(np.float32))

kernel = np.ones((35, 35), np.uint8) 
valid_mask_heightmap_dilation = cv2.dilate(valid_mask_heightmap, kernel )
valid_mask_heightmap_dilation[valid_mask_heightmap_dilation>0]=1
kernel = np.ones((35, 35), np.uint8) 

valid_mask_heightmap_dilation_erosion = cv2.erode(valid_mask_heightmap_dilation*1.0, kernel)
valid_mask_heightmap_dilation_erosion[valid_mask_heightmap_dilation_erosion>0]=1
plt.figure(figsize=(10,10))
plt.imshow(valid_mask_heightmap_dilation_erosion.T, origin="lower")
plt.show()
