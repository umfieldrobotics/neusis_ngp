"""
This script is used to prepare the data for training (real data, Ventana+LASS , 2023-10-10, 13:39-14:17).

A few things to note:
1. The raw data is using NED-FRD convention, we need to convert it to ENU-FLU convention. This includes the pose and the FLS images (fliplr).
2. The point cloud used for training (which could be from altimeter), PC.npy, is E,N,Z, thus the implicit neural represention (h) is also E,N,Z, z= h[e,n]
3. When visualizing the heightmap using pyplot, h.T would be north up, east right.

Specifically for this dataset:
4. We have a prior map with 0.05m resolution, but there is a global offset on X and Y. We have an initial estimation: (nav_offset_easting, nav_offset_northing).
We need to align the prior map with sonar using altimeter, using the flags coarse_search=1 and fine_search=1.
5. The FLS images somehow have different heights 1888 and 1887, we need to make sure all images have the same shape.
6. For other data from Ventana+LASS, we need to use the nav_raw data to double check the poses.
7. The altimeter data is not in the sonar frame, we need to figure it out a way to convert it to the sonar frame.
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
import shutil
import argparse 
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_base_dir', type=str, default="/root/Data/Gemini/LASS", help="where we keep the raw data")
parser.add_argument('--base_dir', type=str, default=".", help="where we keep the processed data ready for training")

parser.add_argument('--nav_offset_easting', type=float, default=584690 , help="navigation offset (easting) initial estimation for the heightmap, in meters")
parser.add_argument('--nav_offset_northing', type=float, default=4067258 , help="navigation offset (nothing) initial estimation for the heightmap, in meters")
parser.add_argument('--nav_offset_northing_best', type=float, default=4067466.4 , help="navigation offset (easting) after alignment for the heightmap, in meters")
parser.add_argument('--nav_offset_easting_best', type=float, default=584687.5 , help="navigation offset (nothing) after alignment for the heightmap, in meters")
parser.add_argument('--res', type=float, default=0.1 , help="res for the heightmap, in meters")
parser.add_argument('--res_gt', type=float, default=0.05 , help="res for the gt heightmap, in meters")

parser.add_argument('--skip_step', type=int, default=50, help="for downsample the high frame rate FLS images")


args = parser.parse_args()

raw_data_base_dir = args.raw_data_base_dir
base_dir = args.base_dir
res_gt = args.res_gt
res = args.res

skip_step = args.skip_step


header=  "#utime,Northing,Easting,Depth,Roll,Pitch,Heading,Latitude,Longitude,Height"
nav_file = raw_data_base_dir + os.sep + "20231010v1/img_navdata.txt"
clean_nav_file = raw_data_base_dir + os.sep + "20231010v1/img_navdata_clean.csv"
img_meta_file = raw_data_base_dir + os.sep + "20231010v1/imgs/img-meta.csv"
nav_corrected_file = raw_data_base_dir + os.sep + "20231010v1/nav_navadj_20231010_survey_fls.npz"
nav_corrected_data=np.load(nav_corrected_file, allow_pickle=True)["nav_adj"].item()
# dict_keys(['t', 'lat', 'lon', 'dep', 'hdg', 'pit', 'rol', 'utm_e', 'utm_n'])



with open(img_meta_file, "r") as f:
   img_meta=f.readlines()
print("img_meta ", img_meta[0])
print("img_meta ", img_meta[1])   

with open(nav_file, "r") as f:
   my_data=f.readlines()

# print(len(my_data), my_data[6])
# my_data_clean = []
# for (i, data) in enumerate(my_data):
#     if i<6:
#         continue
#     data_i = data.split(',')
#     if not data_i[1]=="nan":
#         data_i_arr = []

#         for (j, data_ij) in enumerate(data_i):
#             if j==0:
#                 data_i_arr.append(np.int64(data_ij))
#             elif j==len(data_i)-1:
#                 data_i_arr.append(np.float64(data_ij.split()[0]))
#             else:
#                 data_i_arr.append(np.float64(data_ij))
#         data_i_arr = np.array(data_i_arr)
#         my_data_clean.append(data_i_arr)
# print(my_data_clean[0])
# np.savetxt(clean_nav_file, my_data_clean, delimiter=',', fmt='%9f', header=header)

my_data = genfromtxt(clean_nav_file, delimiter=',')#, skip_header=1)
print(my_data[0])
UTIME = np.int64(my_data[:,0])

print(UTIME[0])

UTIME_offset = 315558000000000
for i in range(0, len(UTIME)):
    filename = glob.glob(raw_data_base_dir + os.sep + "20231010v1/imgs/"+str(UTIME[i]-UTIME_offset)+".png")
    # print(len(filename), filename)
    assert len(filename)==1


map = np.load(raw_data_base_dir + os.sep +"SpongeRidge_LASS_Topo_5cm_UTM_ROI.npz", allow_pickle=True)
heightmap_gt = np.flipud(map["arr_0"])# heightmap_gt[n,e]
geoformat = map["arr_1"]
print("geoformat", geoformat)

map_offset_easting = geoformat[0]
map_offset_northing= geoformat[3]- heightmap_gt.shape[0]*res_gt


# dict_keys(['t', 'lat', 'lon', 'dep', 'hdg', 'pit', 'rol', 'utm_e', 'utm_n'])

# align timestamps
UTIME_float =  UTIME/1e6


idx_start = np.searchsorted(nav_corrected_data["t"], UTIME_float[0])
idx_end = np.searchsorted(nav_corrected_data["t"], UTIME_float[-1])

print(UTIME_float[0], datetime.datetime.fromtimestamp(UTIME_float[0]))
print(UTIME_float[-1], datetime.datetime.fromtimestamp(UTIME_float[-1]))


X = my_data[:,1] # Northing
Y = my_data[:,2] # Easting
Z = my_data[:,3] # Down

R = my_data[:,4] #  roll
P = my_data[:,5] # pitch
H = my_data[:,6] # heading
ALTITUDE = np.copy(my_data[:,9]) # Altitude
ALTITUDE[ALTITUDE>10]=np.nan

diff_dep = []
PC_corrected = []
for i, utime in enumerate(UTIME_float):
    idx = np.searchsorted(nav_corrected_data["t"], utime)
    e = nav_corrected_data["utm_e"][idx]
    n = nav_corrected_data["utm_n"][idx]
    z = -nav_corrected_data["dep"][idx]-ALTITUDE[i] 
    if (not np.isnan(z)) and (not np.isnan(e)) and (not np.isnan(n)):
        PC_corrected.append(np.array([e,n,z]))
        diff_dep.append(Z[i]-nav_corrected_data["dep"][idx])
PC_corrected=np.array(PC_corrected)

print("t", nav_corrected_data["t"][idx_start-1], nav_corrected_data["t"][idx_start],nav_corrected_data["t"][idx_start+1], UTIME_float[0])
print("dep",nav_corrected_data["dep"][idx_start-1], nav_corrected_data["dep"][idx_start],nav_corrected_data["dep"][idx_start+1], Z[0])
# print("alt",nav_corrected_data["alt"][idx_start-1],nav_corrected_data["alt"][idx_start],nav_corrected_data["alt"][idx_start+1], ALTITUDE[0])
print("hdg",nav_corrected_data["hdg"][idx_start-1],nav_corrected_data["hdg"][idx_start],nav_corrected_data["hdg"][idx_start+1], H[0]/math.pi*180)
print("utm_x",nav_corrected_data["utm_e"][idx_start-1],nav_corrected_data["utm_e"][idx_start],nav_corrected_data["utm_e"][idx_start+1], Y[0])
print("utm_n",nav_corrected_data["utm_n"][idx_start-1],nav_corrected_data["utm_n"][idx_start],nav_corrected_data["utm_n"][idx_start+1], X[0])
print("pit",nav_corrected_data["pit"][idx_start-1],nav_corrected_data["pit"][idx_start],nav_corrected_data["pit"][idx_start+1], X[0])
print("rol",nav_corrected_data["rol"][idx_start-1],nav_corrected_data["rol"][idx_start],nav_corrected_data["rol"][idx_start+1], X[0])
plt.figure()
plt.plot(UTIME/1e6,Z,label='Z')
plt.plot(nav_corrected_data["t"][idx_start:idx_end], nav_corrected_data["dep"][idx_start:idx_end],label='dep')
plt.legend()
print(np.mean(diff_dep), np.std(diff_dep))
print(np.mean(np.abs(diff_dep)), np.std(np.abs(diff_dep)))

sonar_z_offset = np.mean(diff_dep)

X_ENU = np.copy(Y) # Easting
Y_ENU = np.copy(X) # Northing
Z_ENU = -np.copy(Z) #- offset_z # Up

R_ENU = np.copy(R)
P_ENU = -np.copy(P)
H_ENU = math.pi/2 - np.copy(H)
# PC = np.array((X_ENU,Y_ENU,Z_ENU-ALTITUDE)).T
PC = np.array((X_ENU[~np.isnan(ALTITUDE)],Y_ENU[~np.isnan(ALTITUDE)],Z_ENU[~np.isnan(ALTITUDE)]-ALTITUDE[~np.isnan(ALTITUDE)])).T


coarse_search = 0
fine_search = 0

if coarse_search:
    # register the map to trajectory using altitude
    ### coarse search
    nav_offset_easting_init =0
    nav_offset_northing_init = 0
    error_min = np.inf
    ERROR = {}
    nav_offset_northing_best = None
    nav_offset_easting_best = None
    for nav_offset_northing in np.arange(nav_offset_northing_init-25, nav_offset_northing_init+10, 1):
        for nav_offset_easting in np.arange(nav_offset_easting_init-25, nav_offset_easting_init+25, 1):
            error =0
            ERROR[(nav_offset_northing, nav_offset_easting)]=[]
            # check if the xy alignmnet is correct
            for (e,n, z) in PC:
                idx_n = int((n+nav_offset_northing-map_offset_northing)/res_gt)
                idx_e = int((e+nav_offset_easting-map_offset_easting)/res_gt)
                z_heightmap = heightmap_gt[idx_n,idx_e]
                if np.isnan(z_heightmap):
                    continue
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
    # plt.title("std")

    plt.scatter(np.arange(len(mean_all)), mean_all,s=1, label="mean")
    # plt.title("mean")
    plt.legend()
    plt.show()
    print("best northing, easting based on mean, error: ", nav_offset_northing_best, nav_offset_easting_best, error_min)


    least_std_idx = np.where(std_all==np.min(std_all))[0][0]
    print(list(STD.keys())[least_std_idx], mean_all[least_std_idx], std_all[least_std_idx])
    nav_offset_northing_best,  nav_offset_easting_best = list(STD.keys())[least_std_idx]
    tide_offset = mean_all[least_std_idx]
    print("tide_offset", tide_offset)
    print("best northing, easting based on std, std: ", nav_offset_northing_best, nav_offset_easting_best, std_all[least_std_idx])

    np.save("ERROR.npy", ERROR)
    print("saved ERROR.npy")
if fine_search:
    # register the map to trajectory using altitude
    ### fine search
    nav_offset_easting_init =0
    nav_offset_northing_init = 0
    error_min = np.inf
    ERROR = {}
    nav_offset_northing_best = 13
    nav_offset_easting_best = 2
    for nav_offset_northing in np.arange(nav_offset_northing_init+nav_offset_northing_best-3, nav_offset_northing_init+nav_offset_northing_best+3, 0.05):
        for nav_offset_easting in np.arange(nav_offset_easting_init+nav_offset_easting_best-3, nav_offset_easting_init+nav_offset_easting_best+3, 0.05):
            error =0
            ERROR[(nav_offset_northing, nav_offset_easting)]=[]
            # check if the xy alignmnet is correct
            for (e,n, z) in PC:
                idx_n = int((n+nav_offset_northing-map_offset_northing)/res_gt)
                idx_e = int((e+nav_offset_easting-map_offset_easting)/res_gt)
                z_heightmap = heightmap_gt[idx_n,idx_e]
                if np.isnan(z_heightmap):
                    continue
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
        
    plt.scatter(np.arange(len(std_all)), std_all,s=1, label="std")
    plt.title("std")

    plt.scatter(np.arange(len(mean_all)), mean_all,s=1, label="mean")
    plt.title("mean")
    plt.legend()
    plt.show()
    print("best northing, easting based on mean, error: ", nav_offset_northing_best, nav_offset_easting_best, error_min)


    least_std_idx = np.where(std_all==np.min(std_all))[0][0]
    print(list(STD.keys())[least_std_idx], mean_all[least_std_idx], std_all[least_std_idx])
    nav_offset_northing_best,  nav_offset_easting_best = list(STD.keys())[least_std_idx]
    print("best northing, easting based on std, std: ", nav_offset_northing_best, nav_offset_easting_best, std_all[least_std_idx])

    tide_offset = mean_all[least_std_idx]
    np.save("ERROR.npy", ERROR)
    print("saved ERROR.npy")

else:
    nav_offset_easting_best = 13#12.75#13
    nav_offset_northing_best = 2#1.9#2
    tide_offset=-1.750100283526973#-1.732#-1.75

X_ENU = np.copy(Y) +nav_offset_easting_best# Easting
Y_ENU = np.copy(X) +nav_offset_northing_best# Northing
Z_ENU = -np.copy(Z)  # Up

R_ENU = np.copy(R)
P_ENU = -np.copy(P)
H_ENU = math.pi/2 - np.copy(H)
# PC = np.array((X_ENU,Y_ENU,Z_ENU-ALTITUDE)).T
PC_new = np.array((X_ENU[~np.isnan(ALTITUDE)],Y_ENU[~np.isnan(ALTITUDE)],Z_ENU[~np.isnan(ALTITUDE)]-ALTITUDE[~np.isnan(ALTITUDE)])).T




# print("map_offset_easting", map_offset_easting)
# print("map_offset_northing", map_offset_northing)
print("map, ", [map_offset_easting,map_offset_easting+heightmap_gt.shape[1]*res_gt, map_offset_northing, map_offset_northing+heightmap_gt.shape[0]*res_gt])
print("X north", X.max(),X.min())
print("Y east", Y.max(),Y.min())
plt.figure()
plt.plot(Y+nav_offset_easting_best, X+nav_offset_northing_best, label="trajectory")
plt.plot(Y, X, label="trajectory_ori")

plt.scatter(Y[0], X[0], c='red')
plt.scatter(Y[0]+nav_offset_easting_best, X[0]+nav_offset_northing_best, c='red')
plt.plot(nav_corrected_data["utm_e"][idx_start:idx_end],nav_corrected_data["utm_n"][idx_start:idx_end], label="nav_rov")
plt.scatter(nav_corrected_data["utm_e"][idx_start],nav_corrected_data["utm_n"][idx_start], c='red')

plt.legend()


plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.imshow(heightmap_gt, origin="lower", cmap='turbo', extent=[map_offset_easting,map_offset_easting+heightmap_gt.shape[1]*res_gt, map_offset_northing, map_offset_northing+heightmap_gt.shape[0]*res_gt])

x_min, x_max = (584229+nav_offset_easting_best, 584309+nav_offset_easting_best) # East
y_min, y_max = (4067826+nav_offset_northing_best, 4067906+nav_offset_northing_best)# North



plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
# plt.show()



#   h[e,n], heightmap_gt[n,e]
PC_heightmap = []
for row in range(heightmap_gt.shape[0]):# north
    for col in range(heightmap_gt.shape[1]):#east
        n= row*res_gt + map_offset_northing
        e = col*res_gt + map_offset_easting
        if e >=x_min and e <=x_max and n >=y_min and n <=y_max:
            z = heightmap_gt[row][col]
            if not np.isnan(z):
                PC_heightmap.append([e,n,z])
PC_heightmap = np.array(PC_heightmap)

diff = []
Z_heightmap_gt = []
UTIME_z_heightmap_gt = []
for i,(e,n,z) in enumerate(PC):
    col=int((e+nav_offset_easting_best-map_offset_easting)/res_gt)
    row=int((n+nav_offset_northing_best-map_offset_northing)/res_gt)
    z_gt = heightmap_gt[row][col]
    if not np.isnan(z_gt):
        diff.append(z_gt-z)
        Z_heightmap_gt.append(z_gt)
        UTIME_z_heightmap_gt.append(UTIME[i])
diff = np.array(diff)
Z_heightmap_gt = np.array(Z_heightmap_gt)

# tide_offset = np.mean(diff)
print("tide_offset", tide_offset)


plt.figure()
# plt.plot(Z_heightmap_gt, label="Z_heightmap_gt")
plt.scatter(np.arange(Z_heightmap_gt.shape[0]), Z_heightmap_gt, label="Z_heightmap_gt", s=1)
plt.scatter(np.arange(PC.shape[0]), PC[:,2]+tide_offset, label="Z_altimeter", s=1)

# plt.plot(PC[:,2]+tide_offset, label="Z_altimeter")
plt.legend()
plt.title("Z_heightmap_gt vs Z_altimeter after tide_offset")
plt.show()


PC = np.array((X_ENU,Y_ENU,Z_ENU-ALTITUDE+tide_offset)).T
PC_new = np.array((X_ENU[~np.isnan(ALTITUDE)],Y_ENU[~np.isnan(ALTITUDE)],Z_ENU[~np.isnan(ALTITUDE)]-ALTITUDE[~np.isnan(ALTITUDE)] +tide_offset )).T


print('here' ,len(UTIME))

to_folder = base_dir + os.sep + "data/Gemini_lass"
if not os.path.exists(to_folder):
    os.makedirs(to_folder)

to_folder_img = to_folder+os.sep+"Data"
if not os.path.exists(to_folder_img):
    os.makedirs(to_folder_img)

vis_folder  = base_dir + os.sep + "Gemini_lass_images/"
if not os.path.exists(vis_folder):
    os.makedirs(vis_folder)



np.save(to_folder+os.sep+"heightmap_init.npy", heightmap_gt.astype(np.float32))
np.save(to_folder+os.sep+"PC.npy", PC_new.astype(np.float32))# after the filtering of nan    
np.save(to_folder+os.sep+"PC_heightmap.npy", PC_heightmap.astype(np.float32))
## 12851 imgs

PC_traj = []
xys=[]
UTIME_traj = []
cal_mask_flag =1
valid_mask_heightmap = np.zeros(( int((x_max-x_min)/res)//1, int((y_max-y_min)/res)//1))
valid_mask_heightmap_cnt = np.zeros(( int((x_max-x_min)/res)//1, int((y_max-y_min)/res)//1))
for i in range(0, len(UTIME)):
    step = 1
    filename = glob.glob(raw_data_base_dir+os.sep+"20231010v1/imgs/"+str(UTIME[i]-UTIME_offset)+".png")
    # print(len(filename), filename)
    assert len(filename)==1
    if i%skip_step==0:
        if np.isnan(ALTITUDE[i]):
            continue
        image = imread(filename[0])
        if len(image.shape)==3:
            image = image[:,:,0]
        if image.shape[0]==1888:
            image=image[1:,:]
        pose = np.array(transform.build_se3_transform([X_ENU[step*i],Y_ENU[step*i],Z_ENU[step*i],R_ENU[step*i],P_ENU[step*i],H_ENU[step*i]])) # TODO check this using GTSAM? Done!
        
        if cal_mask_flag:
            import cv2
            img = cv2.resize(np.fliplr(image), (image.shape[1]//1, image.shape[0]//1))
            
            valid_mask = img> 0.1

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
 
        state = {}
        state["ImagingSonar"] = np.flipud(np.fliplr(image.astype(np.float32)))
        state["PoseSensor"] = pose.astype(np.float32) 
        file = open(to_folder_img+os.sep+str(i)+'.pkl', 'wb')
        pickle.dump(state, file)
        file.close()
        xys.append([X_ENU[step*i],Y_ENU[step*i]])
        PC_traj.append([X_ENU[step*i],Y_ENU[step*i],Z_ENU[step*i]-ALTITUDE[step*i]+tide_offset ])
        UTIME_traj.append(UTIME[i])
        shutil.copy2(filename[0], vis_folder)
        assert image.shape[0]==1887
        assert image.shape[1]==512

plt.figure()
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
xys=np.array(xys)
print(xys.shape)
plt.imshow(heightmap_gt, origin="lower", cmap='turbo', extent=[map_offset_easting,map_offset_easting+heightmap_gt.shape[1]*res_gt, map_offset_northing, map_offset_northing+heightmap_gt.shape[0]*res_gt])
plt.scatter(xys[:,0],xys[:,1], s=1, c="r")
plt.plot(xys[:,0],xys[:,1],c='blue')

plt.show()
PC_traj=np.array(PC_traj)
np.save(to_folder+os.sep+"PC_traj.npy", PC_traj.astype(np.float32))# after the filtering of nan    


plt.figure()
plt.plot(UTIME_z_heightmap_gt, Z_heightmap_gt, label="Z_heightmap_gt")
plt.plot(UTIME_traj, PC_traj[:,2], label="Z_altimeter")

plt.legend()
plt.title("Z_heightmap_gt vs Z_altimeter after tide_offset")


plt.show()

print("PC_traj.shape", PC_traj.shape)
print("PC.shape", PC_new.shape, np.max(PC_new, axis=0), np.min(PC_new, axis=0))

print("PC_heightmap.shape", PC_heightmap.shape)
valid_mask_heightmap = cv2.resize(valid_mask_heightmap, heightmap_gt.shape)
np.save(to_folder+os.sep+"valid_mask_heightmap.npy", valid_mask_heightmap.astype(np.bool_))
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
