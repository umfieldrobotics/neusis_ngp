import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
import sys
from pyhocon import ConfigFactory
import argparse 

def pc2grid(pc, bounds, res):
    """
    pc has shape (N,3), in ENU;
    bounds[0]=xmin,ymin
    bounds[1]=xmax,ymax
    return height_map, h[n][e]=z
    """
    assert bounds.shape == (2,2)
    minx, miny = bounds[0] # 
    maxx, maxy = bounds[1] # 
    xres = res
    yres = res
    cols = int((maxx - minx)/xres)
    rows = int((maxy - miny)/yres)
    height_map = np.zeros((rows,cols))
    for (x,y,z) in pc:
        idx_x = int((x-minx)/xres)
        idx_y = int((y-miny)/yres)
        if idx_y<height_map.shape[0] and idx_x<height_map.shape[1]:
            height_map[idx_y][idx_x]=z
    return height_map

def cal_mae(h_gt, h, res_gt=1.0, res=0.1, n_min=-35, n_max=2,  e_min=-45, e_max=15, bounds=np.array([[-80,-80],[40,40]])):
    """
    h_gt[n,e]
    h[e,n]
    """
    minx, miny = bounds[0]# E, N
    error = []
    for n in np.arange(n_min, n_max+res_gt, res_gt):
        for e in np.arange(e_min, e_max+res_gt, res_gt):
            idx_e = int((e-minx)/res)
            idx_n = int((n-miny)/res)
            idx_e_gt = int((e-minx)/res_gt)
            idx_n_gt = int((n-miny)/res_gt)
            if idx_e<h.shape[0] and idx_n<h.shape[1] and idx_n_gt<h_gt.shape[0] and idx_e_gt<h_gt.shape[1]:

                z = h[idx_e][idx_n]            
                z_gt = h_gt[idx_n_gt][idx_e_gt]
                error.append(z-z_gt)
    return error

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default=".", help="where we keep the processed data ready for training")
parser.add_argument('--dataset_name', type=str, required=True, help="dataset name: scene_aerial_01, Gemini_132429, Gemini_lass")
parser.add_argument('--slurm_id', type=str, default="", help="slurm id")
parser.add_argument('--idx', type=int, default=-1, help="by default load the last saved heightmap")

args = parser.parse_args()
base_dir = args.base_dir

base_exp_dir =  base_dir + os.sep+ "experiments"
base_data_dir =  base_dir + os.sep+ "data"

dataset_name = args.dataset_name
conf_path = "confs/{}.conf".format(dataset_name)

f = open(conf_path)
conf_text = f.read()

conf = ConfigFactory.parse_string(conf_text)

x_max = conf.get_float('mesh.x_max')
x_min = conf.get_float('mesh.x_min')
y_max = conf.get_float('mesh.y_max')
y_min = conf.get_float('mesh.y_min')
z_max = conf.get_float('mesh.z_max')
z_min = conf.get_float('mesh.z_min')
vmax = z_max + 1
vmin = z_min - 1

res = conf.get_float('mesh.res')

slurm_id = args.slurm_id


files = sorted(glob.glob(base_exp_dir+os.sep+ dataset_name +"_"+ slurm_id + "/meshes/*.npy"))
idx = args.idx


h=np.load(files[idx])
print(len(files), files[idx])
print(h[h!=0].max(), h.min())
PC_heightmap = np.load(base_data_dir+os.sep+dataset_name+os.sep+"PC_heightmap.npy")
PC = np.load(base_data_dir+os.sep+dataset_name+os.sep+"PC.npy")


if dataset_name=="scene_aerial_01":
    height_map_from_PC = np.load(base_data_dir+os.sep+dataset_name+os.sep+"heightmap_gt.npy").T
elif dataset_name=="Gemini_132429":
    height_map_from_PC = pc2grid(PC_heightmap, np.array([[x_min,y_min],[x_max,y_max]]),res=1)

elif dataset_name=="Gemini_lass":
    height_map_from_PC = np.load(base_data_dir+os.sep+dataset_name+os.sep+"heightmap_init.npy")

print(height_map_from_PC[height_map_from_PC!=0].max(), height_map_from_PC.min())



 

plt.figure()
ax = plt.gca()
if dataset_name=="Gemini_132429":

    im = plt.imshow(h.T, origin="lower", cmap="turbo", extent=[x_min,x_max,y_min,y_max], vmax=vmax, vmin=vmin)
else:
    im = plt.imshow(h.T, origin="lower", cmap="turbo", extent=[x_min,x_max,y_min,y_max], vmax=vmax, vmin=vmin)

plt.plot(PC[:,0],PC[:,1])
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.title(slurm_id)
plt.colorbar()


plt.figure()
ax = plt.gca()
im = plt.imshow(height_map_from_PC, origin="lower", cmap="turbo", extent=[x_min,x_max,y_min,y_max], vmax=vmax, vmin=vmin)

plt.plot(PC[:,0],PC[:,1])
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")

plt.colorbar()

plt.show()


## beam pattern
files = sorted(glob.glob(base_exp_dir+os.sep+ dataset_name +"_"+ slurm_id + "/checkpoints/*.pth"))

print(files[idx])
model = torch.load(files[idx], map_location=torch.device('cpu'))
variance = model["variance_network_fine"]["variance"]
inv_s = torch.exp(1.0*variance).item()
print("idx, inv_s ", idx, inv_s)
color = model["color_network_fine"]
beamform_k_azimuth = color["beamform_k_azimuth"].cpu()
beamform_k_elevation = color["beamform_k_elevation"].cpu()
print("beamform_k_azimuth ", beamform_k_azimuth.shape, beamform_k_azimuth.max(), beamform_k_azimuth.min(), beamform_k_azimuth.mean())
print("beamform_k_elevation ", beamform_k_elevation.shape, beamform_k_elevation.max(), beamform_k_elevation.min(), beamform_k_elevation.mean())


theta = (torch.arange(120)-60)/180*math.pi
hfov = math.radians(120)
beamform_k=beamform_k_azimuth
nbr_angles = beamform_k.shape[0]
step = hfov/(nbr_angles-1)
kernel_angles = torch.arange(-hfov/2, hfov/2+step, step).unsqueeze(-1).requires_grad_(False) # K x 1 
d_angle = hfov/nbr_angles
bwidth = 1./(2.*d_angle**2)
ang_dist = F.softmax(-bwidth*(theta.view(1,-1)-kernel_angles)**2, dim=0) # K x N
beamform = torch.sum(ang_dist*torch.exp(beamform_k), dim=0) #  N

ang_dist_kernel = F.softmax(-bwidth*(kernel_angles.view(1,-1)-kernel_angles)**2, dim=0) # K x N
beamform_kernel = torch.sum(ang_dist_kernel*torch.exp(beamform_k), dim=0) #  N

beamform_azimuth_kernel = beamform_kernel.detach().cpu().numpy()/nbr_angles

beamform_azimuth = beamform.detach().cpu().numpy()/nbr_angles


kernel_angles = kernel_angles.detach().cpu().numpy()
print("beamform_azimuth ", beamform_azimuth.max(), beamform_azimuth.min() )

plt.figure()
plt.plot(theta/math.pi*180, beamform_azimuth, label="NN")
plt.scatter(kernel_angles/math.pi*180, beamform_azimuth_kernel, s=10,  facecolors='none', edgecolors='r',label="kernel")
plt.xlabel("Azimuth (deg)")
plt.title(slurm_id)
plt.grid()
beamform_azimuth_log = np.log10(beamform_azimuth+1.0)

plt.figure()
ax = plt.subplot(111, polar=True)# Adjust the theta for correct orientation
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.plot(theta, beamform_azimuth_log)
plt.title("Sonar Beam Pattern in dB")
plt.xlabel("Azimuth Angle (degrees)")
plt.ylabel("Intensity (dB)")

# print("beamform_k_elevation ", beamform_k_elevation)
phi = (torch.arange(65)-60)/180*math.pi
beamform_k = torch.concat((-10*torch.ones(1,1), beamform_k_elevation, -10*torch.ones(1,1)),dim=0) 

nbr_angles = beamform_k.shape[0]
print("nbr_angles", nbr_angles)
step = math.radians(50)/(nbr_angles-1)
kernel_angles = torch.arange(-math.radians(55), math.radians(-5)+step, step).unsqueeze(-1).requires_grad_(False) # K x 1 
d_angle = math.radians(50)/nbr_angles
bwidth = 1./(2.*d_angle**2)
ang_dist = F.softmax(-bwidth*(phi.view(1,-1)-kernel_angles)**2, dim=0) # K x N
beamform = torch.sum(ang_dist*torch.exp(beamform_k), dim=0) #  N

ang_dist_kernel = F.softmax(-bwidth*(kernel_angles.view(1,-1)-kernel_angles)**2, dim=0) # K x N
beamform_kernel = torch.sum(ang_dist_kernel*torch.exp(beamform_k), dim=0) #  N

beamform_elevation_kernel = beamform_kernel.detach().cpu().numpy()/nbr_angles


beamform_elevation = beamform.detach().cpu().numpy()/nbr_angles

print("beamform_elevation ", beamform_elevation.max(), beamform_elevation.min() )

plt.figure()
plt.plot(phi/math.pi*180, beamform_elevation, label="NN")
plt.scatter(kernel_angles/math.pi*180, beamform_elevation_kernel, s=10,  facecolors='none', edgecolors='r',label="kernel")
plt.xlabel("Elevation (deg)")
plt.title(slurm_id)
plt.grid()
beamform_elevation_log = np.log10(beamform_elevation+1.0)


plt.figure()
ax = plt.subplot(111, polar=True)# Adjust the theta for correct orientation
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.plot(phi, beamform_elevation_log)
plt.title("Sonar Beam Pattern in dB")
plt.xlabel("Elevation Angle (degrees)")
plt.ylabel("Intensity (dB)")


plt.show()



