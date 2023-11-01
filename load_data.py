import os
# import cv2
import pickle 
import json 
import math
from scipy.io import savemat
import numpy as np

def load_data(target,base_dir="./",PC_name="PC.npy",heightmap_name="heightmap.npy"):
    dirpath = base_dir+"data/{}".format(target)
    pickle_loc = "{}/Data".format(dirpath)
    output_loc = "{}/UnzipData".format(dirpath)
    cfg_path = "{}/Config.json".format(dirpath)
    PC_loc = dirpath + os.sep+ PC_name 
    heightmap_init_loc = dirpath + os.sep+ heightmap_name 

    if os.path.exists(PC_loc):
        print("loading Point Cloud ", PC_loc)
        PC = np.load(PC_loc)
    else:
        PC = None
    if os.path.exists(heightmap_init_loc):
        print("loading heightmap ", heightmap_init_loc)

        heightmap_init = np.load(heightmap_init_loc)
    else:
        heightmap_init = None

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    for agents in cfg["agents"][0]["sensors"]:
        if agents["sensor_type"] != "ImagingSonar": continue
        hfov = agents["configuration"]["Azimuth"]
        vfov = agents["configuration"]["Elevation"]
        min_range = agents["configuration"]["RangeMin"]
        max_range = agents["configuration"]["RangeMax"]
        hfov = math.radians(hfov)
        vfov = math.radians(vfov)
        if "ElevationMin" in agents["configuration"]:
            min_phi = agents["configuration"]["ElevationMin"]
            max_phi = agents["configuration"]["ElevationMax"]
            min_phi = math.radians(min_phi)
            max_phi = math.radians(max_phi)
        else:
            min_phi = None
            max_phi = None

    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
    images = []
    sensor_poses = []


    for pkls in os.listdir(pickle_loc):
        filename = "{}/{}".format(pickle_loc, pkls)
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            image = state["ImagingSonar"]
            s = image.shape
            # image[image < 0.2] = 0
            # image[s[0]- 200:, :] = 0
            pose = state["PoseSensor"]
            images.append(image)
            sensor_poses.append(pose)

    data = {
        "images": images,
        "images_no_noise": [],
        "sensor_poses": sensor_poses,
        "min_range": min_range,
        "max_range": max_range,
        "hfov": hfov,
        "vfov": vfov,
        "min_phi": min_phi,
        "max_phi": max_phi,
        "heightmap_init":heightmap_init,
        "PC":PC,
    }
    
    # savemat('{}/{}.mat'.format(dirpath,target), data, oned_as='row')
    return data
