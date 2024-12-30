# Bathymetric Surveying With Imaging Sonar Using Neural Volume Rendering

This repo contains the source code for IEEE RAL paper (OA) [Bathymetric Surveying With Imaging Sonar Using Neural Volume Rendering
](https://ieeexplore.ieee.org/document/10631294) , a work collabrating with CoMPAS Lab, MBARI.


# Usage

To run it (on bigtuna, MBARI):

## Docker Environment

Build docker:

```shell
docker build -f  Dockerfile -t mbari/fls:neusis_ngp --no-cache .
```

Before running the docker container, make sure you have access to ```/DATA/COOK/neusis_ngp/Data``` where I keep the data for different missions.

Run docker image, start the container:
```shell
bash run_docker.sh
```
The docker container by default mounts ```/DATA/COOK/neusis_ngp/Data``` to ```/root/Data```, where the data is stored.
And```/DATA/COOK/neusis_ngp/Experiments``` to ```/root/repos/neusis_ngp/experiments```, where all the training results will be stored.


## Prepare Data 
For different datasets, i.e., simulation data, Ventana data, and LASS data, run `prepare_data_<dataset_name>.py` inside of the container. It will prepare the cooked data into the ```/root/repos/neusis_ngp/data``` directory.   

The data is organized as follows:

```
data/<dataset_name>
|-- Data
    |-- <timestamp 1>.pkl        # data for each view (includes the sonar image and SE3 pose)
    |-- <timestamp 2>.pkl 
    ...
|-- PC.npy      # Point clouds coming from the altimeter
|-- heightmap_gt.npy # Ground truth bathymetry (heightmap)
```

## Cooked Data

Simulation dataset (scene_aerial_01, Fig.5 in the paper) is available here [sim_dataset](https://drive.google.com/drive/folders/1OpSWMz7LJencPayykG5wYcR4ZFFHgruZ?usp=sharing).


## Pip Install

Alternatively, one can skip docker and create a virtual conda environment and install using pip: 

```
  conda create -n neusis_ngp python=3.8
  conda activate neusis_ngp
  pip install -r requirements.txt
  pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

```

## Running

Training. Inside of the container ```cd /root/repos/neusis_ngp``` and run:


``` python run_sdf_2.5D.py --conf confs/<dataset_name>.conf --gpu=1```

Example:
``` python run_sdf_2.5D.py --conf confs/scene_aerial_01.conf  --gpu=1```

The resulting heightmaps are saved in the following directory ```experiments/<dataset_name>/meshes```. 
The checkpoints are saved in ```experiments/<dataset_name>/checkpoints```. 
The logs are saved in ```experiments/<dataset_name>/logs```, which can be visualized using tensorboard:
```shell
tensorboard --logdir=experiments/<dataset_name>/logs --port=6006

```



# Notes on training
1) We used an NVIDIA 3090 GPU for training. Depending on available compute, consider adjusting the following parameters in ```confs/<dataset_name>.conf```:

Parameter  | Description
------------- | -------------
arc_n_samples  | number of samples along each arc
n_samples | number of samples along each acoustic ray
n_importance | number of samples along each arc using importance sampling

The final samples along the arc would be arc_n_samples+n_importance.  If n_importance is set to 0, then we would only have arc_n_samples along the arc.

# Citation 
Consider citing as below if you find our work helpful to your project:

```
@ARTICLE{10631294,
  author={Xie, Yiping and Troni, Giancarlo and Bore, Nils and Folkesson, John},
  journal={IEEE Robotics and Automation Letters}, 
  title={Bathymetric Surveying With Imaging Sonar Using Neural Volume Rendering}, 
  year={2024},
  volume={9},
  number={9},
  pages={8146-8153},
  doi={10.1109/LRA.2024.3440843}}

```

# Acknowledgement
Some code snippets are borrowed from [IDR](https://github.com/lioryariv/idr), [NeuS](https://github.com/Totoro97/NeuS), [NeusIS](https://github.com/rpl-cmu/neusis), [Instant-NSR](https://github.com/zhaofuq/Instant-NSR), [zipnerf-pytorch](https://github.com/SuLvXiangXin/zipnerf-pytorch). Thanks for these projects!

