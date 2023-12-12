# Bathymetry Reconstruction using Imaging Sonar with Neural Rendering

This repo contains the source code for the work collabrating with MBARI.


# Usage

To run it on bigtuna:

## Docker Environment

Build docker:

```shell
docker build -f  Dockerfile -t mbari/fls:neusis_ngp --build-arg="READ_TOKEN=<GITHUB_TOKEN>" --no-cache .
```

The ```<GITHUB_TOKEN>``` is needed since it is a private repo, it could be found in the nottion page.

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
    |-- <pose 1>.pkl        # data for each view (includes the sonar image and pose)
    |-- <pose 2>.pkl 
    ...
|-- PC.npy      # Point clouds coming from the altimeter
|-- PC_heightmap.npy      # Point clouds coming from the ground truth bathymetric map

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

The final samples along the arc would be arc_n_samples+n_importance



# Acknowledgement
Some code snippets are borrowed from [IDR](https://github.com/lioryariv/idr), [NeuS](https://github.com/Totoro97/NeuS), [NeusIS](https://github.com/rpl-cmu/neusis), [Instant-NSR](https://github.com/zhaofuq/Instant-NSR), [zipnerf-pytorch](https://github.com/SuLvXiangXin/zipnerf-pytorch). Thanks for these projects!

