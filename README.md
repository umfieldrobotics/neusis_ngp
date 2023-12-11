# Bathymetry Reconstruction using Imaging Sonar with Neural Rendering

This repo contains the source code for the work collabrating with MBARI.


# Usage

## Docker Environment

Build docker:

```shell
docker build -f  Dockerfile -t mbari/fls:neusis_ngp --build-arg="READ_TOKEN=<GITHUB_TOKEN>" .
```

Run docker image, start the container:
```shell
bash run_docker.sh
```

## Data 
For different datasets, i.e., simulation data, Ventana data, and LASS data, run `prepare_data_<dataset_name>.py`. It will prepare the cooked data into the ```data``` directory.   
The data is organized as follows:

```
data/<dataset_name>
|-- Data
    |-- <pose 1>.pkl        # data for each view (includes the sonar image and pose)
    |-- <pose 2>.pkl 
    ...
|-- Config.json      # Sonar configuration
```

## Running
Copy the ```data/<dataset_name>``` inside of the docker container.

```
docker cp ./data/<dataset_name> <container_id>/root/repos/neusis_ngp/data/
```

Training, inside of the container ```cd /root/repos/neusis_ngp``` and run:

``` python run_sdf_2.5D.py --conf confs/<dataset_name>.conf --gpu=1```

Example:
``` python run_sdf_2.5D.py --conf confs/scene_aerial_01.conf  --gpu=1```

The resulting heightmaps are saved in the following directory ```experiments/<dataset_name>/meshes```. 


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

