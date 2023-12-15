#!/usr/bin/env bash

xhost +local:root

export XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    touch $XAUTH
    xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
fi

docker run -it \
    --name neusis_ngp \
    --privileged \
    -e "DISPLAY=$DISPLAY" \
    -e "XAUTHORITY=$XAUTH" \
    -e "QT_X11_NO_MITSHM=1" \
    -v "$XAUTH:$XAUTH" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
    -v "/DATA/COOK/neusis_ngp/Data:/root/Data:ro" \
    -v "/DATA/COOK/neusis_ngp/Experiments:/root/repos/neusis_ngp/experiments:rw" \
    --runtime=nvidia \
    --net=host \
    $@ \
    mbari/fls:neusis_ngp
