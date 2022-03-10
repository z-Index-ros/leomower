# Web Leo Mower
This branch is used to manipulate the leomower from a web page

> The Husqvarna Automower playground is delimited by an electrical cable buried in the ground, the mower turns when it reaches the limit.

> Leo Mower will use its front camera plus a collision detection model to detect the grass edge. The collision detection model is based on the [Jetbot Collision Avoidance notebook](https://jetbot.org/master/examples/collision_avoidance.html).

The steps are

1. [prepare Leo with the code](#Preparation)
2. [setup the prerequisites](#Python-Prerequisites)
3. [3, 2, 1, Launch!](#launch-the-ROS-nodes)
4. After that, we'll [explore](#explore-how-it-works) how it work

## Preparation

On the Leo, create a catkin workspace and clone the `leomower` package to the source space:
``` bash
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src
git clone https://github.com/z-Index-ros/leomower.git
```

Build the workspace:
``` bash
cd ~/ros_ws/

# windows
catkin_make

# Ubuntu (sudo apt install python3-catkin-tools)
catkin build
```

Source the workspace:
``` bash
# windows
devel\setup.bat

# Ubuntu
source ./devel/setup.bash
```

## Python Prerequisites

As the LeoMower uses PyTorch for inference, we have to install Pytorch, plus other packages used in thee project:

``` bash
sudo pip3 install torch torchvision 
sudo pip3 install -U numpy
sudo pip3 install readkeys
```

> Refer to [PyTorch](https://pytorch.org/get-started/locally/) documentation


## Launch the ROS nodes

Now we can launch the LeoMower ROS nodes (place your Leo in the center of your garden first :blush: ), and connect to Leo's Wifi

``` bash
roslaunch leomower leomower.launch
```
## install the websocket rosbridge_suite
Refer to [rosbridge_suite](http://wiki.ros.org/rosbridge_suite) to install and launch the websocket

## Web page
Now that your leomower package and the websocket are running you can open the index.html page on your favorite browser and click the start/stop buttons to enable the leomower movement

## Host webPage on local server
If you want to host your web page on a local server on your rover, you can download and configure [nginx](https://ubuntu.com/tutorials/install-and-configure-nginx#1-overview). 

## Explore how it works

The first step is collecting data, then training the model before using the trained model in the realtime inference.

The three steps are explained separately:

* [Data Collection](./doc/data_collection.md)
* [Train the model](./doc/train.md)
* [Infer and drive](./doc/infer.md)


