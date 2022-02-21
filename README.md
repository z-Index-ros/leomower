# Leo Mower
As a learning exercice we're going to mimic an [Husqvarna Automower](https://www.husqvarna.com/fr/robots-tondeuses/automower315/) with the [Leo Rover](https://www.leorover.tech/). 

> The Husqvarna Automower playground is delimited by an electrical cable buried in the ground, the mower turns when it reaches the limit.

> Leo Mower will use its front camera plus a collision detection model to detect the grass edge. The collision detection model is based on the [Jetbot Collision Avoidance notebook](https://jetbot.org/master/examples/collision_avoidance.html).

The steps are

1. [prepare Leo with the code](#Preparation)
2. [setup the prerequisites](#Python-Prerequisites)
3. [Launch!](#launch-the-ROS-nodes)

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

# Ubuntu
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

As the LeoMower uses PyTorch for inference, we have to install Pytorch.

``` bash
pip3 install torch torchvision 
```

> Refer to [PyTorch](https://pytorch.org/get-started/locally/) documentation


## Launch the ROS nodes

Now we can launch the LeoMower ROS nodes (place your Leo in the center of your garden first :blush: ), and connect to Leo's Wifi

``` bash
roslaunch leomower leomower.launch
```

[![Watch the video](https://i.ytimg.com/vi/L12pAv4vds8/default.jpg)](https://youtu.be/L12pAv4vds8)






