# Leo Mower
As a learning exercice we're going to mimic an [Husqvarna Automower](https://www.husqvarna.com/fr/robots-tondeuses/automower315/) with the [Leo Rover](https://www.leorover.tech/). 

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


## Start leo in gazebo
Add the leo_mower_gazebo/models folders to /home/.gazebo/models so that the models are added to the gazebo db
you can now launch the simulation 

```
roslaunch leo_mower_gazebo golf_course.launch
```

## Launch the ROS nodes
open a new terminal 

```
catkin build
source ./devel/setup.bash
roslaunch leomower leomower.launch
```


Here's what it should look like

https://user-images.githubusercontent.com/15012463/155016426-3410192b-1dfd-4be2-9028-2d1da45a1008.mov

## Explore how it works

The first step is collecting data, then training the model before using the trained model in the realtime inference.

The three steps are explained separately:

* [Data Collection](./doc/data_collection.md)
* [Train the model](./doc/train.md)
* [Infer and drive](./doc/infer.md)

