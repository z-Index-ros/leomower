# Leo Mower
As a learning exercice we're going to mimics an [Husqvarna Automower](https://www.husqvarna.com/fr/robots-tondeuses/automower315/) with the [Leo Rover](https://www.leorover.tech/). 

The Husqvarna Automower playground is delimited by an electrical cable buried in the ground, the mower turns when it reaches the limit.

Leo Mower will use its front camera plus a collision detection model to detect the grass edge.

## Preparation

Create a catkin workspace and clone the `leomower` package to the source space:
``` bash
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src
git clone https://github.com/z-Index-ros/leomower.git
```

Build the workspace:
``` bash
cd ~/ros_ws/
catkin_make
```

Source the workspace:
``` bash
# windows
devel\setup.bat
```

Launch (temp, will change)
``` bash
rosrun leomower scripts/talker.py

# you should see
[INFO] [1643064135.638000]: hello world 1643064135.64
[INFO] [1643064136.650000]: hello world 1643064136.65
```

> Don't forget to launch the ROS Core in a separate console : 
>``` bash
> roscore
>```
>Launch the listener node
>``` bash
> rosrun leomower scripts/listener.py
> [INFO] [1643064564.902000]: /listener_15116_1643064564105I heard hello world 1643064135.64
> [INFO] [1643064565.902000]: /listener_15116_1643064564105I heard hello world 1643064136.65
>```




