# Infer in realtime 

Now it's time to program the mimic of the Automower.

For that we need two distinct nodes:

* infer.py
* drive.py

## infer.py

The `infer` node subscribes to the __/camera/image_raw__ topic and register continuously the image to a private field `self.lastImage`.

The infer node starts also a timer that will trigger an image inference at a rate lower than the image rate sent by the camera. 

> This way we will avoid the flooding of the node as the infer takes longer than the image flow. With the Leo's Rapsberry Pi, the inference takes about 0.5 seconds.

The goal of the infer node is to publish indications to a __/leomower/collision__ topic following the rule:

* __free__: when the model prediction of __blocked__ label is less than the threshold (node parameter)
* __blocked__: when the prediction is higher than the threshold

## drive.py

The `drive` node subscribes to the __/leomower/collision__ topic.

Depending on the __/leomower/collision__ topic value the Leo must go straight forward (when __free__) or must spin (when __blocked__). 

Therefore the node publishes `Twists` on the  __cmd_vel__ topic to command the rover moves.

> Hint to get smooth rover movement: as we need to publish Twists in a 'continuous way' but as the inference is performed every 0.5 seconds only, the 'last twist' is saved in a private `self.twist` and a timer triggers the publish of this `self.twist` at a higher rate.

> Remember, to move the rover you only have to publish `Twist` to the __cmd_vel__ topic. You can do it using the `rostopic pub` command
> ``` shell
> rostopic pub -1 /cmd_vel geometry_msgs/Twist '{linear:  {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0,y: 0.0,z: 0.0}}'
> ```

Thats all folks...
