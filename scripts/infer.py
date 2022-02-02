#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import torch
import torchvision


class Infer():

    def __init__(self):

        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, 2)

        # subscribe to image topic
        rospy.Subscriber('image_raw', String, self.callback)

        # Create ROS publisher
        self.publisher = rospy.Publisher("collision", String, queue_size=1)


    def callback(self, image):

        collision = 'free'
        self.publisher.publish(collision)
        rospy.loginfo()

def main():
    # init the node
    rospy.init_node('infer', anonymous=True)
    Infer()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    main()
