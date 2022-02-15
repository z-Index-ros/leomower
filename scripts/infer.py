#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image
import torch.nn.functional as F
import cv2
from cv_bridge import CvBridge
import os
import numpy as np
from torch.autograd import Variable
from datetime import datetime

# ROS node that subscribes to images topic, process to image inference and publishes the 
# collision status (free or blocked)
class Infer():

    def __init__(self):

        rospy.loginfo("Start working in %s", os.getcwd())

        # set the threshold to compare the "blocked probability", lower the value is, safer the pathway will be
        self.blocked_threshold = 0.15

        # inference rate (per second)
        self.infer_freq = 0.5

        # set the path to the trained model (output from the train_model_resnet18.py)
        state_dict_path = 'src/leomower/scripts/best_model_resnet18_free_blocked.pth'
        rospy.loginfo("Loading %s", state_dict_path)

        # prepare the model based on resnet18 and load the state dic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(512, 2)
        self.model.load_state_dict(torch.load(state_dict_path, map_location=self.device))
        self.model.eval()

        # set the transormation that will be applied on the infered image aat runtime
        self.transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

        # need cv_bridge to convert ROS Image to torchvision image (PIL image)
        self.bridge = CvBridge()

        # subscribe to image topic
        rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

        # Create ROS publisher
        self.publisher = rospy.Publisher("collision", String, queue_size=10)

        # the inference will be done on a frequence lower than the image are published on the topic
        # otherwise the topic will be flooded
        self.timer = rospy.Timer(rospy.Duration(self.infer_freq), self.timer_callback)

    # image pre-processor function
    def preprocess_image(self, image):
        cv_image =  self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        image = PIL.Image.fromarray(cv_image)
        return image        

    # image inference fonction
    def predict_image(self, image):
        # prepare the image
        image_tensor = self.transforms(image).float()
        image_tensor = image_tensor.unsqueeze(0)
        input = Variable(image_tensor)
        input = input.to(self.device)

        # pass the mage into the model and get output tensor
        y = self.model(input)
        y = F.softmax(y, dim=1)
        y = y.flatten()

        # the model has 2 labels, the first one is the "blocked" probablity
        prob_blocked = float(y[0])
        return prob_blocked  

    # the image callback fonction
    # images are published on a rate too high for inference on a raspberry pi
    # so we put the last received image in "cache"
    # this image will be used by the timer callback
    def image_callback(self, image):
        self.lastImage = image

    # the timer callback
    # triggered on a self.infer_freq rate
    # process the last image 
    def timer_callback(self, timer):

        rospy.loginfo(str(datetime.now()) + '> I got an image')
        # pre-process the images
        preprocessedimage = self.preprocess_image(self.lastImage)

        rospy.loginfo(str(datetime.now()) + "> Image pre-processed")
        # infer
        prob_blocked = self.predict_image(preprocessedimage)
        
        # determine if the rover is blocked depending on the threshold
        if prob_blocked < self.blocked_threshold:
            collision = 'free'
        else:
            collision = 'blocked'

        rospy.loginfo(str(datetime.now()) + f"> Blocked probability {prob_blocked} -> {collision}")
        # publsh the collision status
        self.publisher.publish(collision)



def main():
    # init the node
    rospy.init_node('infer', anonymous=True)
    Infer()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    main()
