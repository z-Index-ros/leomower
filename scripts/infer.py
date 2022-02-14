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


class Infer():

    def __init__(self):

        rospy.loginfo("Start working in %s", os.getcwd())

        self.blocked_threshold = 0.05

        modelPath = 'src/leomower/scripts/best_model_resnet18_free_blocked.pth'
        rospy.loginfo("Loading %s", modelPath)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(512, 2)
        self.model.load_state_dict(torch.load(modelPath, map_location=self.device))
        self.model.eval()

        self.transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        self.bridge = CvBridge()

        # subscribe to image topic
        rospy.Subscriber('/camera/image_raw', Image, self.callback)

        # Create ROS publisher
        self.publisher = rospy.Publisher("collision", String, queue_size=1)

    def preprocess_image(self, image):

        cv_image =  self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        image = PIL.Image.fromarray(cv_image)
        return image        

    def predict_image(self, image):
        image_tensor = self.transforms(image).float()
        image_tensor = image_tensor.unsqueeze(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        y = self.model(input)
        y = F.softmax(y, dim=1)
        y = y.flatten()
        prob_blocked = float(y[0])
        return prob_blocked  

    def callback(self, image):

        rospy.loginfo(str(datetime.now()) + '> I got an image')

        preprocessedimage = self.preprocess_image(image)

        rospy.loginfo(str(datetime.now()) + "> Image pre-processed")

        prob_blocked = self.predict_image(preprocessedimage)

        
        if prob_blocked < self.blocked_threshold:
            collision = 'free'
        else:
            collision = 'blocked'

        rospy.loginfo(str(datetime.now()) + f"> Blocked probability {prob_blocked} -> {collision}")

        self.publisher.publish(collision)

def main():
    # init the node
    rospy.init_node('infer', anonymous=True)
    Infer()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    main()
