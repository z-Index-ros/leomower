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


class Infer():

    def __init__(self):

        rospy.loginfo("Start working in %s", os.getcwd())
        modelPath = 'src/leomower/scripts/best_model_resnet18_free_blocked.pth'
        rospy.loginfo("Loading %s", modelPath)

        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, 2)
        model.load_state_dict(torch.load(modelPath))

        """ self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)
        model = model.eval().half()

        if torch.cuda.is_available():
            self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
            self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()
        else:
            self.mean = torch.Tensor([0.485, 0.456, 0.406]).half()
            self.std = torch.Tensor([0.229, 0.224, 0.225]).half()

        normalize = torchvision.transforms.Normalize(self.mean, self.std) """

        # subscribe to image topic
        rospy.Subscriber('/camera/image_raw', Image, self.callback)

        # Create ROS publisher
        self.publisher = rospy.Publisher("collision", String, queue_size=1)

    def preprocess(self, image):
        #img_arr = np.frombuffer(image.data, dtype=np.uint8)#.reshape(image.height, image.width, -1)
        #image = PIL.Image.fromarray(image.data).convert('RGB')

        bridge = CvBridge()
        cv_image =  bridge.imgmsg_to_cv2(self.lastImage, desired_encoding='passthrough')
		
        #image = transforms.functional.to_tensor(img_arr).to(self.device).half()
		#img=torch.tensor(np.array(img,dtype=np.float64))/255.0


        image = PIL.Image.fromarray(cv_image)
                
        preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

        img_preprocessed = preprocess(image)
        batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)   
        return batch_img_tensor     

    def callback(self, image):

        x = self.preprocess(image)
        y = self.model(x)
        y = F.softmax(y, dim=1)

        prob_blocked = float(y.flatten()[0])
        #rospy.loginfo("Blocked probability %f" % prob_blocked)
        
        if prob_blocked < 0.5:
            collision = 'free'
        else:
            collision = 'blocked'

        #rospy.loginfo("Blocked probability %f -> %s" % prob_blocked, collision)

        self.publisher.publish(collision)

def main():
    # init the node
    rospy.init_node('infer', anonymous=True)
    Infer()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    main()
