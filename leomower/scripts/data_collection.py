import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import os
import cv2
from uuid import uuid1
from cv_bridge import CvBridge

blocked_dir = 'dataset/blocked'
free_dir = 'dataset/free'

class DataCollection():

    def __init__(self):
        rospy.loginfo("Starting node")
        rospy.Subscriber('/camera/image_raw', Image, self.imagecallback)
        rospy.Subscriber('/leomower/teleop_key', String, self.teleopcallback)
        rospy.loginfo("Start working in %s", os.getcwd())
        try:
            os.makedirs(free_dir)
            os.makedirs(blocked_dir)
            rospy.loginfo('Dataset Directories created : "%s" and "%s"', free_dir, blocked_dir)
        except FileExistsError:
            rospy.loginfo('Directories not created because they already exist')

    def save_image(self, directory):
        rospy.loginfo("Saving image to %s", directory)
        image_path = os.path.join(directory, str(uuid1()) + '.jpg')
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(self.lastImage, desired_encoding='passthrough')
        cv2.imwrite(image_path, cv_image)
        rospy.loginfo("%d free / %d blocked", len(os.listdir(free_dir)), len(os.listdir(blocked_dir)))


    def imagecallback(self, image):
        self.lastImage = image

    def teleopcallback(self, key):
        if key.data == 'f':
            self.save_image(free_dir)
        elif key.data == 'b':
            self.save_image(blocked_dir)
        else:
            rospy.loginfo("Nothing to do %s (press f(ree) of b(locked))", key.data)
        

def main():
    # init the node
    rospy.init_node('data_collection', anonymous=True)
    DataCollection()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    main()