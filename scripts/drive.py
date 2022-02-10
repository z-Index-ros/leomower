#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class Drive():

    def __init__(self):

        rospy.loginfo("Starting driver")
        
        self.twist = Twist()
        
        # subscribe to /collision topic
        rospy.Subscriber('collision', String, self.callback)

        # Create ROS publisher
        self.publisher = rospy.Publisher("cmd_vel", Twist, queue_size=1)



    def callback(self, collision):
        
        if (collision.data == "free"):
            self.twist.linear.x = 0.5
            self.twist.angular.z = 0

        elif (collision.data == "blocked"):
            self.twist.linear.x = 0
            self.twist.angular.z = 1

        rospy.loginfo(self.twist)

        self.publisher.publish(self.twist)

def main():
    # init the node
    rospy.init_node('drive', anonymous=True)
    Drive()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    main()
