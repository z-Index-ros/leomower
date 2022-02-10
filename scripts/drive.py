#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class Drive():

    def __init__(self):

        rospy.loginfo("Starting driver")
        # subscribe to /collision topic
        rospy.Subscriber('collision', String, self.callback)

        # Create ROS publisher
        self.publisher = rospy.Publisher("cmd_vel", Twist, queue_size=1)


    def callback(self, collision):
        twist = Twist()
        if (collision.data == "free"):
            twist.linear.x = 0.5
            twist.angular.z = 0

        if (collision.data == "blocked"):
            twist.linear.x = 0
            twist.angular.z = 1

        rospy.loginfo(twist)

        self.publisher.publish(twist)

def main():
    # init the node
    rospy.init_node('drive', anonymous=True)
    Drive()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    main()
