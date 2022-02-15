#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

# ROS node that drives to rover according to the collision status published to the collision topic
# when the status is "free" the rover goes straight
# when the status is blocked the rove spin
# the timer ensure that the Twist is published to the cmd_vel topic regularly, this will 
# ensure a smooth move
class Drive():

    def __init__(self):

        rospy.loginfo("Starting driver")

        # set the rover speed (m/s)
        self.speed = 0.2

        # prepare a Twist message
        self.twist = Twist()
        
        # subscribe to /collision topic
        rospy.Subscriber('/leomower/collision', String, self.collision_callback)

        # Create ROS publisher
        self.publisher = rospy.Publisher("cmd_vel", Twist, queue_size=1)

        # setup the timer that publishes Twist on a regular rate
        self.timer = rospy.Timer(rospy.Duration(0.2), self.timer_callback)


    # set the Twist according to the collision status
    # the Twist is "saved" in the class property and will be used by the timer
    def collision_callback(self, collision):
        
        if (collision.data == "free"):
            self.twist.linear.x = self.speed
            self.twist.angular.z = 0

        elif (collision.data == "blocked"):
            self.twist.linear.x = 0
            self.twist.angular.z = self.speed

        rospy.loginfo(self.twist)

    # publish on a regular rate the calculated Twist
    def timer_callback(self, timer):
        self.publisher.publish(self.twist)



def main():
    # init the node
    rospy.init_node('drive', anonymous=True)
    Drive()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    main()
