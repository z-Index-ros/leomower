#!/usr/bin/env python

import rospy
from std_msgs.msg import String

respond = rospy.get_param('listener/respond', 'mate')


def callback(data):

    response =f"{rospy.get_caller_id()} - {data.data} {respond}"
    #response = str(rospy.get_param_names())
    pub = rospy.Publisher('listener', String, queue_size=10)
    rospy.loginfo(response)
    pub.publish(response)


def listener():
    
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber('chatter', String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
