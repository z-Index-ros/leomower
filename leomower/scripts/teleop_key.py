import rospy
from std_msgs.msg import String
import readkeys


class TeleOp():

    def __init__(self):
        rospy.loginfo("Starting node")
        self.pub = rospy.Publisher('/leomower/teleop_key', String, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.time_callback)


    def time_callback(self, timer):
        key = readkeys.getch()
        if key == 'q' : 
            rospy.loginfo("Shuting down node")
            rospy.signal_shutdown("shutdown")

        rospy.loginfo("Key pressed : %s", key)
        msg = String()
        msg.data = key
        self.pub.publish(msg)



def main():
    # init the node
    rospy.init_node('teleop', anonymous=True)
    TeleOp()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    main()