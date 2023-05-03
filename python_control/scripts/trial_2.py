#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState

def joint_callback(data):
	pub_msg = JointState()
	#pub_msg.header = Header()
	pub_msg.name=data.name
	pub_position = data.position
	#print(pub_msg.name)
	print(type(pub_position[0]))
  
def listener():
    # run simultaneously.
	rospy.init_node('listener_joint_states', anonymous=True)
	p=rospy.Subscriber("/robot2/joint_states", JointState,joint_callback)
	
	#joint_state = JointState()
	#print(joint_state)
	rospy.spin()

if __name__ == '__main__':
	listener()




	


