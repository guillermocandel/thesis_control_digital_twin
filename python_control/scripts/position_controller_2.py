#!/usr/bin/env python

import numpy as np
import scipy.linalg
from scipy.linalg import block_diag
from numpy.linalg import multi_dot
#import matplotlib.pyplot as plt


import rospy
import math
from std_msgs.msg import Float64
from math import sin,cos,atan2,sqrt,fabs


def M(q2):
	go = 9.85;
	m1 = 1.2
	m2 = 1;
	Io1 = 0.225;
	Io2 = 0.1875;
	l1 =1 ;
	l2 =1;

	M_11 = Io1 + m1*l1**2/4 + Io2+ m2*l2**2/4 +m2*l1**2+ m2*l1*l2*np.cos(q2) 
	M_12 = Io2 + m2*l2**2/4 +m2*l1*l2*np.cos(q2)/2
	M_22 = Io2 + m2*l2**2/4
	M=np.array([[M_11, M_12],[M_12, M_22]])
	return(M)


#Damping Matrix 2Dof


def D(q2,q1_dot,q2_dot):
	go = 9.85;
	m1 = 1.2
	m2 = 1;
	Io1 = 0.225;
	Io2 = 0.1875;
	l1 =1 ;
	l2 =1;
	D_11 = -(m2*l1*l2*np.sin(q2)/2)*q2_dot
	D_12= -(m2*l1*l2*np.sin(q2)/2)*(q1_dot+q2_dot)
	D_21 = (m2*l1*l2*np.sin(q2)/2)*q1_dot
	D_22 = 0;
	D =np.array([[D_11, D_12],[D_21, D_22]])
	return(D)
#gravity matrix 


def g(q1,q2):
	g_11=m1*go*l1*np.cos(q1)/2 + m2*go*l1*np.cos(q1) + m2*go*l2*np.cos(q1+q2)/2
	g_22=m2*go*l1*np.cos(q1+q2)
	g =np.array([[g_11],[g_22]])
	return(g)

 #Jacobian Matrix

 
def J(q,l1,l2):
 	J_11 = np.array(-l1*np.sin(q[0])-l2*np.sin(q[0]+q[1]))
 	J_12 = np.array(-l2*np.sin(q[0]+q[1]))
 	J_21 = np.array(l1*np.cos(q[0])+ l2*np.cos(q[0]+q[1]))
 	J_22 = np.array(l2*np.cos(q[0]+q[1]))
 	J = np.array([[J_11, J_12],[J_21, J_22]])
 	return(J) 
 
 #Derivative Jacobian matrix q_

	
def J_q_dot(q,l2):
	J_q_dot_1 = np.array([[-l2*np.cos(q[0]+q[1])-l1*np.cos(q[0]), -l2*np.cos(q[0]+q[1])],[-l2*np.sin(q[0]+q[1])-l1*np.sin(q[0]), -l2*np.sin(q[0]+q[1])]])
	J_q_dot_2 =np.array([[-l2*np.cos(q[0]+q[1]),-l2*np.cos(q[0]+q[1])],[-l2*np.sin(q[0]+q[1]),-l2*np.sin(q[0]+q[1])]])
	J_q_dot =np.hstack((J_q_dot_1,J_q_dot_2))
	return(J_q_dot)
 
#Derivatie Jacobian matrix a_k
	
def J_ak_dot(q,l1,l2):
	J_ak_dot_1= np.array([[-np.sin(q[0]), 0],[np.cos(q[0]),0]])
	J_ak_dot_2= np.array([[-np.sin(q[0]+q[1]), -np.sin(q[0]+q[1])],[np.cos(q[0]+q[1]),np.cos(q[0]+q[1])]])
	J_ak_dot =np.hstack((J_ak_dot_1,J_ak_dot_2))
	return(J_ak_dot)
#positon planar geometry
def pos(q,l1,l2):
	pos_x = l1*np.cos(q[0]) + l2*np.cos(q[0]+q[1])
	pos_y =l1*np.sin(q[0]) + l2*np.sin(q[0]+q[1])
	pos = np.array([[pos_x],[pos_y]])
	return(pos)
#Kinematic regressor
def Y_k(q,q_dot):
	Y_k_11 = -q_dot[0]*np.sin(q[0])
	Y_k_12 = -np.sin(q[0]+q[1])*(q_dot[0]+q_dot[1])
	Y_k_21 = q_dot[0]*np.cos(q[0])
	Y_k_22 = np.cos(q[0]+q[1])*(q_dot[0]+q_dot[1])
	Y_k = np.array([[Y_k_11, Y_k_12],[Y_k_21, Y_k_22]])
	return(Y_k)
def vec_Y_k(q,q_dot):
	Y_k_11 = -q_dot[0]*np.sin(q[0])
	Y_k_12 = -np.sin(q[0]+q[1])*(q_dot[0]+q_dot[1])
	Y_k_21 = q_dot[0]*np.cos(q[0])
	Y_k_22 = np.cos(q[0]+q[1])*(q_dot[0]+q_dot[1])
	vec_Y_k = np.array([[Y_k_11],[Y_k_12],[Y_k_21],[Y_k_22]])
	return(vec_Y_k)
	
	
def Y_d(q,q_dot,psi,psi_dot):
	Y_d=np.dot(M(q[1]),psi_dot) + np.dot(D(q[1],q_dot[0],q_dot[1]),psi)-g(q[0],q[1])
	Y_c = np.dot(M(q[1]),psi_dot)
	return(Y_d)
	




def position_controller(pos_2,x_ref,q,q_dot,Z,chi,a_z,a_d,a_k,psi,psi_dot,v,x2):
	#pos_2 = #from controller
	pos_2= pos_2.reshape(2,1)

	e = pos_2-x_ref[:,0].reshape(2,1);
	a_z_dot = Lambda_z*np.dot((Z.reshape(2,1)).T,(chi.reshape(2,1)-2*e.reshape(2,1)));
	a_z = a_z.reshape(1,1)+1/fs*a_z_dot.reshape(1,1);
	
	a_k_dot = -Lambda_k*np.dot(Y_k(q[:,0],q_dot[:,0]).T,(chi.reshape(2,1)-2*e.reshape(2,1)));
	#print(a_k_dot)
	a_k =a_k.reshape(2,1)+1/fs*a_k_dot.reshape(2,1)
	
	#print(Y_d(q,q_dot,psi.reshape(2,1),psi_dot.reshape(2,1)))
	a_d_dot= -Lambda_d*np.dot(Y_d(q[:,0],q_dot[:,0],psi.reshape(2,1),psi_dot.reshape(2,1)).T,(q_dot.reshape(2,1)-psi.reshape(2,1)));
	a_d = a_d.reshape(1,1)+a_d_dot.reshape(1,1)/fs
	
	v_dot = np.dot(M_1,v.reshape(8,1)) + np.dot(N,vec_Y_k(q[:,0],q_dot[:,0]));
	v = v.reshape(8,1)+v_dot/fs
	v_1= v[0:4,:].reshape(4,1);
	v_2 = v[4:8,:].reshape(4,1);
	mat_v = np.hstack((v_1 ,v_2)) 
	Z = multi_dot([Gamma,mat_v,a_k.reshape(2,1)])
	v_1= v_dot[0:4,:].reshape(4,1);
	v_2 = v_dot[4:8,:].reshape(4,1);
	mat_v_dot = np.hstack((v_1 ,v_2)) 
	Z_dot = np.dot(Gamma,np.dot(mat_v_dot,a_k.reshape(2,1)))
	chi_dot = np.dot(J(q[:,0],a_k[0,0],a_k[1,0]),q_dot.reshape(2,1))-np.dot(Z.reshape(2,1),a_z[0,0].reshape(1,1))- k0*(chi.reshape(2,1)-e.reshape(2,1))
	chi = chi.reshape(2,1)+1/fs*chi_dot;
	psi= np.dot(np.linalg.pinv(J(q[:,0],a_k[0,0],a_k[1,0])),(Z.reshape(2,1)*a_z.reshape(1,1)-k1*chi.reshape(2,1)));
	z_2 = -k2*(q_dot.reshape(2,1)-psi.reshape(2,1))-np.dot(J(q[:,0],a_k[0,0],a_k[1,0]).T,e.reshape(2,1))+np.dot(Y_d(q[:,0],q_dot[:,0],psi.reshape(2,1),psi_dot.reshape(2,1)),a_d.reshape(1,1));
	tau_input =z_2
	dx2 = np.dot(np.linalg.inv(M(q[1,0])),(np.dot(-D(q[1,0],q_dot[0,0],q_dot[1,0]),(q_dot.reshape(2,1)))-g(q[0,0],q[1,0])+tau_input.reshape(2,1)))-psi_dot.reshape(2,1)
	x2 = x2.reshape(2,1)+1/fs*dx2 # euler method
	
	q_dot_2 = np.array([q_dot[0,:] ,q_dot[0,:] ,q_dot[1,:], q_dot[1,:]]).reshape(4,1)
	dv_v= multi_dot([np.linalg.pinv(J(q[:,0],a_k[0,0],a_k[1,0])),Z_dot.reshape(2,1),a_z.reshape(1,1)]);
	dv_chi= multi_dot([-np.linalg.pinv(J(q[:,0],a_k[0,0],a_k[1,0])),k1*chi_dot.reshape(2,1)]);
	dv_q_dot_2 =np.linalg.pinv(np.dot((J_q_dot(q[:,0],a_k[1,0])),q_dot_2))
	dv_q_dot_3 =np.dot(Z.reshape(2,1),a_z.reshape(1,1))-k1*chi.reshape(2,1);
	dv_q_dot = np.dot(dv_q_dot_2,dv_q_dot_3);
	a_k_dot_2 = np.array([a_k_dot[0,0],a_k_dot[0,0],a_k_dot[1,0],a_k_dot[1,0]]).reshape(4,1)
	dv_ak_dot = np.dot(np.linalg.pinv(np.dot(J_ak_dot(q[:,0],a_k[0,0],a_k[1,0]),a_k_dot_2)),dv_q_dot_3);
	dv_az_dot = np.dot(np.linalg.pinv(J(q[:,0],a_k[0,0],a_k[1,0])),np.dot(Z.reshape(2,1),a_z_dot.reshape(1,1)))
	psi_dot = dv_chi+ dv_az_dot+dv_q_dot+dv_v+dv_ak_dot
	
	q_dot = x2.reshape(2,1) + psi.reshape(2,1);
	q = q.reshape(2,1)+q_dot.reshape(2,1)/fs;	
	return(e,q,q_dot,Z,chi,a_z,a_d,a_k,psi,psi_dot,v,x2)
	
	

#Define a RRBot joint positions publisher for joint controllers.
def robot2_joint_positions_publisher():

	#Initiate node for controlling joint1 and joint2 positions.
	rospy.init_node('joint_positions_node', anonymous=True)
	#rospy.init_node('joint_states_2', anonymous=True)
	#rospy.Subscriber('joint_states', JointState,joint_states_callback)	
	#Define publishers for each joint position controller commands.
	pub1 = rospy.Publisher('/robot2/joint2_position_controller/command', Float64, queue_size=10)
	pub2 = rospy.Publisher('/robot2/joint3_position_controller/command', Float64, queue_size=10)

	rate = rospy.Rate(100) #100 Hz

	#While loop to have joints follow a certain position, while rospy is not shutdown.
	i = 1
	while not rospy.is_shutdown():
		#hello_str = "hello world %s" % rospy.get_time()
		position = math.pi/2
		#rospy.loginfo(position)
		#Have each joint follow a sine movement of sin(i/100).
		sine_movement = sin(i/100)
		print(sine_movement)

		#Publish the same sine movement to each joint.
		pub1.publish(sine_movement)
		pub2.publish(sine_movement)

		i = i+1 #increment i

		rate.sleep() #sleep for rest of rospy.Rate(100)


#Main section of code that will continuously run unless rospy receives interuption (ie CTRL+C)
if __name__ == '__main__':
	try: robot2_joint_positions_publisher()
	except rospy.ROSInterruptException: pass






	


