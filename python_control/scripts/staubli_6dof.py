#!/usr/bin/env python

import numpy as np
import rospy
import math
from std_msgs.msg import Float64
from math import sin,cos,atan2,sqrt,fabs
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_msgs.msg import Header
import numpy as np

	
	
	

class Controller:
	def __init__(self):
		# controller variables
		rospy.on_shutdown(self.shutdown)
		self.t=0
		self.pub_msg = JointState()
		self.sending_pos_1 = Float64()
		self.sending_pos_2 = Float64()
		self.pub_msg.position = [0,0,0,0,0,0]
		self.rate=rospy.Rate(100)
		self.L1= 0.425
		self.L2= 0.525
		self.kp=0.001
		self.publisher_2=rospy.Publisher('/staubli_tx90_robot/joint2_position_controller/command', Float64, queue_size=10)
		self.publisher_3=rospy.Publisher('/staubli_tx90_robot/joint3_position_controller/command', Float64, queue_size=10)
		rospy.Subscriber('/staubli_tx90_robot/joint_states', JointState, callback=self.joint_state_callback)
		self.rate.sleep()
		
		
	
	def XJ(self):
		#kinmetatics # L length
		#q joint angle , position base 
		L1= self.L1 ;  L2=self.L2; q1 =self.pub_msg.position[1];q2=self.pub_msg.position[2];
		P11 = L1*cos(q1) + L2*cos(q1+q2);
		P21 = L1*sin(q1)+L2*sin(q1+q2);

		Position = np.array([[P11],[P21]]);
		Real_position = Position
		J_11 = -L1*sin(q1)-L2*sin(q1+q2)
		J_12 = -L2*sin(q1+q2)
		J_21 = L1*cos(q1)+ L2*cos(q1+q2)
		J_22 = L2*cos(q1+q2)
		J = np.array([[J_11, J_12],[J_21, J_22]])
		self.real_position=Real_position
		self.J=J
	
	def joint_state_callback(self,data):
		self.pub_msg.name=data.name
		self.pub_msg.position= data.position
		pub_2= self.pub_msg.position[1]
		pub_3= self.pub_msg.position[2]
		self.t=self.t+0.01
		pub = np.array([[pub_2],[pub_3]])
		trajectory = np.array([[0.3+0.1*cos(self.t/100)],[0.4*cos(self.t/100)]])
		self.XJ()
		error=trajectory-self.real_position
		rospy.loginfo(error)
		position = self.kp*error+pub
		self.sending_pos_1.data = position[0,0]
		self.sending_pos_2.data = position[1,0]
		self.publisher_2.publish(self.sending_pos_1)
		self.publisher_3.publish(self.sending_pos_2)
		

	def shutdown(self):
		rospy.loginfo('Shutting Down')
		rospy.sleep(1)

  
  
if __name__ == '__main__':
    try:
    	rospy.init_node('trajectory_controller')
    	Controller()
    	rospy.spin()
    except rospy.ROSInterruptException:
    	rospy.loginfo("Controller terminated.")
    	pass	

  
# Load the PID gains from the YAML file






	
'''
q=np.array([[1],[2],[1],[1],[1],[1]]);

print((C_6dof(q,q)))

def robust(J):
  U,S,V=svd(J);
  Z=svd(J,compute_uv=False);
  lambda_max=1; delta=10,
  output=np.zeros((6,3));
  for i in range(len(Z)):
    lambda_g = lambda_max*math.exp(-(Z[i]/delta)**2);
    output_2 = output+Z[i]/(Z[i]**2+lambda_g**2)*np.dot(V[:,i].reshape(6,1),U[:,i].reshape(1,3))
  return(output_2)


def eetrack(t,x,e):
	#print(t)
	dx = np.zeros((146,1),dtype = np.float64);
	go = 9.8 #cm/s2
	#reference        
	a=np.ones((6,1));
	L= np.array([[0.05],[0.425],[0.05],[0.425],[0.1]])
	base =np.array([[0],[0],[0.478]]);
	q = x[0:6].reshape(6,1); q1 =q[0,0];q2 = q[1,0];
	dq = x[6:12].reshape(6,1); dq1 =dq[0,0];dq2 = dq[1,0];xi =dq;
	eta= x[12:42].reshape(30,1)
	zeta = x[42:45].reshape(3,1)
	est_ak = x[45:50].reshape(5,1)
	est_az = x[50:140].reshape(90,1)
	est_ad = x[140:146].reshape(6,1)
	base =np.array([[0],[0],[0.478]]);

	#if t< 100:
		#Lambda_k = 100; Lambda_z =100; Lambda_d=100;
	#else:
	Lambda_k = 1; Lambda_z =1000; Lambda_d=0.1;
	x,vel = XJ_6dof(L,q,dq,base);
	#e = x-x_ref;
	#print(e)

	Yk_2 = Yk_6dof(q,dq);
	est_J = Jacobian_6dof(est_ak,q);
	est_ak_update = -Lambda_k*np.dot(Yk_2.T,(zeta-2*e));
	d_est_J = dJhat_6dof(est_ak,est_ak_update,q,dq);
	
	H=robust(est_J);
	##IM 
	#if t<100:
		#M_3 = np.array([[0,1],[-4,-2]])
	#else:
	M_3 = np.array([[0,1],[-1,-1.414]])

	I_2 = np.identity(3);	
	M_2 = block_diag(M_3,M_3,M_3,M_3,M_3)
	M_1 = np.kron(M_2,I_2);
	N_3 = np.array([[0],[1]])	
	N_2 = block_diag(N_3,N_3,N_3,N_3,N_3)
	N = np.kron(N_2,I_2);

	im = np.dot(M_1,eta) + np.dot(N,Yk_2.reshape(15,1));
	Z = np.kron(eta.T,I_2);
	d_Z = np.kron(im.T,I_2);

	##filter
	k_zeta = 30;
	filter_2= np.dot(est_J,xi)-np.dot(Z,est_az) - np.dot(k_zeta,(zeta-e));
	est_az_update= Lambda_z*np.dot(Z.T,(zeta-2*e));

	##input
	k11 = 10; k22= 10;
	xi_r = np.dot(H,(-k11*zeta+ np.dot(Z,est_az)));
	d_inv_J = -np.dot(np.dot(H,d_est_J),H);

	dxi_r = np.dot(d_inv_J,(-k11*zeta+np.dot(Z,est_az)))+np.dot(H,(-k11*filter_2 + np.dot(Z,est_az_update) + np.dot(d_Z,est_az)));
	Y = Yd_6dof(q,dq,dxi_r,xi_r)
	est_ad_update = -Lambda_d*np.dot(Y.T,(xi-xi_r));
	u = -k22*(xi-xi_r) - np.dot(est_J.T,e) + np.dot(Y,est_ad);
	
	dx[0:12,] = EL2_6dof(a,q,dq,u);
	dx[12:42,]= im;
	dx[42:45,]=filter_2;
	dx[45:50,]=est_ak_update;
	dx[50:140,]=est_az_update;
	dx[140:146,]=est_ad_update;
	dx = dx.reshape(146,)
	return(dx)


x0=np.array([[3.087],[1.54],[-0.6]]);
x_2=np.zeros((3,1));
x0=np.vstack((x0,x_2));
x=np.zeros((39,1));
x0=np.vstack((x0,x));
x=np.array([[0.055],[0.42],[0.054],[0.42],[0.105]]);
x0=np.vstack((x0,x));
x=np.zeros((90,1)); x0=np.vstack((x0,x));
x=np.zeros((6,1));x0=np.vstack((x0,x));
print(x0.shape)

t = np.arange(0,10,1/20)
dt=1/20
i=0
L= np.array([[0.05],[0.425],[0.05],[0.425],[0.1]])
x =scipy.integrate.ode(eetrack).set_integrator('vode',method = 'bdf',order=15)
x.set_initial_value(x0,0)
base=np.array([[0],[0],[0.478]]);
while i<10:
	x_ref = np.array([[0.8*np.cos(np.pi*i/10)],[0.8*np.sin(np.pi*i/10)],[0.2+0.6*np.cos(np.pi*i/10)]],dtype = np.float64)
	#x1,vel = XJ(L,x_real_2.reshape(2,1),(np.array([0,0])).reshape(2,1),base)
	x1,vel = XJ_6dof(L,x0[0:6,].reshape(6,1),x0[6:12,].reshape(6,1),base);
	#print(x1)
	e = x1-x_ref;	
	print(e)
	x.set_f_params(e)
		#print(e)
	x0=x.integrate(x.t+dt)
	#print(x0)
	assert x.successful()
	print(i)
	i=i+dt
#eetrack(0,x0,0)	
'''

