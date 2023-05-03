#!/usr/bin/env python
import numpy as np
import scipy.linalg
import tf
from scipy.linalg import block_diag,svd
from numpy.linalg import multi_dot,inv
from scipy.integrate import solve_ivp,odeint
#import matplotlib.pyplot as plt
import rospy
import math
from std_msgs.msg import Float64
from math import sin,cos,atan2,sqrt,fabs
import matplotlib.pyplot as plt
from sensor_msgs.msg import JointState
from controller_manager_msgs.srv import SwitchController
import threading

def XJ(L,q,dq,base):
	#kinmetatics # L length
	#q joint angle , position base 
	L1= L[0,0]; L2 = L[1,0]; q1 = q[0,0];q2=q[1,0];
	P11 = L1*cos(q1) + L2*cos(q1+q2);
	P21 = L1*sin(q1)+L2*sin(q1+q2);
	
	Position = np.array([[P11],[P21]]);
	Real_position = Position + base; 
	J_11 = -L1*sin(q1)-L2*sin(q1+q2)
	J_12 = -L2*sin(q1+q2)
	J_21 = L1*cos(q1)+ L2*cos(q1+q2)
	J_22 = L2*cos(q1+q2)
	J = np.array([[J_11, J_12],[J_21, J_22]])
	velocity = np.dot(J,dq)
	return(Real_position,velocity)
	
def robust(J):
	U,S,V=svd(J,full_matrices=False,lapack_driver='gesvd');
	V=np.transpose(V)
	P,C,A=np.linalg.svd(J)
	Z=svd(J,compute_uv=False,);
	lambda_max=0.01; delta=0.1,
	output_2=np.zeros((2,2));
	for i in range(len(Z)):
		lambda_g = lambda_max*math.exp(-(Z[i]/delta)**2);
		output_2 = output_2+Z[i]/(Z[i]**2+lambda_g**2)*np.dot(V[:,i].reshape(2,1),U[:,i].reshape(1,2))
	return(output_2)

def Jacobian(L,q):
	L1= L[0,0]; L2 = L[1,0]; q1 = q[0,0];q2=q[1,0];
	J_11 = -L1*sin(q1)-L2*sin(q1+q2)
	J_12 = -L2*sin(q1+q2)
	J_21 = L1*cos(q1)+ L2*cos(q1+q2)
	J_22 = L2*cos(q1+q2)
	J = np.array([[J_11, J_12],[J_21, J_22]])
	return(J)
	
def Yd(position,velocity,x,y):
	q0 = position;q1 = position[0,0];q2 = position[1,0];
	dq = velocity;dq1 =dq[0,0];dq2 = dq[1,0];
	x1= x[0,0]; x2= x[1,0];
	y1 =y[0,0];y2=y[1,0];
	#Y13 = (2*x1+x2)*cos(q2)-(dq2*y1+ dq1*y2 +dq2*y2)*sin(q2);
	#Y23 = x1*cos(q2)+ dq1*y1*sin(q2);#
	Y13 = -sin(q2)*(2*x1*x2+x2*x2)
	Y23=-sin(q2)*(dq1*dq2)
	output = np.array([[x1,x2,Y13,cos(q1)*1,cos(q1+q2)*1],[0,x1+x2,Y23,0,cos(q1+q2)*1]])
	output=output.reshape(2,5);

	return(output)
	
def Yk(q,dq):
	q1 = q[0,0]; q2 = q[1,0];
	dq1 = dq[0,0];dq2 = dq[1,0];
	Y_k_11 = -dq1*sin(q1);Y_k_12 = -sin(q1+q2)*(dq1+dq2);
	Y_k_21 = dq1*cos(q1);Y_k_22 =cos(q1+q2)*(dq1+dq2)
	Y_k = np.array([[Y_k_11, Y_k_12],[Y_k_21, Y_k_22]])
	return(Y_k)
	
def EL2(theta,position,velocity,torque):
	theta_1=theta[0,0];theta_2=theta[1,0];
	theta_3= theta[2,0]; theta_4=theta[3,0];
	theta_5=theta[4,0];
	q=position;q_1=q[0,0] ;q_2=q[1,0];
	dq =velocity;dq_1 = dq[0,0];dq_2 =dq[1,0];
	
	M = np.array([[theta_1+ 2*theta_3*cos(q_2),theta_2 +theta_3*cos(q_2)],[theta_2+theta_3*cos(q_2),theta_2]]);
	C = np.array([[-theta_3*sin(q_2)*dq_2,-theta_3*sin(q_2)*(dq_1+dq_2)],[theta_3*sin(q_2)*dq_1,0]]);
	g = np.array([[theta_4*cos(q_1)+theta_5*cos(q_1+q_2)],[theta_5*cos(q_1+q_2)]]);
	output_2 = np.dot(inv(M),(torque - np.dot(C,dq) -np.dot(g,1)))
	velocity =velocity.reshape(2,1)
	output = np.vstack((velocity,output_2))
	return(output)

def dJhat(Lhat,dLhat,q,dq):
	Lhat1 = Lhat[0,0]; Lhat2 = Lhat[1,0];
	dLhat1 = dLhat[0,0]; dLhat2 = dLhat[1,0];
	q1 = q[0,0]; q2 = q[1,0];
	dq1 = dq[0,0];dq2 = dq[1,0];
	
	
	output_1 = np.array([[-dLhat1*sin(q1)-dLhat2*sin(q1+q2),-dLhat2*sin(q1+q2)],[dLhat1*cos(q1)+dLhat2*cos(q1+q2), dLhat2*cos(q1+q2)]])
	#recuerda el signo menos ahi)
	output_2= np.array([[-Lhat1*cos(q1)*dq1-Lhat2*cos(q1+q2)*(dq1+dq2),-Lhat2*cos(q1+q2)*(dq1+dq2)],[-Lhat1*sin(q1)*dq1-Lhat2*sin(q1+q2)*(dq1+dq2),-Lhat2*sin(q1+q2)*(dq1+dq2)]])	
	output = output_1 +output_2;
	return(output)


	
	
class Controller:
	def __init__(self):
		# controller variables
		rate=rospy.Rate(100)
		self.u = np.array([[0],[0]])
		self.boolean = 0
		self.boolean_2=0
		self.i=1
		
		
		# Prepare shutdown
		rospy.on_shutdown(self.shutdown)
		rospy.loginfo(self.boolean)
		self.pub_rate = 0.01

		# message type
		self.pub_msg = JointState()

		# Topic to be published to 
		self.joint_names = ['joint2' , 'joint3']
		self.pubs = [rospy.Publisher('/robot2/'+name+'_effort_controller/command', Float64, queue_size=10) for name in self.joint_names]
		rospy.Subscriber("/robot2/joint_states", JointState, callback = self.controller)
		rate.sleep()
		rospy.Timer(rospy.Duration(self.pub_rate), self.publish_effort)

  
   
   
	def switch_controller(self):
		if self.boolean ==0:
			rospy.wait_for_service('/robot2/controller_manager/switch_controller')
			try:
				self.boolean = 1
				switch_controller = rospy.ServiceProxy('/robot2/controller_manager/switch_controller', SwitchController)
				ret = switch_controller(['joint2_effort_controller','joint3_effort_controller'],['joint2_position_controller','joint3_position_controller'], 2,False,5)
				#ret = switch_controller(['joint2_position_controller','joint3_position_controller'],['joint2_effort_controller','joint3_effort_controller'], 2,True,1)
			except rospy.ServiceException:
				print ("Service call failed")
				self.boolean = 1
		else :
			self.boolean =1

	def publish_effort(self,data):
		#rospy.loginfo("this is self.u %s",self.u)
		self.switch_controller()
		for i,e in enumerate(self.u):
			#rospy.loginfo("this is effort uno %s",e)
			msg=Float64()
			msg.data=e
			self.pubs[i].publish(msg)
			
			
	def vel_publishing(self, data):
		#self.switch_controller()
		msg=Float64()
		msg.data=1
		rospy.loginfo("hey estoy mandando info")
		self.pub3.publish(msg)
		
		
	def eetrack(self,t,x):
		#rate = rospy.Rate(100) #100 Hz
		b=0.01
		dx = np.zeros((33,1),dtype = np.float64);
		q=self.pos
		dq=self.vel
		q = q.reshape(2,1)
		dq=dq.reshape(2,1)
		eta= x[0:8].reshape(8,1)
		zeta = x[8:10].reshape(2,1)
		est_ak = x[10:12].reshape(2,1)
		rospy.loginfo(est_ak)
		est_az = x[12:28].reshape(16,1)
		est_ad = x[28:33].reshape(5,1)
		dq = dq.reshape(2,1); #dq1 =dq[0,0];dq2 = dq[1,0];
		xi =dq;
		x_ref = np.array([[1.7-0.3*np.cos(np.pi*t/10)],[0.3*np.sin(np.pi*t/10)]],dtype = np.float64)
		Lambda_k = 100; Lambda_z =100; Lambda_d=100;
		L= np.array([[1.5],[1.5]])
		base=np.array([[0],[0]])
		x_2,vel = XJ(L,q.reshape(2,1),dq.reshape(2,1),base);
		e = x_2-x_ref;
		#rospy.loginfo(e)
		Yk_2 = Yk(q,dq);
		est_J = Jacobian(est_ak,q);
		est_ak_update = -Lambda_k*np.dot(Yk_2.T,(zeta-2*e));
		d_est_J = dJhat(est_ak,est_ak_update,q,dq);
		#H=robust(est_J);
		H=inv(est_J)
		M_3 = np.array([[0,1],[-1,-1.414]])

		I_2 = np.identity(2);	M_2 = block_diag(M_3,M_3);M_1 = np.kron(M_2,I_2);
		N_3 = np.array([[0],[1]]);N_2 = block_diag(N_3,N_3);N = np.kron(N_2,I_2);

		im = np.dot(M_1,eta) + np.dot(N,Yk_2.reshape(4,1));
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
		Y = Yd(q,dq,dxi_r,xi_r)
		est_ad_update = -Lambda_d*np.dot(Y.T,(xi-xi_r));
		#rate.sleep()
		u = -k22*(xi-xi_r) - np.dot(est_J.T,e) + np.dot(Y,est_ad);
		#rospy.loginfo('this is torque inside ODE %s',u)
		#rospy.loginfo('this is time inside ODE %s',t)
		self.u=u

		dx[0:8,]= im;
		dx[8:10,]=filter_2;
		dx[10:12,]=est_ak_update;
		dx[12:28,]=est_az_update;
		dx[28:33,]=est_ad_update;
		dx = dx.reshape(33,)
		#self.switch_controller()
		
		
		
		return(dx)      

	def controller(self, data):

		self.pub_msg.name=data.name
		self.pub_position = data.position
		pub_velocity = data.velocity
		vel_1= pub_velocity[0]
		vel_2= pub_velocity[1]
		self.vel=np.array([vel_1,vel_2])
		self.pos_1 =self.pub_position[0]
		self.pos_3 =self.pub_position[1]
		self.pos = np.array([self.pos_1,self.pos_3]);

			 
		b=0.01
		t0=0;tf=20;dt=0.01
		t_span=(t0,t0+10)
		rate = rospy.Rate(100) 
		time=np.arange(t0,tf+dt,dt)
		#sol=np.zeros((len(time),len(x0)))
		tspan=[time[self.i-1],time[self.i]]
		self.i = self.i+1	
		rate.sleep()
		if self.boolean_2 == 0:
			x0 = np.zeros((10,1))
			x4 = np.zeros((15,1));x5 = np.array([[1.4],[1.6]]);
			x6 = np.zeros((6,1))
			x0 = np.vstack((x0,x5))
			x0 = np.vstack((x0,x4));
			#x0 = np.vstack((x0,x4));
			x0=np.vstack((x0,x6));
			x0=x0.reshape(33,)
			self.sol = x0
			sol_i=solve_ivp(self.eetrack,tspan,self.sol,method='BDF')
			self.sol=sol_i.y[:,-1]
			self.boolean_2 = 1
		else :
			self.boolean_2=1
			sol_i=solve_ivp(self.eetrack,tspan,self.sol,method='BDF')
			self.sol=sol_i.y[:,-1]
		

	def shutdown(self):
		rospy.loginfo('Shutting Down')
		rospy.sleep(1)

		
			
if __name__ == '__main__':
    
    try:
        rospy.init_node('Controller', anonymous=False)
        # r = rospy.Rate(10)
        Controller()
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Controller terminated.")  
        pass	



	


