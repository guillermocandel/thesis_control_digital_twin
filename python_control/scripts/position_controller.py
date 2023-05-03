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

#from tf.transformation import quaternion_from_euler ,euler_from_quaternion 

b=0.01
L = np.array([[17.2*b],[6.43*b]])
base =np.array([[9.19*b],[0]]);

pos=None
velocity=None

#define known  variables of the dynamic equation
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
  U,S,V=svd(J);
  Z=svd(J,compute_uv=False);
  lambda_max=0.1; delta=0.5,
  output=np.zeros((2,2));
  for i in range(len(Z)):
    lambda_g = lambda_max*math.exp(-(Z[i]/delta)**2);
    output_2 = output+Z[i]/(Z[i]**2+lambda_g**2)*np.dot(V[:,i].reshape(2,1),U[:,i].reshape(1,2))
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
	Y13 = (2*x1+x2)*cos(q2)-(dq2*y1+ dq1*y2 +dq2*y2)*sin(q2);
	Y23 = x1*cos(q2)+ dq1*y1*sin(q2);
	output = np.array([[x1,x2,Y13,cos(q1)*0,cos(q1+q2)*0],[0,x1+x2,Y23,0,cos(q1+q2)*0]])
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
	output_2 = np.dot(inv(M),(torque - np.dot(C,dq) -np.dot(g,0)))
	velocity =velocity.reshape(2,1)
	output = np.vstack((velocity,output_2))
	return(output)

def dJhat(Lhat,dLhat,q,dq):
	Lhat1 = Lhat[0,0]; Lhat2 = Lhat[1,0];
	dLhat1 = dLhat[0,0]; dLhat2 = dLhat[1,0];
	q1 = q[0,0]; q2 = q[1,0];
	dq1 = dq[0,0];dq2 = dq[1,0];
	
	output_1 = np.array([[-dLhat1*sin(q1)-dLhat2*sin(q1+q2),-dLhat2*sin(q1+q2)],[dLhat1*cos(q1)+dLhat2*cos(-q1+q2), dLhat2*cos(q1+q2)]])
	output_2= np.array([[-Lhat1*cos(q1)*dq1-Lhat2*cos(q1+q2)*(dq1+dq2),-Lhat2*cos(q1+q2)*(dq1+dq2)],[-Lhat1*sin(q1)*dq1-Lhat2*sin(q1+q2)*(dq1+dq2),-Lhat2*sin(q1+q2)*(dq1+dq2)]])	
	output = output_1 +output_2;
	return(output)
i=0
def eetrack(t,x,pos,velocity):
	#global pos,velocity
	global u
	#rate = rospy.Rate(100) #100 Hz
	b=0.01
	dx = np.zeros((33,1),dtype = np.float64);
	m1=0.5; m2=0.2; #kg
	l1 = 17.2*b; l2 =6.43*b #cm
	b1 = 9.19*b; b2=0;
	go = 980*b #cm/s2 
	Io1 = (1/12)*m1*l1**2;Io2 = (1/12)*m2*l2**2;
	theta_1 = Io1 + m1*(l1**2)/4+Io2+m2*l2**2/4+m2*l1**2;
	theta_3 = m2*l1*l2/2;
	theta_2 = Io2+m2*l2**2/4;
	theta_4 = m1*go*l1/2+m2*go*l1;
	theta_5 = m2*go*l2/2

	#rospy.loginfo(velocity)

	##system parameters
	a = np.array([[theta_1],[theta_2],[theta_3],[theta_4],[theta_5]],dtype = np.float64)
	L = np.array([[17.2*b],[6.43*b]])
	base =np.array([[9.19*b],[0]]);
	q=pos
	#print(q)
	q = q.reshape(2,1);# q1 =q[0,0];q2 = q[1,0];
	
	#rospy.loginfo('this is %s',q)
	dq=velocity
	dq = dq.reshape(2,1); #dq1 =dq[0,0];dq2 = dq[1,0];
	xi =dq;
	eta= x[0:8].reshape(8,1)
	zeta = x[8:10].reshape(2,1)
	est_ak = x[10:12].reshape(2,1)
	est_az = x[12:28].reshape(16,1)
	est_ad = x[28:33].reshape(5,1)

	x_ref = np.array([[9*b+16*b-3*np.cos(np.pi*t/10)*b],[3*np.sin(np.pi*t/10)*b]],dtype = np.float64)
	#e=e.reshape(2,1)
	Lambda_k = 1; Lambda_z =100; Lambda_d=5;
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

	I_2 = np.identity(2);	
	M_2 = block_diag(M_3,M_3)
	M_1 = np.kron(M_2,I_2);
	N_3 = np.array([[0],[1]])	
	N_2 = block_diag(N_3,N_3)
	N = np.kron(N_2,I_2);

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
	#rospy.loginfo(u)
	#rospy.loginfo('this is time inside ODE %s',t)
	
	
	#dx[0:2,] =q;
	dx[0:8,]= im;
	dx[8:10,]=filter_2;
	dx[10:12,]=est_ak_update;
	dx[12:28,]=est_az_update;
	dx[28:33,]=est_ad_update;
	dx = dx.reshape(33,)

	return(dx)
	
def trial(t,pos,velocity):
	#global pos,velocity
	global e
	#rate = rospy.Rate(100) #100 Hz
	b=0.01
	dx = np.zeros((33,1),dtype = np.float64);
	m1=0.5; m2=0.2; #kg
	l1 = 17.2*b; l2 =6.43*b #cm
	b1 = 9.19*b; b2=0;
	go = 980*b #cm/s2 
	Io1 = (1/12)*m1*l1**2;Io2 = (1/12)*m2*l2**2;
	theta_1 = Io1 + m1*(l1**2)/4+Io2+m2*l2**2/4+m2*l1**2;
	theta_3 = m2*l1*l2/2;
	theta_2 = Io2+m2*l2**2/4;
	theta_4 = m1*go*l1/2+m2*go*l1;
	theta_5 = m2*go*l2/2

	#rospy.loginfo(velocity)

	##system parameters
	a = np.array([[theta_1],[theta_2],[theta_3],[theta_4],[theta_5]],dtype = np.float64)
	L = np.array([[17.2*b],[6.43*b]])
	base =np.array([[9.19*b],[0]]);
	q=pos
	q = q.reshape(2,1);# q1 =q[0,0];q2 = q[1,0];
	dq=velocity
	dq = dq.reshape(2,1); #dq1 =dq[0,0];dq2 = dq[1,0];
	xi =dq;

	x_ref = np.array([[9*b+16*b-3*np.cos(np.pi*t/10)*b],[3*np.sin(np.pi*t/10)*b]],dtype = np.float64)
	x_2,vel = XJ(L,q.reshape(2,1),dq.reshape(2,1),base);
	e = x_2-x_ref;
	#rospy.loginfo(e)
	est_J = Jacobian(L,q);
	Y = Yd(q,dq,q,dq)
	#rate.sleep()
	k_40=100
	u = - k_40*np.dot(est_J.T,e) + np.dot(Y,a);
	rospy.loginfo('this is something else %s',np.dot(est_J.T,e))
	pub2 = rospy.Publisher('/robot2/joint2_effort_controller/command', Float64, queue_size=10)
	pub3 = rospy.Publisher('/robot2/joint3_effort_controller/command', Float64, queue_size=10)
	rospy.loginfo('this is my torque %s',u)
	pub2.publish(u[0,0])
	pub3.publish(u[1,0])
	return(u)
	

def joint_callback(data):
	global pos,velocity
	pub_msg = JointState()
	pub_msg.name=data.name
	pub_position = data.position
	pub_3= data.velocity
	vel_1=pub_3[0];vel_2=pub_3[1];
	velocity=np.array([vel_1,vel_2]);
	pos_1 =pub_position[0]
	#print(pos_1)
	pos_3 =pub_position[1]
	pos = np.array([pos_1,pos_3]);
	#rospy.loginfo(pos)
	

def publisher():
	global pos,velocity
	#rospy.loginfo(pos)
	b=0.01
	rate = rospy.Rate(100) #100 Hz
	x0 = np.zeros((10,1))
	x4 = np.zeros((15,1));x5 = np.array([[17*b],[6*b]]);
	x6 = np.ones((6,1))
	x0 = np.vstack((x0,x5))
	x0 = np.vstack((x0,x4));
	#x0 = np.vstack((x0,x4));
	x0=np.vstack((x0,x6));
	x0=x0.reshape(33,)
	t0=0;tf=10;dt=0.1
	t_span=(t0,t0+10)
	time=np.arange(t0,tf+dt,dt)
	sol=np.zeros((len(time),len(x0)))
	sol[0,:]=x0
	options={'rtol':1e-6,'atol':1e-8}
	#sol_2 = solve_ivp(eetrack,[0,100],x0,method='BDF')
	t=t0
	for i in range(1,len(time)):
		tspan=[time[i-1],time[i]]
		sol_i=solve_ivp(eetrack,tspan,sol[i-1,:],method='BDF')
		assert sol_i.success()
		sol[i,:]=sol_i.y[:,-1]
		
def publisher_trial():
	rospy.init_node('joint_positions_node', anonymous=True)
	rospy.Subscriber("/robot2/joint_states", JointState,joint_callback)
	b=0.01
	pub2 = rospy.Publisher('/robot2/joint2_effort_controller/command', Float64, queue_size=10)
	pub3 = rospy.Publisher('/robot2/joint3_effort_controller/command', Float64, queue_size=10)
	rate = rospy.Rate(100) #100 Hz
	x0 = np.zeros((10,1))
	x4 = np.zeros((21,1));x5 = np.array([[17*b],[6*b]]);
	x0 = np.vstack((x0,x5))
	x0 = np.vstack((x0,x4));
	x0=x0.reshape(33,)
	def solve_and_publish():
		global pos,velocity,e
		assert pos == None 
		switch_controller()
		for i in range(1,100):
			rospy.loginfo('this is %s',pos)
		t0=0;tf=10;dt=0.01
		t_span=(t0,t0+10)
		time=np.arange(t0,tf+dt,dt)
		sol=np.zeros((len(time),len(x0)))
		sol[0,:]=x0
		options={'rtol':1e-4,'atol':1e-6}
		#sol_2 = solve_ivp(eetrack,[0,100],x0,method='BDF')
		t=t0
		
		for i in range(1,len(time)):
			tspan=[time[i-1],time[i]]
			sol_i=solve_ivp(fun=lambda t,y: eetrack(t,y,pos,velocity),t_span=tspan,y0=sol[i-1,:],method='BDF')
			rospy.loginfo(time[i])
			#rospy.loginfo(e)
			pub2.publish(u[0,0])
			pub3.publish(u[1,0])
			sol[i,:]=sol_i.y[:,-1]
			rate.sleep()
	solve_and_publish()

		
def switch_controller():
	rospy.wait_for_service('/robot2/controller_manager/switch_controller')
	try:
		switch_controller = rospy.ServiceProxy('/robot2/controller_manager/switch_controller', SwitchController)
		ret = switch_controller(['joint1_effort_controller','joint2_effort_controller','joint3_effort_controller'],['joint1_position_controller','joint2_position_controller','joint3_position_controller'], 2,False,5)
		#ret = switch_controller(['joint1_position_controller','joint2_position_controller','joint3_position_controller'],['joint1_effort_controller','joint2_effort_controller','joint3_effort_controller'], 2,True,1)
	except rospy.ServiceException:
		print ("Service call failed")

def subscriber():
	rospy.Subscriber("/robot2/joint_states", JointState,joint_callback)

#Main section of code that will continuously run unless rospy receives interuption (ie CTRL+C)
if __name__ == '__main__':
	try:
		publisher_trial()
	except rospy.ROSInterruptException:
		pass
	#rospy.init_node('joint_positions_node', anonymous=True)
	#switch_controller()
	#rospy.Subscriber("/robot2/joint_states", JointState,joint_callback)
	#publisher_thread = threading.Thread(target=publisher)
	#publisher_thread_2 = threading.Thread(target=subscriber)
	#publisher_thread.start()



	

'''
if __name__ == '__main__':
	try: 
		print('this is not working')
		rrbot_joint_positions_publisher()
	except rospy.ROSInterruptException:
		pass


	#t = np.arange(0,10,1/10)
	x =scipy.integrate.ode(eetrack).set_integrator('vode',method = 'bdf',order=15)
	x.set_initial_value(x0,0)
	x.set_f_params(x_real)
	print(x_real)
	print('esto')
	x5 = 100
	dt=0.01
	#While loop to have joints follow a certain position, while rospy is not shutdown.
	i = 0
	print('this is not working 2')
	
	
	
	#Have each joint follow a sine movement of sin(i/100).
		dx=x.integrate(x.t+dt)
		x_real = np.array([pos_1,pos_3]);
		x_real,vel = XJ(L,x_real.reshape(2,1),(np.array([0,0])).reshape(2,1),base);
		x.set_f_params(x_real)
		assert x.successful()
		print(x.t)
		i=i+1
		sine_movement = sin(i/100.)
		#print(sine_movement)
		#Publish the same sine movement to each joint.
		#pub1.publish(0)
		pub2.publish(dx[0])
		pub3.publish(dx[1])
		i = i+1 #increment i


x = np.arange(0, i+1,1)  # Sample data.
#print(x)
fig, axs = plt.subplots(figsize=(10, 12.7))
#print(pos_1[0,:])
#print(x_ref)
axs.plot(x, pos_1[0,0:i+1], label='Real')  # Plot some data on the axes.
axs.plot(x, x_ref[0,:], label='referencia')  # Plot more data on the axes...
#ax.plot(x, x**3, label='cubic') 
axs.grid(True)
plt.show()

fig_2, axs = plt.subplots(figsize=(10, 12.7))
axs.plot(x, pos_1[1,0:i+1], label='Real')  # Plot some data on the axes.
axs.plot(x, x_ref[1,:], label='referencia')  # Plot more data on the axes...
axs.grid(True)
plt.show()

fig_3, axs = plt.subplots(figsize=(10, 12.7))
axs.plot(pos_1[0,0:i+1], pos_1[1,0:i+1], label='Real')  # Plot some data on the axes.
axs.plot(x_ref[0,:], x_ref[1,:], label='referencia')  # Plot more data on the axes...
axs.grid(True)
plt.show()

fig_4, axs = plt.subplots(figsize=(10, 12.7))
axs.plot(x, e[0,0:i+1], label='Real')  # Plot some data on the axes.
axs.plot(x, e[1,0:i+1], label='referencia')  # Plot more data on the axes...
axs.grid(True)
plt.show()

print(pos_1)
	
	L = np.array([[17.2],[6.43]])
	base =np.array([[9.19],[0]]);
	#print(rate)
	x_real = np.array([pos_1,pos_3]);
	print(x_real)
	x_real,vel = XJ(L,x_real.reshape(2,1),(np.array([0.001,0.001])).reshape(2,1),base);
	x3 = np.zeros((12,1))
	x4 = np.zeros((21,1));x5 = np.array([[17],[6]]);
	x0 = np.array([[0],[0]],dtype = np.float64)
	x0 = np.vstack((x0,x3))
	x0 = np.vstack((x0,x5));
	x0 = np.vstack((x0,x4))
	x0=x0.reshape(37,)
	x_ref_2= np.array([[],[]])
	x1_1 = np.array([[],[]])
	x3 = np.array([])
	pos =np.array([[],[]])
	#t = np.arange(0,10,1/10)
	x =scipy.integrate.ode(eetrack).set_integrator('vode',method = 'bdf',order=15)
	x.set_initial_value(x0,0)
	print(x_real)
	x.set_f_params(x_real)
	x5 = 100
	dt=0.1
	#While loop to have joints follow a certain position, while rospy is not shutdown.
	i = 0
	print('this is not working 2')
	i=1
	

	
def joint_callback(data):
	global x0
	pub_msg = JointState()
	#print(pub_msg)
	#pub_msg.header = Header()
	pub_msg.name=data.name
	pub_position = data.position
	global pos_3,pos_1
	pos_1 =pub_position[0]
	#print(pos_1)
	pos_3 =pub_position[1]
	q = np.array([pos_1,pos_3]);
	print(q)
	dt=0.001;
	x1,vel = XJ(L,q.reshape(2,1),x0[0:2,].reshape(2,1),base);
	x.set_f_params(q)
	x0=x.integrate(x.t+dt)
	assert x.successful()
	print(i)
	print(u)
	number_1=u[0,0] 
	assert number_1<0.5
	assert u[1,0]<0.5
	pub2.publish(u[0,0])
	pub3.publish(u[1,0])
	
	
	
		
	#print(pos_3)
	#rospy.loginfo("I will publish to the topic %s",pub_position )

def rrbot_joint_positions_publisher():
	global L,base,i
	#Initiate node for controlling joint1 and joint2 positions.
	rospy.init_node('joint_positions_node', anonymous=True)
	rate = rospy.Rate(10) #100 Hz
	dt=1/50
	i=0+dt
	#print(t[i])
	#pos_3= -2.43; pos_1=0.32;
	while i<50:
		i=i+dt
		print(i)
		pub2 = rospy.Publisher('/robot2/joint2_position_controller/command', Float64, queue_size=10)
		pub3 = rospy.Publisher('/robot2/joint3_position_controller/command', Float64, queue_size=10)
		sub=rospy.Subscriber("/robot2/joint_states", JointState,joint_callback)
		rate.sleep()
			
	#while not rospy.is_shutdown():	

	
	
'''





	


