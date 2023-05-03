import numpy as np
import scipy.linalg
import scipy.integrate
from scipy.linalg import block_diag
from numpy.linalg import multi_dot,inv
from scipy.integrate import odeint,solve_ivp,LSODA
import matplotlib.pyplot as plt
import time
import math 
from numpy import cos,sin
from time import time, sleep, strftime, localtime




#define known  variables of the dynamic equation
#Mass Matrix 2Dof 

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

def eetrack(t,x,e,q):
	global u
	#print(t)
	dx = np.zeros((35,1),dtype = np.float64);
	m1=0.2; m2=0.2; #kg
	l1 = 17.2; l2 =6.43 #cm
	b1 = 9.19; b2=0;
	go = 980 #cm/s2 
	Io1 = (1/12)*m1*l1**2;Io2 = (1/12)*m2*l2**2;
	theta_1 = Io1 + m1*(l1**2)/4+Io2+m2*l2**2/4+m2*l1**2;
	theta_3 = m2*l1*l2/2;
	theta_2 = Io2+m2*l2**2/4;
	theta_4 = m1*go*l1/2+m2*go*l1;
	theta_5 = m2*go*l2/2
	#reference		
	##system parameters
	a = np.array([[theta_1],[theta_2],[theta_3],[theta_4],[theta_5]],dtype = np.float64)
		
	#x_ref = np.array([[9+16-3*np.cos(np.pi*t/10)],[3*np.sin(np.pi*t/10)]],dtype = np.float64)
	#reference		
	##system parameters
	#a = np.array([[3.9],[0.75],[1.125],[23.52],[7.35]])
	L = np.array([[17.2],[6.43]])
	base =np.array([[9.19],[0]]);
	#print(x)
	#x=x.astype(np.float)
	##states
	#print(x.shape)
	
	q = q.reshape(2,1); q1 =q[0,0];q2 = q[1,0];
	dq = x[0:2].reshape(2,1); dq1 =dq[0,0];dq2 = dq[1,0];xi =dq;
	eta= x[2:10].reshape(8,1)
	zeta = x[10:12].reshape(2,1)
	est_ak = x[12:14].reshape(2,1)
	est_az = x[14:30].reshape(16,1)
	est_ad = x[30:35].reshape(5,1)
	
	#if t< 100:
		#Lambda_k = 100; Lambda_z =100; Lambda_d=100;
	#else:
	Lambda_k = 1; Lambda_z =100; Lambda_d=1000;
	x,vel = XJ(L,q.reshape(2,1),x[0:2].reshape(2,1),base);
	#e = x-x_ref;
	#print(e)
	#Jacobian 
	Yk_2 = Yk(q,dq);
	est_J = Jacobian(est_ak,q);
	est_ak_update = -Lambda_k*np.dot(Yk_2.T,(zeta-2*e));
	d_est_J = dJhat(est_ak,est_ak_update,q,dq);

	##IM 
	#if t<100:
		#M_3 = np.array([[0,1],[-4,-2]])
	#else:
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
	xi_r = np.dot(inv(est_J),(-k11*zeta+ np.dot(Z,est_az)));
	d_inv_J = -np.dot(np.dot(inv(est_J),d_est_J),inv(est_J));

	dxi_r = np.dot(d_inv_J,(-k11*zeta+np.dot(Z,est_az)))+np.dot(inv(est_J),(-k11*filter_2 + np.dot(Z,est_az_update) + np.dot(d_Z,est_az)));
	Y = Yd(q,dq,dxi_r,xi_r)
	est_ad_update = -Lambda_d*np.dot(Y.T,(xi-xi_r));
	u = -k22*(xi-xi_r) - np.dot(est_J.T,e) + np.dot(Y,est_ad);
	print(u)
	dx[0:2,] =q;
	dx[2:10,]= im;
	dx[10:12,]=filter_2;
	dx[12:14,]=est_ak_update;
	dx[14:30,]=est_az_update;
	dx[30:35,]=est_ad_update;
	dx = dx.reshape(35,)
	return(dx)
	
def euler_eetrack(t,x,e):
	for i in range(0,100,1):
		dx = np.zeros((37,1));
		m1=0.2; m2=0.2; #kg
		l1 = 17.2; l2 =6.43 #cm
		b1 = 9.19; b2=0;
		go = 980 #cm/s2 
		Io1 = (1/12)*m1*l1**2;Io2 = (1/12)*m2*l2**2;
		theta_1 = Io1 + m1*(l1**2)/4+Io2+m2*l2**2/4+m2*l1**2;
		theta_3 = m2*l1*l2/2;
		theta_2 = Io2+m2*l2**2/4;
		theta_4 = m1*go*l1/2+m2*go*l1;
		theta_5 = m2*go*l2/2
		#reference		
		##system parameters
		a = np.array([[theta_1],[theta_2],[theta_3],[theta_4],[theta_5]])
		#a = np.array([[3.9],[0.75],[1.125],[23.52],[7.35]])
		L = np.array([[l1],[l2]])
		base =np.array([[b1],[b2]]);
		x2=x;
		#print(type(x))
		#print(x)
		#x=x.astype(np.float)
		##states
		#print(x.shape)

		q = x[0:2].reshape(2,1); q1 =q[0,0];q2 = q[1,0];
		dq = x[2:4].reshape(2,1); dq1 =dq[0,0];dq2 = dq[1,0];xi =dq;
		eta= x[4:12].reshape(8,1)
		zeta = x[12:14].reshape(2,1)
		est_ak = x[14:16].reshape(2,1)
		est_az = x[16:32].reshape(16,1)
		est_ad = x[32:37].reshape(5,1)

		#if t< 100:
			#Lambda_k = 100; Lambda_z =100; Lambda_d=100;
			#Lambda_k = 100; Lambda_z =100; Lambda_d=100;
		#else:
		Lambda_k = 1; Lambda_z =1000; Lambda_d=100;

		#kinematics
		#error
		#x,vel = XJ(L,q,dq,base);
		#print(x_ref)
		#e = x-x_ref;
		#print(e)
		#print(e)
		
		#print(i)
		#print(e)
		#print(e.min())
		#print(e)
		#print(np.max(e))
		#if abs(np.max(e)) < 10**-6:
			#print('hey break')
			#break
		#Jacobian 
		Yk_2 = Yk(q,dq);
		est_J = Jacobian(est_ak,q);
		est_ak_update = -np.dot(Lambda_k,np.dot(Yk_2.T,(zeta-np.dot(2,e))));
		d_est_J = dJhat(est_ak,est_ak_update,q,dq);

		##IM 
		#if t<100:
			#M_3 = np.array([[0,1],[-4,-2]])
		#else:
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
		est_az_update= np.dot(Lambda_z,np.dot(Z.T,(zeta-np.dot(2,e))));

		##input
		k11 = 10; k22= 10;
		xi_r = np.dot(inv(est_J),(-np.dot(k11,zeta)+ np.dot(Z,est_az)));
		d_inv_J = -np.dot(np.dot(inv(est_J),d_est_J),inv(est_J));
		
		dxi_r = np.dot(d_inv_J,(-np.dot(k11,zeta)+np.dot(Z,est_az)))+np.dot(inv(est_J),(-np.dot(k11,filter_2) + np.dot(Z,est_az_update) + np.dot(d_Z,est_az)));
		Y = Yd(q,dq,dxi_r,xi_r)
		est_ad_update = -Lambda_d*np.dot(Y.T,(xi-xi_r));
		u = -np.dot(k22,(xi-xi_r)) - np.dot(est_J.T,e) + np.dot(Y,est_ad);

		dx[0:4,] = EL2(a,q,dq,u);
		dx[4:12,]= im;
		dx[12:14,]=filter_2;
		dx[14:16,]=est_ak_update;
		dx[16:32,]=est_az_update;
		dx[32:37,]=est_ad_update;
		dx = dx.reshape(37,)
		fd =1/100
		x2 = x2 + np.dot(fd,dx) 
		x=x2
		#print(x)
	return(x2)
	
def get_t_local():
    t = strftime("%H:%M:%S", localtime()) #current local time
    return sum(int(y) * 60 ** i for i,y in enumerate(reversed(t.split(":")))) #current local time in seconds

y = np.array([0,150])
x0 = np.zeros((12,1))
x4 = np.zeros((21,1));x5 = np.array([[17],[6]]);
q = np.array([[0.3],[-2.43]],dtype = np.float64)
x0 = np.vstack((x0,x5))
x0 = np.vstack((x0,x4));
x0=x0.reshape(35,)
times = [get_t_local()] ## starting time point
t=0

L = np.array([[17.2],[6.43]])
base =np.array([[9.19],[0]]);
#L = np.array([[1.5],[1.5]])
#base =np.array([[0],[0]]);
x_ref_2= np.array([[],[]])
x1_1 = np.array([[],[]])
i=0
x3 = np.array([])
pos =np.array([[],[]])

t = np.arange(0,10,1/10)
print(t.shape)
x =scipy.integrate.ode(eetrack).set_integrator('vode',method = 'bdf',order=15)
print(x0)
x.set_initial_value(x0,0)

x.integrate
x5 = 0.1
dt=0.1
i=0+dt
#print(t[i])
print(scipy.__version__)
while  x.t<x5:
	x_ref = np.array([[9+16-3*np.cos(np.pi*i/10)],[3*np.sin(np.pi*i/10)]],dtype = np.float64)
	x1,vel = XJ(L,q.reshape(2,1),x0[0:2,].reshape(2,1),base);
	e = x1-x_ref;
	print(e)
	e_11 = e[0,:]; e_22 = e[1,:];
	if e_11>0:
		e[0,:]=math.log(abs(e_11)+1)
	elif e_11<0:
		e[0,:]=math.log(abs(e_11)+1)*-1
	
	hola = 2
	print(e_11)
	if e_22>0:
		e[1,:]=math.log(abs(e_22)+1)
	elif e_22<0:
		e[1,:]=math.log(abs(e_22)+1)*-1
		
	print(e)
	x.set_f_params(e,q)
		#print(e)
	x0=x.integrate(x.t+dt)
	
	print(x0)
	assert x.successful()
	print(u)
	i=i+dt
	#print(z[0])
	#print(z[1])
	#print(type(z))
	#print(z.shape)
	
	
#print(x)

'''

print('finished')

print(x1_1.shape)
print(x_ref_2.shape)
#print(x_ref_2.shape)
fig_3, axs = plt.subplots(figsize=(10, 12.7))
#axs.plot(pos_1[0,0:i+1], pos_1[1,0:i+1], label='Real')  # Plot some data on the axes.
axs.plot(x_ref_2[0,:],x_ref_2[1,:], label='referencia')
print(x1_1)
axs.plot(x1_1[0,:],x1_1[1,:], label='siguiendo')
axs.grid(True)
plt.show()
	
fd=1/100
T=100
t=np.arange(0,(T-fd),fd)
for i in range(0,len(t)):
	#print(i)
	#if i<(100*100):
		#print(i)
		#t_2 = (i)*fd;
		#x_ref = np.array([[20-4*np.cos(np.pi*t_2/10)],[4*np.sin(np.pi*t_2/10)]])
		#x_ref = np.array([[1.7-0.3*np.cos(np.pi*t_2/10)],[0.3*np.sin(np.pi*t_2/10)]])
	#else:
	#print(i)
	t_2 = (i)*fd;
	print(t_2)
	x_ref = np.array([[9+16-3*np.cos(np.pi*t_2/10)],[3*np.sin(np.pi*t_2/10)]])
		#x_ref = np.array([[1.1+0.3*np.cos(np.pi*t_2/10)],[0.05+0.3*np.sin(np.pi*t_2/5)]])
	#print(x_ref)
	x_ref_2 = np.hstack((x_ref,x_ref_2));	
	q=(x0[0:2,].reshape(2,1));
	dq=(x0[2:4,].reshape(2,1));
	x1,vel = XJ(L,q,dq,base);
	x1_1 = np.hstack((x1_1,x1))
	e = x1-x_ref
	print(e)
	x0=euler_eetrack(t_2,x0,e)
	
		
while 1:
	sleep(0.1)
	x3= np.append(x3,np.array([i]))
	if t< 100:
		x_ref = np.array([[1.7-0.3*np.cos(np.pi*t/10)],[0.3*np.sin(np.pi*t/10)]])
	else:
		x_ref = np.array([[1.1+0.3*np.cos(np.pi*t/10)],[0.05+0.3*np.sin(np.pi*t/5)]])
		
	t_new = get_t_local() # get new timepoint

	inputs = (x_ref,) 

	# solve differential equation, take final result only
	#x0= odeint(eetrack,t=[i,i+0.1],y0=x0,args=inputs)[-1]
	x0= solve_ivp(eetrack,[t,t+0.1],x0,method='BDF',args=(x_ref,))
	#x0=x0.y 
	print(x0.shape)
	t=t+0.1
	q=(x0[0:2,].reshape(2,1));
	dq=(x0[2:4,].reshape(2,1));
	x1,vel = XJ(L,q,dq,base);
	pos = np.hstack((pos,x1));
	x_ref_2 = np.hstack((x_ref_2,x_ref))
	print(x1-x_ref)
	#axs.plot(x3, pos[0,:]-x_ref_2[0,:] ,label='Error_x')  # Plot some data on the axes.
	#axs.plot(x3, pos[1,:]-x_ref_2[1,:], label='Error_y')  # Plot more data on the axes...
	#axs.grid(True)
	#fig1.canvas.draw()
	times.append(t_new)
	i=i+0.1
	#print(t_new)

x0 = solve_ivp(eetrack,[100,110],x0,rtol = 1e-3,atol = 1e-3,dense_output=True,method = 'LSODA')
print('finished')
t =x0.t
print(x0.message)
x0=x0.y 
print(x0.message)
print(t)
print(x0.shape);
for i in range(0,len(x0)):
	q=(x0[0:2,i].reshape(2,1));
	dq=(x0[2:4,i].reshape(2,1));
	x1,vel = XJ(L,q,dq,base);
	pos = np.hstack((pos,x1));
	x_ref = np.array([[9+16-3*np.cos(np.pi*t[i]/10)],[3*np.sin(np.pi*t[i]/10)]])
	x_ref_2 = np.hstack((x_ref_2,x_ref))

print('finished')

print(x_ref_2.shape)
print(pos.shape)

fig_3, axs = plt.subplots(figsize=(10, 12.7))
axs.plot(pos[0,:], pos[1,:], label='Real')  # Plot some data on the axes.
axs.plot(x_ref_2[0,:],x_ref_2[1,:], label='referencia')
axs.grid(True)
plt.show()

t = np.arange(0,100,1/10)
print(t.shape)
x =scipy.integrate.ode(eetrack).set_integrator('dopri5')
x.set_initial_value(x0,0)
x.integrate
x5 = 100
dt=0.01
i=0
while  x.t<x5:
	x.integrate(t[i+1])
	assert x.successful()
	i=i+1
print(x)

def eetrack(t,x,e):
	#print(t)
	dx = np.zeros((37,1),dtype = np.float64);
	m1=0.2; m2=0.2; #kg
	l1 = 17.2; l2 =6.43 #cm
	b1 = 9.19; b2=0;
	go = 980 #cm/s2 
	Io1 = (1/12)*m1*l1**2;Io2 = (1/12)*m2*l2**2;
	theta_1 = Io1 + m1*(l1**2)/4+Io2+m2*l2**2/4+m2*l1**2;
	theta_3 = m2*l1*l2/2;
	theta_2 = Io2+m2*l2**2/4;
	theta_4 = m1*go*l1/2+m2*go*l1;
	theta_5 = m2*go*l2/2
	#reference		
	##system parameters
	a = np.array([[theta_1],[theta_2],[theta_3],[theta_4],[theta_5]],dtype = np.float64)
		
	#x_ref = np.array([[9+16-3*np.cos(np.pi*t/10)],[3*np.sin(np.pi*t/10)]],dtype = np.float64)
	#reference		
	##system parameters
	#a = np.array([[3.9],[0.75],[1.125],[23.52],[7.35]])
	L = np.array([[17.2],[6.43]])
	base =np.array([[9.19],[0]]);
	#print(x)
	#x=x.astype(np.float)
	##states
	#print(x.shape)
	
	q = x[0:2].reshape(2,1); q1 =q[0,0];q2 = q[1,0];
	dq = x[2:4].reshape(2,1); dq1 =dq[0,0];dq2 = dq[1,0];xi =dq;
	eta= x[4:12].reshape(8,1)
	zeta = x[12:14].reshape(2,1)
	est_ak = x[14:16].reshape(2,1)
	est_az = x[16:32].reshape(16,1)
	est_ad = x[32:37].reshape(5,1)
	
	#if t< 100:
		#Lambda_k = 100; Lambda_z =100; Lambda_d=100;
	#else:
	Lambda_k = 1; Lambda_z =100; Lambda_d=1000;
	x,vel = XJ(L,x[0:2].reshape(2,1),x[2:4].reshape(2,1),base);
	#e = x-x_ref;
	#print(e)
	#Jacobian 
	Yk_2 = Yk(q,dq);
	est_J = Jacobian(est_ak,q);
	est_ak_update = -Lambda_k*np.dot(Yk_2.T,(zeta-2*e));
	d_est_J = dJhat(est_ak,est_ak_update,q,dq);

	##IM 
	#if t<100:
		#M_3 = np.array([[0,1],[-4,-2]])
	#else:
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
	xi_r = np.dot(inv(est_J),(-k11*zeta+ np.dot(Z,est_az)));
	d_inv_J = -np.dot(np.dot(inv(est_J),d_est_J),inv(est_J));

	dxi_r = np.dot(d_inv_J,(-k11*zeta+np.dot(Z,est_az)))+np.dot(inv(est_J),(-k11*filter_2 + np.dot(Z,est_az_update) + np.dot(d_Z,est_az)));
	Y = Yd(q,dq,dxi_r,xi_r)
	est_ad_update = -Lambda_d*np.dot(Y.T,(xi-xi_r));
	u = -k22*(xi-xi_r) - np.dot(est_J.T,e) + np.dot(Y,est_ad);
	
	dx[0:4,] = EL2(a,q,dq,u);
	dx[4:12,]= im;
	dx[12:14,]=filter_2;
	dx[14:16,]=est_ak_update;
	dx[16:32,]=est_az_update;
	dx[32:37,]=est_ad_update;
	dx = dx.reshape(37,)
	return(dx)


'''



