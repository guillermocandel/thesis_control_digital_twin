# Abstract
Digital twins have emerged as a promising technology for the real-time simulation and replication of the physical behavior of robotic systems. In this research, the development and implementation of digital twins for robotic manipulators using ROS-Gazebo software simulation is presented. Two simple 2 degree-of-freedom robotic arms, the Quanser 2DOF Serial Flexible Link and the Philips Experimental Robot Arm, as well as a robot with 7 degrees of freedom are studied and their digital twins are developed. Three different controllers are implemented to demonstrate the effectiveness of ROS as a digital twin platform for control. They are compared with the experimental results and with Simulink. Trajectory tracking for a specific case in the industry is the main focus. The system is modeled in the port-Hamiltonian framework for mechanical systems and the Euler-Lagrange.\\

The effectiveness of digital twins for control and validation of robotic manipulators is demonstrated in the experimental results. Accurate and reliable simulation results are provided by the digital twins, which can be used to test and validate the control algorithms of the robotic manipulator. Insights into the usability, problems, and the research future are shown by the results.

* Author: Guillermo Candel <guillermocandel@outlook.com>
* License: GNU General Public License, version 3 (GPL-3.0)

## Matlab

[ROS URDF](http://gazebosim.org/tutorials/?tut=ros_urdf)

## ROS

For using this repo, the master_ros contains the files to be used in ROS. It contains three different controllers for trajectory tracking. One of them is implemented through Python, the other two throught the use of Matlab Simulink. The version used is ROS-Noetic in Linux. For using this repo copy the files inside the master_ros file into your src folder source one level before and 
      
      source devel/setup.bash
      catkin_make
      source devel/setup.bash
Inside the python_control you can find the launch files for the QUANSER and PERA robots from the master.
Quanser
           
           roslaunch python_control SDOF_launch.launch
           
PERA:
           
           roslaunch python_control PERA_launch.launch

both the robots are loaded with PID controllers in case you want to put them in different start position or control them with those. For controlling the PID controller through a GUI yoiu can use

            rosrun rqt_gui rqt_gui 
For swappping between the PID controllers and the effort controllers you can do it thorught a python command ,a service command or through 
             
             rosrun rqt_controller_manager rqt_controller_manager

For using the Matlab Simulink files one must initialize the MATLAB with rosinit with that it connect to the robot


In the case of the PERA the inertia and collision files works perfectly so it can be used for simulations but when it was designed by PHILIPS the mesh was empty so for visualization use the collision view. An update should be made to fix this but there is a lack of time in this project.
