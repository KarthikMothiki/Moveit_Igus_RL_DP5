#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from moveit_msgs.msg import ExecuteTrajectoryActionResult
import dryve_D1 as dryve
import numpy as np

speed = 5
accel = 100
homespeed = 5
homeaccel = 100


#following are the fucntions that we want to expose through ROS
#workspace will be /rl_dp_5
#1. publisher: /status for a joint such as {mode of operation, current position, is_initialized }, this will require calling multiple functiosn from dryve_D1.py
#2. service: /setMode : integer as an input passed on to function set_mode from dryve_D1.py -> check the arguments
#3. service: /home : this will call homing from dryve_D1.py -> check the arguments
#4. subsriber: /cmd/set_joint_position : this will set desired joint position by calling profile_pos_mode -> check arguments
#5. 
#
#
#
#
#
#
#start ROS Node code here


class Rl_DP_5:
    def __init__(self):
        # Initialize your 5 D1 axes here
        Aaxis = dryve.D1("169.254.0.1", 502, 'Axis 1', -140, -140, 140)
        Baxis = dryve.D1("169.254.0.2", 502, 'Axis 2', -100, -100, 50)
        Caxis = dryve.D1("169.254.0.3", 502, 'Axis 3', -115, -115, 115)
        Daxis = dryve.D1("169.254.0.4", 502, 'Axis 4', -100, -100, 100)
        Eaxis = dryve.D1("169.254.0.5", 502, 'Axis 5', -180, -179, 179)

        self.axis_controller = [Aaxis, Baxis, Caxis, Daxis, Eaxis]
        print('Created dryve interfaces')

    def set_target_position(self, axis, desired_absolute_position):
        if 0 <= axis < len(self.axis_controller):
            self.axis_controller[axis].profile_pos_mode(desired_absolute_position, speed, accel)

    def home(self, axis):
        print(f"Started homing Axis {axis + 1}")
        self.axis_controller[axis].homing(homespeed, homeaccel)

    def home_all(self):
        for axis in self.axis_controller:
            print(f"Started homing {axis.Axis}")
            axis.homing(homespeed, homeaccel)

    def get_current_position(self, axis):
        return self.axis_controller[axis].getPosition()


class MoveItInterface:
            
#here we should have two things
#1. Subsriber to joint_states from moveit to getjoint space trajectory to plannned pose
#2. Published to fake_joint_controller to simulate the robot pose in Moveit based on current postiison of the real robot
    def __init__(self, robot):
        self.robot = robot
        self.execution_result = None

        rospy.init_node('joint_states_subscriber', anonymous=True)
        self.fake_controller_joint_states_pub = rospy.Publisher('/move_group/fake_controller_joint_states', JointState, queue_size=10)

        #rospy.Subscriber('/joint_states', JointState, self.callback_fn)

    def robot_position_to_moveit(self):
        for i in range(5):
            self.fake_controller_joint_states_pub.publish(np.deg2rad(robot.get_current_position(i)))

        rospy.Subscriber('/joint_states', JointState, self.callback_fn)

        # joint_state_position = rospy.Subscriber("/joint_states", JointState, callback_fn, queue_size=10)
        # return joint_state_position

    # def joint_state_sub():
    #     rospy.init_node('joint_state_sub_node', anonymous=True)
    #     rospy.Subscriber("/joint_states", JointState, callback_fn)

    def callback_fn(self, data):
        self.position = list(data.position)
        for axis, position in enumerate(self.position):
            self.robot.set_target_position(axis, position)
        rospy.sleep(0.01)

if __name__ == "__main__":
    print('Initialized an object for the robot')
    robot = Rl_DP_5()
    print('Initialized an object for Moveit interface')
    move_it_interface = MoveItInterface(robot)

    try:
        while not rospy.is_shutdown():
            move_it_interface.robot_position_to_moveit()
            rospy.sleep(0.01)

    except rospy.ROSInterruptException:
        pass