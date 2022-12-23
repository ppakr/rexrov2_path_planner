import numpy as np

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from geometry_msgs.msg import TwistStamped, WrenchStamped
from nav_msgs.msg import Odometry

from rexrov2_path_planner.potential_field import PotentialField
from rexrov2_path_planner.pure_pursuit import PurePursuit
from tabulate import tabulate

class AUVPathPlanner(Node):
    def __init__(self):
        super().__init__('rexrov2_path_planner_node')

        # initialize parameters
        self.planning_rate = 5.0  # Hz
        self.planning_period = 1 / self.planning_rate

        self.planning_gain = 0.5
        self.tracking_gain = 0.5

        self.path = []

        self.start_x = 10.0  # start x position [m]
        self.start_y = 0.0  # start y position [m]
        self.goal_x = 30.0  # goal x position [m]
        self.goal_y = 0.0  # goal y position [m]
        # obstacle x position list [m]
        # self.obstacle_x = [15.0, 5.0, 20.0, 25.0]
        self.obstacle_x = [-1.0]
        # obstacle y position list [m]
        # self.obstacle_y = [25.0, 15.0, 26.0, 25.0]
        self.obstacle_y = [-1.0]

        self.resolution = 2.0
        self.tol = 0.8
        self.robot_radius = 10.0

        self.odom_gt = np.zeros([2, 1])

        self.potential_field = PotentialField()
        self.pure_pursuit = PurePursuit(kp=0.1)

        # create subscriber
        self.pose_gt_sub = self.create_subscription(
            Odometry, "/rexrov/pose_gt", self.odom_gt_callback, 10)

        # create publisher
        self.tracking_pub = self.create_publisher(Odometry, "cmd_odom", 10)

        # calculate path
        self.path = self.path_planner()
        self.idx = 0
        self.is_goal = False

        # create timer callback
        self.timer = self.create_timer(
            self.planning_period, self.planning_callback)

    def odom_gt_callback(self, odom):
        self.odom_gt[0, 0] = odom.pose.pose.position.x
        self.odom_gt[1, 0] = odom.pose.pose.position.y

    def path_planner(self):
        # get path from potential field planning
        path = self.potential_field.potential_field_planning(
            self.start_x, self.start_y, self.goal_x, self.goal_y, self.obstacle_x, self.obstacle_y, self.resolution, self.robot_radius)
        return np.array(path)

    def planning_callback(self):
        
        if(self.is_goal == False):
            # # calculate pure pursuit vel
            chi_d, u_d, d = self.pure_pursuit.pure_pursuit(np.array([self.path[self.idx, 0], self.path[self.idx, 1]]),
                                                           np.array([self.odom_gt[0, 0], self.odom_gt[1, 0]]))
            # Check idx
            if (d < self.tol):  # Switch to next point
                self.idx += 1

                if(self.idx >= len(self.path)):
                    self.is_goal = True
                    u_d = 0.0
                    chi_d = 0.0
                    pass

            # chi_d to quanternion
            [qx, qy, qz, qw] = self.euler_to_quaternion(0.0, 0.0, chi_d)

            # TODO publish message
            odom_cmd_msg = Odometry()
            odom_cmd_msg.header.stamp = self.get_clock().now().to_msg()

            odom_cmd_msg.twist.twist.linear.x = u_d
            odom_cmd_msg.pose.pose.orientation.x = qx
            odom_cmd_msg.pose.pose.orientation.y = qy
            odom_cmd_msg.pose.pose.orientation.z = qz
            odom_cmd_msg.pose.pose.orientation.w = qw

            self.tracking_pub.publish(odom_cmd_msg)
            
            
            print("idx: ", self.idx)
            print("chi_d: ", chi_d)
            print("u_d: ", u_d)
            print("d: ", d)
            table = [[self.idx, chi_d, u_d, d, self.path[self.idx, 0], self.path[self.idx, 1], self.odom_gt[0, 0], self.odom_gt[1, 0]]]
            print(tabulate(table, headers=["idx","chi_d", "u_d", "d", "x_d", "y_d", "x", "y"]))
        else:
            pass

    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.

        Input
            :param roll: The roll (rotation around x-axis) angle in radians.
            :param pitch: The pitch (rotation around y-axis) angle in radians.
            :param yaw: The yaw (rotation around z-axis) angle in radians.

        Output
            :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
            np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
            np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
            np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
            np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]


def main(args=None):
    rclpy.init(args=args)

    rexrov2_path_planner = AUVPathPlanner()

    rclpy.spin(rexrov2_path_planner)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    rexrov2_path_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
