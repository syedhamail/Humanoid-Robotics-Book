#!/usr/bin/env python3

"""
Simple ROS 2 node for controlling a humanoid arm.

This node demonstrates basic ROS 2 concepts including:
- Creating a node
- Publishing messages
- Subscribing to messages
- Using services
- Using actions
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionServer, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import math


class ArmController(Node):
    """
    A simple ROS 2 node for controlling a humanoid arm.
    """

    def __init__(self):
        super().__init__('arm_controller')

        # Create a publisher for joint commands
        self.joint_command_publisher = self.create_publisher(
            Float64MultiArray,
            'joint_commands',
            10
        )

        # Create a subscriber for joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Timer for periodic publishing
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Store current joint positions
        self.current_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6 DOF arm

        self.get_logger().info('Arm Controller node initialized')

    def joint_state_callback(self, msg):
        """Callback for joint state subscription."""
        if len(msg.position) >= 6:
            self.current_positions = list(msg.position[:6])
            self.get_logger().debug(f'Current joint positions: {self.current_positions}')

    def timer_callback(self):
        """Timer callback to publish joint commands."""
        # Example: Publish a simple oscillating trajectory
        msg = Float64MultiArray()
        t = self.get_clock().now().nanoseconds / 1e9  # time in seconds

        # Create oscillating motion for first 3 joints
        positions = [
            math.sin(t) * 0.5,           # Shoulder joint
            math.sin(t * 0.7) * 0.3,     # Elbow joint
            math.sin(t * 1.3) * 0.2,     # Wrist joint
            0.0,                         # Fixed for simplicity
            0.0,                         # Fixed for simplicity
            0.0                          # Fixed for simplicity
        ]

        msg.data = positions
        self.joint_command_publisher.publish(msg)


def main(args=None):
    """Main function to run the arm controller node."""
    rclpy.init(args=args)

    arm_controller = ArmController()

    try:
        rclpy.spin(arm_controller)
    except KeyboardInterrupt:
        pass
    finally:
        arm_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()