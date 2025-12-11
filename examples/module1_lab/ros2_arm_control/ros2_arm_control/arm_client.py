#!/usr/bin/env python3

"""
Simple ROS 2 client for controlling a humanoid arm.

This node demonstrates basic ROS 2 client concepts including:
- Creating a client node
- Sending messages to publishers
- Subscribing to feedback
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import time


class ArmClient(Node):
    """
    A simple ROS 2 client for controlling a humanoid arm.
    """

    def __init__(self):
        super().__init__('arm_client')

        # Create a publisher to send commands to the arm controller
        self.command_publisher = self.create_publisher(
            Float64MultiArray,
            'joint_commands',
            10
        )

        # Create a subscriber to listen to joint states
        self.state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.state_callback,
            10
        )

        # Timer to send commands periodically
        self.timer = self.create_timer(1.0, self.send_command)

        self.command_counter = 0

        self.get_logger().info('Arm Client node initialized')

    def state_callback(self, msg):
        """Callback for receiving joint states."""
        if len(msg.name) > 0:
            self.get_logger().info(f'Received joint states: {msg.name[:3]}...')

    def send_command(self):
        """Send a command to the arm."""
        msg = Float64MultiArray()

        # Send different positions based on counter
        if self.command_counter % 3 == 0:
            msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Home position
        elif self.command_counter % 3 == 1:
            msg.data = [0.5, 0.3, 0.2, 0.0, 0.0, 0.0]  # Position 1
        else:
            msg.data = [-0.5, -0.3, -0.2, 0.0, 0.0, 0.0]  # Position 2

        self.command_publisher.publish(msg)
        self.get_logger().info(f'Sent command: {msg.data}')
        self.command_counter += 1


def main(args=None):
    """Main function to run the arm client node."""
    rclpy.init(args=args)

    arm_client = ArmClient()

    try:
        rclpy.spin(arm_client)
    except KeyboardInterrupt:
        pass
    finally:
        arm_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()