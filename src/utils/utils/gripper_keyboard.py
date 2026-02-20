#!/usr/bin/env python3
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand

class KeyGripper(Node):
    def __init__(self):
        super().__init__('key_gripper')
        self._ac = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')

        self.declare_parameter('open_pos', 0.025)
        self.declare_parameter('close_pos', 0.0)
        self.declare_parameter('max_effort', 20.0)

    def send(self, pos: float):
        if not self._ac.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Action server not available: /gripper_controller/gripper_cmd")
            return

        goal = GripperCommand.Goal()
        goal.command.position = float(pos)
        goal.command.max_effort = float(self.get_parameter('max_effort').value)

        future = self._ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        gh = future.result()
        if gh is None or not gh.accepted:
            self.get_logger().error("Goal rejected")
            return

        self.get_logger().info(f"Sent gripper pos={pos}")

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

def main():
    rclpy.init()
    node = KeyGripper()

    open_pos = float(node.get_parameter('open_pos').value)
    close_pos = float(node.get_parameter('close_pos').value)

    print("Press [1]=close, [2]=open, [q]=quit")

    try:
        while rclpy.ok():
            c = getch()
            if c == '1':
                node.send(close_pos)
            elif c == '2':
                node.send(open_pos)
            elif c.lower() == 'q':
                break
            rclpy.spin_once(node, timeout_sec=0.01)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
