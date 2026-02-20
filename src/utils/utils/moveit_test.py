#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, PoseStamped
import math
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint, RobotState
from moveit_msgs.srv import GetCartesianPath

def quaternion_from_euler(roll=3.14, pitch=0.0, yaw=0.0):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


class MoveItMinimal(Node):
    GROUP_NAME = "manipulator"
    BASE_FRAME = "base_link"
    EE_LINK = "link_6"

    def __init__(self):
        super().__init__("moveit_minimal_with_cartesian")
        self.move_ac = ActionClient(self, MoveGroup, "move_action")
        self.exec_ac = ActionClient(self, ExecuteTrajectory, "execute_trajectory")
        self.cart_cli = self.create_client(GetCartesianPath, "compute_cartesian_path")

    # -------------------------
    # 1) 일반 플래닝(관절 목표) + 실행
    # -------------------------
    def plan_execute_joint_goal(self, joint_targets: dict) -> bool:
        if not self.move_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("move_action 서버가 없습니다. move_group 실행 확인.")
            return False

        goal_c = Constraints()
        for name, pos in joint_targets.items():
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = 1e-3
            jc.tolerance_below = 1e-3
            jc.weight = 1.0
            goal_c.joint_constraints.append(jc)

        req = MotionPlanRequest()
        req.group_name = self.GROUP_NAME
        req.allowed_planning_time = 3.0
        req.num_planning_attempts = 5
        req.max_velocity_scaling_factor = 0.5
        req.max_acceleration_scaling_factor = 0.5
        req.goal_constraints = [goal_c]
        # start_state 비우면 "현재 상태"를 사용합니다.

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = False  # plan + execute
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2

        f_send = self.move_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, f_send)
        gh = f_send.result()
        if gh is None or not gh.accepted:
            self.get_logger().error("MoveGroup goal rejected.")
            return False

        f_res = gh.get_result_async()
        rclpy.spin_until_future_complete(self, f_res)
        res = f_res.result().result
        self.get_logger().info(f"[MoveGroup] error_code={res.error_code.val} (1=SUCCESS)")
        return res.error_code.val == 1

    # -------------------------
    # 2) 카테시안(직선) 경로 + 실행
    # -------------------------
    def cartesian_execute(self, waypoints: list, eef_step=0.005, jump_threshold=0.0) -> bool:
        # 서비스/액션 서버 확인
        if not self.cart_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("compute_cartesian_path 서비스가 없습니다.")
            return False
        if not self.exec_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("execute_trajectory 액션 서버가 없습니다.")
            return False

        req = GetCartesianPath.Request()
        req.header.frame_id = self.BASE_FRAME
        req.group_name = self.GROUP_NAME
        req.link_name = self.EE_LINK
        req.waypoints = waypoints
        req.max_step = float(eef_step)
        req.jump_threshold = float(jump_threshold)
        req.avoid_collisions = True

        # start_state를 비우면 어떤 상태를 기준으로 할지 불명확할 수 있어,
        # 기본값으로 "현재 상태"를 쓰도록 is_diff=True로 둔 빈 RobotState를 넣습니다.
        req.start_state = RobotState()
        req.start_state.is_diff = True

        f = self.cart_cli.call_async(req)
        rclpy.spin_until_future_complete(self, f)
        res = f.result()
        if res is None:
            self.get_logger().error("compute_cartesian_path 호출 실패")
            return False

        frac = res.fraction
        self.get_logger().info(f"[Cartesian] fraction={frac:.3f}")
        if frac < 0.95:
            self.get_logger().warning("직선 경로가 충분히 계산되지 않았습니다(fraction 낮음).")
            return False

        # 실행
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = res.solution

        f_send = self.exec_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, f_send)
        gh = f_send.result()
        if gh is None or not gh.accepted:
            self.get_logger().error("ExecuteTrajectory goal rejected.")
            return False

        f_res = gh.get_result_async()
        rclpy.spin_until_future_complete(self, f_res)
        out = f_res.result().result
        self.get_logger().info(f"[ExecuteTrajectory] error_code={out.error_code.val} (1=SUCCESS)")
        return out.error_code.val == 1


def main():
    rclpy.init()
    node = MoveItMinimal()

    # (A) 일반 플래닝 예시: 관절 목표
    node.plan_execute_joint_goal({
        # "joint_1": 0.0,
        # "joint_2": -1.0,
        # "joint_3": 1.0,
        # "joint_4": 0.0,
        # "joint_5": 1.57,
        # "joint_6": 0.0,
    })

    # (B) 카테시안 예시: z로 10cm 내려갔다가 올라오기
    w1 = Pose()
    w1.position.x = 0.8
    w1.position.y = -0.2
    w1.position.z = 0.50
    w1.orientation.x, w1.orientation.y, w1.orientation.z, w1.orientation.w = quaternion_from_euler(roll=3.14, pitch=0.0, yaw=0.0)

    w2 = Pose()
    w2.position.x = w1.position.x
    w2.position.y = w1.position.y
    w2.position.z = w1.position.z - 0.10
    w2.orientation.x, w2.orientation.y, w2.orientation.z, w2.orientation.w = quaternion_from_euler(roll=3.14, pitch=0.0, yaw=0.0)

    node.cartesian_execute([w1, w2], eef_step=0.005, jump_threshold=0.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
