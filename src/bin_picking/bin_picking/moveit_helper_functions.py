#!/usr/bin/env python3
import time
import math
import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import (
    RobotState,
    RobotTrajectory,
    Constraints,
    JointConstraint,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive


class MoveItMoveHelper(Node):
    def __init__(self):
        super().__init__("moveit_move_helper")

        # ===== 사용 환경에 맞게 수정 =====
        self.BASE_FRAME = "base_link"
        self.GROUP_NAME = "manipulator"
        self.EE_LINK = "link_6"

        self.MOVE_ACTION_NAME = "/move_action"
        self.EXEC_ACTION_NAME = "/execute_trajectory"
        self.CART_SERVICE_NAME = "/compute_cartesian_path"
        # ===============================

        self.move_ac = ActionClient(self, MoveGroup, self.MOVE_ACTION_NAME)
        self.exec_ac = ActionClient(self, ExecuteTrajectory, self.EXEC_ACTION_NAME)
        self.cart_cli = self.create_client(GetCartesianPath, self.CART_SERVICE_NAME)

    # ============================================================
    # [NEW] 내부 헬퍼: 입력 데이터(List/Numpy/Msg) -> PoseStamped 변환
    # ============================================================
    def _build_pose_stamped(self, xyz, quat) -> PoseStamped:
        """
        xyz: [x, y, z] (list, np.array) or Point msg
        quat: [x, y, z, w] (list, np.array) or Quaternion msg
        """
        ps = PoseStamped()
        ps.header.frame_id = self.BASE_FRAME
        ps.header.stamp = self.get_clock().now().to_msg()

        # XYZ 처리
        if isinstance(xyz, Point):
            ps.pose.position = xyz
        else:
            # list or numpy
            ps.pose.position.x = float(xyz[0])
            ps.pose.position.y = float(xyz[1])
            ps.pose.position.z = float(xyz[2])

        # Quaternion 처리
        if isinstance(quat, Quaternion):
            ps.pose.orientation = quat
        else:
            # list or numpy (x, y, z, w)
            ps.pose.orientation.x = float(quat[0])
            ps.pose.orientation.y = float(quat[1])
            ps.pose.orientation.z = float(quat[2])
            ps.pose.orientation.w = float(quat[3])

        return ps

    def _build_pose(self, xyz, quat) -> Pose:
        """PoseStamped가 아닌 순수 Pose 메시지 생성 (Cartesian Waypoints용)"""
        return self._build_pose_stamped(xyz, quat).pose

    # ============================================================
    # (1) OMPL 이동 (MoveGroup) - 인터페이스 변경
    # ============================================================
    def move_ompl_to_pose(
        self,
        xyz,
        quat,
        collision: bool = True,
        planning_time: float = 2.0,
        attempts: int = 5,
        vel_scale: float = 0.3,
        acc_scale: float = 0.3,
        pos_tol: float = 0.003,
        ori_tol: float = 0.05
    ) -> bool:
        """
        Args:
            xyz: [x,y,z] or Point
            quat: [x,y,z,w] or Quaternion
            collision: OMPL에서는 기본적으로 True(충돌 회피). False 설정 시 경고 로그(OMPL은 PlanningScene 조작 없이는 충돌 무시가 어려움)
        """
        target_pose_stamped = self._build_pose_stamped(xyz, quat)

        # OMPL Plan
        traj = self.plan_ompl_to_pose(
            target_pose_stamped,
            planning_time, attempts, vel_scale, acc_scale, pos_tol, ori_tol
        )
        
        if traj is None:
            return False
        
        # Execute
        return self.execute_trajectory(traj)

    def plan_ompl_to_pose(
        self,
        target: PoseStamped,
        planning_time: float = 2.0,
        attempts: int = 5,
        vel_scale: float = 0.3,
        acc_scale: float = 0.3,
        pos_tol: float = 0.003,
        ori_tol: float = 0.05,
        start_state: Optional[RobotState] = None,
    ) -> Optional[RobotTrajectory]:
        
        if not self.move_ac.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("MoveGroup action server not available")
            return None

        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.GROUP_NAME
        req.num_planning_attempts = int(attempts)
        req.allowed_planning_time = float(planning_time)
        req.max_velocity_scaling_factor = float(vel_scale)
        req.max_acceleration_scaling_factor = float(acc_scale)

        if start_state is None:
            req.start_state = RobotState()
            req.start_state.is_diff = True
        else:
            req.start_state = start_state

        # Constraint 생성 로직 내부 호출
        req.goal_constraints = [self._make_pose_goal_constraints(target, pos_tol, ori_tol)]

        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False

        fut = self.move_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        
        if not fut.done() or fut.result() is None:
            return None
        gh = fut.result()
        if not gh.accepted:
            return None

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=planning_time + 5.0)
        
        if not res_fut.done() or res_fut.result() is None:
            return None

        traj = getattr(res_fut.result().result, "planned_trajectory", None)
        if traj is None or not traj.joint_trajectory.points:
            return None

        return traj

    def _make_pose_goal_constraints(self, target: PoseStamped, pos_tol, ori_tol) -> Constraints:
        c = Constraints()
        pc = PositionConstraint()
        pc.header = target.header
        pc.link_name = self.EE_LINK
        pc.weight = 1.0
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [float(pos_tol)]
        bv = BoundingVolume()
        bv.primitives = [sphere]
        bv.primitive_poses = [target.pose]
        pc.constraint_region = bv
        c.position_constraints = [pc]

        oc = OrientationConstraint()
        oc.header = target.header
        oc.link_name = self.EE_LINK
        oc.orientation = target.pose.orientation
        oc.absolute_x_axis_tolerance = float(ori_tol)
        oc.absolute_y_axis_tolerance = float(ori_tol)
        oc.absolute_z_axis_tolerance = float(ori_tol)
        oc.weight = 1.0
        c.orientation_constraints = [oc]
        return c

    # ============================================================
    # (2) Cartesian 이동 - 인터페이스 변경
    # ============================================================
    def move_cartesian(
            self,
            item,           # [x,y,z] (Numpy/List) 혹은 [Pose, Pose...] (List)
            quat=None,      # [x,y,z,w] (Numpy/List/Msg). item이 Waypoints 리스트면 None
            collision: bool = True,
            max_step: float = 0.01,
            min_fraction: float = 0.95
        ) -> bool:
            """
            사용법 1: 단일 목표점으로 이동 (XYZ, Quat 직접 입력)
            -> move_cartesian(center_pos, grasp_quat, collision=False)
            
            사용법 2: 여러 경유지(Waypoints) 이동
            -> move_cartesian([pose1, pose2], collision=True)
            """
            
            final_waypoints = []

            # [Case A] 첫 번째 인자가 Pose 객체의 리스트인 경우 (기존 Lying 로직 호환)
            if isinstance(item, list) and len(item) > 0 and isinstance(item[0], Pose):
                final_waypoints = item
                
            # [Case B] 두 번째 인자(quat)가 들어온 경우 -> 단일 목표점 이동 (요청하신 형태)
            elif quat is not None:
                # item을 xyz로, quat을 orientation으로 간주하여 Pose 생성
                target_pose = self._build_pose(item, quat)
                final_waypoints = [target_pose]

            else:
                self.get_logger().error("[move_cartesian] Invalid arguments. Provide (xyz, quat) or (waypoints_list).")
                return False

            # 계획 및 실행
            traj, frac = self.plan_cartesian(
                final_waypoints,
                avoid_collisions=collision,
                max_step=max_step
            )
            
            if traj is None or frac < min_fraction:
                self.get_logger().warn(f"Cartesian fraction low: {frac:.3f}")
                return False
                
            return self.execute_trajectory(traj)

    # [NEW] 편의 함수: 현재 위치에서 특정 좌표로 직선 이동 (Cartesian)
    def move_cartesian_line(self, xyz, quat, collision: bool = True, step: float=0.01) -> bool:
        target_pose = self._build_pose(xyz, quat)
        return self.move_cartesian([target_pose], collision=collision, max_step=step)

    def plan_cartesian(
        self,
        waypoints: List[Pose],
        start_js: Optional[JointState] = None,
        max_step: float = 0.01,
        jump_threshold: float = 0.0,
        avoid_collisions: bool = True,
    ) -> Tuple[Optional[RobotTrajectory], float]:
        
        if not self.cart_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Cartesian service not available")
            return None, 0.0

        req = GetCartesianPath.Request()
        req.header.frame_id = self.BASE_FRAME
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = self.GROUP_NAME
        req.link_name = self.EE_LINK
        req.max_step = float(max_step)
        req.jump_threshold = float(jump_threshold)
        req.avoid_collisions = bool(avoid_collisions)
        req.waypoints = waypoints

        if start_js is None:
            req.start_state = RobotState()
            req.start_state.is_diff = True
        else:
            rs = RobotState()
            rs.joint_state = start_js
            rs.is_diff = False
            req.start_state = rs

        future = self.cart_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if not future.done() or future.result() is None:
            return None, 0.0

        resp = future.result()
        return resp.solution, float(resp.fraction)

    # ============================================================
    # (2) 조인트 값으로 이동: plan (MoveGroup, JointConstraints)
    # ============================================================
    def plan_to_joint_values(
        self,
        joint_goal: Dict[str, float],
        tol: float = 0.001,
        planning_time: float = 5.0,
        attempts: int = 10,
        vel_scale: float = 0.3,
        acc_scale: float = 0.3,
        start_state: Optional[RobotState] = None,
    ) -> Optional[RobotTrajectory]:
        
        # 1. Action Server 연결 확인
        if not self.move_ac.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("MoveGroup action server not available")
            return None

        # 2. Goal 설정
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.GROUP_NAME
        req.num_planning_attempts = int(attempts)
        req.allowed_planning_time = float(planning_time)
        req.max_velocity_scaling_factor = float(vel_scale)
        req.max_acceleration_scaling_factor = float(acc_scale)

        # 3. Start State 설정
        if start_state is None:
            req.start_state = RobotState()
            req.start_state.is_diff = True # 현재 상태에서 시작
        else:
            req.start_state = start_state

        # 4. Joint Constraints 생성 (핵심 로직)
        c = Constraints()
        c.name = "joint_target"
        
        for name, pos in joint_goal.items():
            jc = JointConstraint()
            jc.joint_name = str(name)      # 예: "joint_1"
            jc.position = float(pos)       # 예: 1.622
            jc.tolerance_above = float(tol)
            jc.tolerance_below = float(tol)
            jc.weight = 1.0
            c.joint_constraints.append(jc)

        req.goal_constraints = [c]

        # 5. 계획만 수행 (Plan Only)
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False

        # 6. Action 보내기
        fut = self.move_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        
        if not fut.done() or fut.result() is None:
            self.get_logger().error("MoveGroup send_goal failed")
            return None

        gh = fut.result()
        if not gh.accepted:
            self.get_logger().error("MoveGroup goal rejected")
            return None

        # 7. 결과 대기
        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=float(planning_time) + 2.0)
        
        if not res_fut.done() or res_fut.result() is None:
            self.get_logger().error("MoveGroup result timeout/failed")
            return None

        res = res_fut.result().result
        
        # 8. Trajectory 추출 (MoveIt 에러 코드 확인은 생략하고 궤적 유무로 판단)
        traj = getattr(res, "planned_trajectory", None)
        if traj is None or not traj.joint_trajectory.points:
            self.get_logger().warn("Joint plan failed or empty")
            return None

        return traj

    def move_to_joint_values(self, joint_goal: Dict[str, float], **kwargs) -> bool:
        """
        사용법:
        move_to_joint_values({"joint_1": 1.5, "joint_2": 0.0}, vel_scale=0.5)
        """
        traj = self.plan_to_joint_values(joint_goal, **kwargs)
        if traj is None:
            return False
        return self.execute_trajectory(traj)
    
    # ============================================================
    # 공통: Execute / Utils
    # ============================================================
    def execute_trajectory(self, traj: RobotTrajectory, timeout_sec: float = 30.0) -> bool:
        if not self.exec_ac.wait_for_server(timeout_sec=2.0):
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj
        future = self.exec_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            return False
        gh = future.result()
        if not gh.accepted:
            return False
        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=timeout_sec)
        if not res_fut.done() or res_fut.result() is None:
            return False
        return True

    def joint_state_from_traj_end(self, traj: RobotTrajectory) -> JointState:
        js = JointState()
        jt = traj.joint_trajectory
        if jt.points:
            js.name = list(jt.joint_names)
            js.position = list(jt.points[-1].positions)
        return js

    # ============================================================
    # 계산 유틸리티 (Quaternion, Grasp 등)
    # ============================================================
    def make_grasp_quat_for_approach(self, approach_vec: np.ndarray, obj_axis_x: np.ndarray) -> Quaternion:
        """
        approach_vec: 접근 벡터 (World 기준)
        obj_axis_x: 물체의 기준 축 (World 기준)
        """
        approach = np.array(approach_vec, dtype=float)
        norm = np.linalg.norm(approach)
        if norm < 1e-12:
            return Quaternion(w=1.0)
        
        # 그리퍼가 물체를 바라보는 방향(-Z)이 approach 벡터와 같아야 함 -> Z축은 -approach
        look_dir = - (approach / norm)
        return self._quat_gripper_z_align(look_dir, ref_up=obj_axis_x)

    def _quat_gripper_z_align(self, vz_world: np.ndarray, ref_up: Optional[np.ndarray] = None) -> Quaternion:
        z = np.array(vz_world, dtype=float)
        z /= np.linalg.norm(z)

        if ref_up is None:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array(ref_up, dtype=float)
            if np.linalg.norm(up) < 1e-9:
                up = np.array([0.0, 0.0, 1.0])
            else:
                up /= np.linalg.norm(up)

        if abs(float(np.dot(up, z))) > 0.95:
            up = np.array([0.0, 1.0, 0.0])

        x = np.cross(up, z)
        if np.linalg.norm(x) < 1e-12:
            up = np.array([1.0, 0.0, 0.0])
            x = np.cross(up, z)
        
        x /= np.linalg.norm(x)
        y = np.cross(z, x)

        R_mat = np.column_stack((x, y, z))
        q = R.from_matrix(R_mat).as_quat() # (x, y, z, w)
        return Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))


def main():
    rclpy.init()
    node = MoveItMoveHelper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()