#!/usr/bin/env python3
import json
import time
import math
import threading
import numpy as np
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from control_msgs.action import GripperCommand
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point, Quaternion,Pose

from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
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
from scipy.spatial.transform import Rotation as R
from moveit_helper_functions import MoveItMoveHelper as helper
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

# ==========================================
# [설정]
# ==========================================
BASE_FRAME = "base_link"
GRIPPER_TOPIC = "/gripper_controller/gripper_cmd"

GRIPPER_OPEN  = 0.05
GRIPPER_CLOSE = 0.00

OFFSET_DIST   = 0.3

# ==========================================
# [Main Class]
# ==========================================
class SmartGraspNode(Node):
    def __init__(self):
        super().__init__("smart_grasp_node")
        self.helper=helper()
        self.cb_group = ReentrantCallbackGroup()

        self.sub = self.create_subscription(
            String, "/grasp_planner/target_info", self.on_target, 10, callback_group=self.cb_group
        )
        self.cart_cli = self.create_client(GetCartesianPath, "/compute_cartesian_path", callback_group=self.cb_group)
        self.exec_cli = ActionClient(self, ExecuteTrajectory, "/execute_trajectory", callback_group=self.cb_group)
        self.gripper_ac = ActionClient(self, GripperCommand, GRIPPER_TOPIC, callback_group=self.cb_group)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.lock = threading.Lock()
        self.is_busy = False
        self.get_logger().info(">>> Smart Grasp Node Ready (Clean Mode) <<<")

    def on_target(self, msg):
        with self.lock:
            if self.is_busy:
                return
            self.is_busy = True

        threading.Thread(target=self.process_request, args=(msg.data,), daemon=True).start()

    def process_request(self, json_data):
        try:
            data = json.loads(json_data)
            pos_dict = data["pose"]["position"]
            ori_dict = data["pose"]["orientation"]
            status = data.get("status", "Standing")
            center_pos = np.array([pos_dict["x"], pos_dict["y"], pos_dict["z"]], dtype=float)

            rot_mat = R.from_quat(
                [ori_dict["x"], ori_dict["y"], ori_dict["z"], ori_dict["w"]]
            ).as_matrix()

                    # TF broadcast
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = BASE_FRAME
            t.child_frame_id = "grasp_target"
            t.transform.translation.x = pos_dict["x"]
            t.transform.translation.y = pos_dict["y"]
            t.transform.translation.z = pos_dict["z"]
            t.transform.rotation.x = ori_dict["x"]
            t.transform.rotation.y = ori_dict["y"]
            t.transform.rotation.z = ori_dict["z"]
            t.transform.rotation.w = ori_dict["w"]
            self.tf_broadcaster.sendTransform(t)
            self.helper.move_to_joint_values(joint_goal={
                "joint_1": 0.1945,
                "joint_2": 0.1722,
                "joint_3": 1.6341,
                "joint_4": 0.0021,
                "joint_5": 1.3097,
                "joint_6": -1.3113
            })
            self.get_logger().info(f"Target Received: {status}")
            self.last_status=status
            success = False

            if status == "Flipped":
                self.get_logger().info("Flipped")             
                success = self.handle_flipped(center_pos, rot_mat)
            elif status == "Standing":
                self.get_logger().info("Standing")
                self.operate_gripper(GRIPPER_OPEN)
                time.sleep(0.3)    
                success = self.handle_standing(center_pos, rot_mat)
            elif status == "Lying":
                self.get_logger().info("Lying")
                self.operate_gripper(GRIPPER_OPEN)
                time.sleep(0.3)    
                success = self.handle_lying(center_pos, rot_mat)
            time.sleep(0.3)
            
            # if success:
            #     self.get_logger().info(">>> 작업 성공 <<<")
            #     self.handle_convey(convey_pos=np.array([0.5, 0.0, 0.3]), grasp_quat=np.array([0.0,1.0,0.0,0.0]))
            #     time.sleep(2.0)
            # else:
            #     self.get_logger().error(">>> 작업 실패 <<<")

        except Exception as e:
            self.get_logger().error(f"Logic Error: {e}")
        finally:
            with self.lock:
                self.is_busy = False

# =========================================================================
    # [1] Flipped
    # =========================================================================
    def handle_flipped(self, center_pos, rot_mat,bin_center=[0.0,0.8]):
        self.last_status= "Flipped"
        self.get_logger().info(str(bin_center))
        self.get_logger().info("[Flipped] Strategy Start with Multi-Vectors")

        # 2. 파라미터 설정
        OFFSET_DIST = 0.25  # 접근 대기 거리
        obj_local_x = -rot_mat[:3, 0] 
        approach_vec=np.array([0.0,0.0,1.0])
        # P1: 접근 대기 위치, P2: 진입(Grasp) 위치
        p1_approach_pos = center_pos + (approach_vec * OFFSET_DIST)
        p2_grasp_pos    = center_pos

        # 쿼터니언 계산
        grasp_quat = self.helper.make_grasp_quat_for_approach(approach_vec, obj_local_x)

        # ---------------------------------------------------------
        # [Step 1] 일단 접근 위치(P1)로 이동 시도
        # ---------------------------------------------------------
        self.operate_gripper(GRIPPER_CLOSE)
        time.sleep(0.3)

        # P1 이동 (충돌 감지 ON)
        if not self.helper.move_cartesian([self.helper._build_pose(p1_approach_pos, grasp_quat)], collision=True):
            self.get_logger().info(f"[Standing] Vector approach(P1) failed/collided. Next.")

        # ---------------------------------------------------------
        # [Step 2] 진입 위치(P2)로 이동 시도 (여기가 핵심 기준)
        # ---------------------------------------------------------
        # P2 진입 (충돌 감지 ON - 물체나 주변 방해물 확인)
        if not self.helper.move_cartesian([self.helper._build_pose(p2_grasp_pos, grasp_quat)], collision=True,min_fraction=0.95):
            self.get_logger().warn(f"[Standing] Vector insertion(P2) failed.")
            time.sleep(0.5)

            bin_vec= np.array([bin_center[0], bin_center[1], center_pos[2]]) - center_pos
            bin_vec = bin_vec / np.linalg.norm(bin_vec)

            approach_vec=obj_local_x + 0.25*bin_vec
            approach_vec = approach_vec / np.linalg.norm(approach_vec)

            p1_approach_pos = center_pos + (approach_vec * OFFSET_DIST) - (0.01*bin_vec)
            p2_grasp_pos    = center_pos - (0.01*bin_vec)
            grasp_quat = self.helper.make_grasp_quat_for_approach(approach_vec, obj_local_x)
            wps= [self.helper._build_pose(pos, grasp_quat) for pos in [p1_approach_pos, p2_grasp_pos]]

            self.helper.move_cartesian(wps, collision=False)

        # ---------------------------------------------------------
        # [Step 3] 진입 성공 시, 작업 수행 (Sequence)
        # ---------------------------------------------------------
        self.get_logger().info(f"[Standing] Vector Success! Executing Grasp Sequence.")

        # 1. 그리퍼 열기 (물체 잡기 준비 or 놓기 등 로직에 맞춰 조정)
        self.operate_gripper(GRIPPER_OPEN)
        time.sleep(0.8)

        # 2. 다시 P1(접근 위치)으로 후퇴 (물체를 문 상태일 수도 있음)
        if not self.helper.move_cartesian([self.helper._build_pose(p1_approach_pos, grasp_quat)], collision=False):
            # 후퇴 실패 시 위험하므로 여기서 로직 종료 혹은 에러 처리
            return False
        
        return True # 전체 성공
    # =========================================================================
    # [2] Standing
    # =========================================================================
    def handle_standing(self, center_pos, rot_mat,bin_center=[0.0,0.8]): 
        self.last_status= "Flipped"
        self.get_logger().info(str(bin_center))
        self.get_logger().info("[Flipped] Strategy Start with Multi-Vectors")

        # 2. 파라미터 설정
        OFFSET_DIST = 0.25  # 접근 대기 거리
        INSERT_DIST = 0.0 # 진입 깊이 (중심으로부터의 거리)
        obj_local_x = rot_mat[:3, 0] 
        approach_vec=np.array([0.0,0.0,1.0])
        # P1: 접근 대기 위치, P2: 진입(Grasp) 위치
        p1_approach_pos = center_pos + (approach_vec * OFFSET_DIST)
        p2_grasp_pos    = center_pos

        # 쿼터니언 계산
        grasp_quat = self.helper.make_grasp_quat_for_approach(approach_vec, obj_local_x)

        # ---------------------------------------------------------
        # [Step 1] 일단 접근 위치(P1)로 이동 시도
        # ---------------------------------------------------------
        self.operate_gripper(GRIPPER_CLOSE)
        time.sleep(0.3)

        # P1 이동 (충돌 감지 ON)
        if not self.helper.move_cartesian([self.helper._build_pose(p1_approach_pos, grasp_quat)], collision=True):
            self.get_logger().info(f"[Standing] Vector approach(P1) failed/collided. Next.")

        # ---------------------------------------------------------
        # [Step 2] 진입 위치(P2)로 이동 시도 (여기가 핵심 기준)
        # ---------------------------------------------------------
        # P2 진입 (충돌 감지 ON - 물체나 주변 방해물 확인)
        if not self.helper.move_cartesian([self.helper._build_pose(p2_grasp_pos, grasp_quat)], collision=True):
            self.get_logger().warn(f"[Standing] Vector insertion(P2) failed.")
            time.sleep(0.5)

            bin_vec= np.array([bin_center[0], bin_center[1], center_pos[2]]) - center_pos
            bin_vec = bin_vec / np.linalg.norm(bin_vec)

            approach_vec=obj_local_x + 0.25*bin_vec
            approach_vec = approach_vec / np.linalg.norm(approach_vec)

            p1_approach_pos = center_pos + (approach_vec * OFFSET_DIST) - (0.01*bin_vec)
            p2_grasp_pos    = center_pos - (0.01*bin_vec)
            grasp_quat = self.helper.make_grasp_quat_for_approach(approach_vec, obj_local_x)
            wps= [self.helper._build_pose(pos, grasp_quat) for pos in [p1_approach_pos, p2_grasp_pos]]

            self.helper.move_cartesian(wps, collision=False)

        # ---------------------------------------------------------
        # [Step 3] 진입 성공 시, 작업 수행 (Sequence)
        # ---------------------------------------------------------
        self.get_logger().info(f"[Standing] Vector Success! Executing Grasp Sequence.")

        # 1. 그리퍼 열기 (물체 잡기 준비 or 놓기 등 로직에 맞춰 조정)
        self.operate_gripper(GRIPPER_OPEN)
        time.sleep(0.8)

        # 2. 다시 P1(접근 위치)으로 후퇴 (물체를 문 상태일 수도 있음)
        if not self.helper.move_cartesian([self.helper._build_pose(p1_approach_pos, grasp_quat)], collision=False):
            # 후퇴 실패 시 위험하므로 여기서 로직 종료 혹은 에러 처리
            return False
        
        return True # 전체 성공
    # =========================================================================
    # [3] Lying
    # =========================================================================
    def handle_lying(self, center_pos, rot_mat):
        """
        center_pos: np.array [x, y, z]
        rot_mat: np.array 3x3 rotation matrix
        helper: MoveItMoveHelper 인스턴스
        """
        self.last_status = "Lying"
        self.get_logger().info("[Lying] Start operation...")

        # --- [Step 0] 설정 변수들 ---
        PRE_GRASP_DIST = 0.25      # 접근 전 대기 거리 (m)
        RETREAT_DIST = 0.25        # 후퇴 거리 (m)
        # 층(Layer)을 형성하기 위한 오프셋 목록
        inner_offsets = np.arange(0.0, 0.135, 0.001) 

        # --- [Step 1] 접근 벡터 후보군 계산 ---
        candidates = []
        obj_axis_x = rot_mat[:3, 0]
        world_z = np.array([0.0, 0.0, 1.0])
        
        # 평면 법선 벡터 계산
        plane_normal = np.cross(obj_axis_x, world_z)
        norm_val = np.linalg.norm(plane_normal)
        if norm_val < 1e-6:
            plane_normal = np.array([1.0, 0.0, 0.0])
        else:
            plane_normal /= norm_val

        for deg in range(0, 360, 5):
            rad = np.deg2rad(deg)

            # [Case A] Roll 방식 접근 벡터
            v_local = np.array([0.0, np.cos(rad), np.sin(rad)], dtype=float)
            v_roll = rot_mat @ v_local
            if v_roll[2] >= np.sin(np.deg2rad(25)): 
                candidates.append((abs(v_roll[2]), v_roll, deg))

            # [Case B] Pitch 방식 접근 벡터
            rot_obj = R.from_rotvec(plane_normal * rad)
            v_pitch = rot_obj.apply(obj_axis_x)
            if v_pitch[2] >= np.sin(np.deg2rad(25)):
                candidates.append((abs(v_pitch[2]), v_pitch, deg + 1000))

        # Z축 성분이 큰 순서(위에서 아래로 찍는 형태 선호)로 정렬
        candidates.sort(key=lambda x: x[0], reverse=True)

        # --- [Step 2] 실행 루프 ---
        for score, target_vec, deg in candidates:
            # 접근을 위한 그리퍼 쿼터니언 계산
            grasp_quat = self.helper.make_grasp_quat_for_approach(target_vec, obj_axis_x)
            
            # 그리퍼 열기
            self.operate_gripper(GRIPPER_OPEN)
            
            # 설정한 오프셋(깊이)들을 순차적으로 시도
            for dist_val in inner_offsets:
                # [Phase 1] 접근 (Pre-grasp -> Grasp Position)
                p1_pos = center_pos + (target_vec * PRE_GRASP_DIST)
                p2_pos = center_pos + (target_vec * dist_val)                
                wps = [self.helper._build_pose(pos, grasp_quat) for pos in [p1_pos, p2_pos]]

                # 충돌 체크를 포함한 카르테시안 이동
                if self.helper.move_cartesian(wps, collision=True, min_fraction=0.92):
                    self.get_logger().info(f"Approach Success! Deg: {deg}, Offset: {dist_val}")
                    
                    # [Phase 2] 그리핑
                    self.operate_gripper(GRIPPER_CLOSE)
                    time.sleep(0.8)

                    # [Phase 3] 후퇴 (들어온 방향 그대로 빠짐)
                    p3_pos = center_pos + (target_vec * RETREAT_DIST)
                    p3 = self.helper._build_pose(p3_pos, grasp_quat)

                    if self.helper.move_cartesian([p3], collision=False, min_fraction=0.95):
                        self.get_logger().info("Retreat Success.")
                        return True
                    else:
                        self.get_logger().warn("Retreat Failed. Releasing.")
                        self.operate_gripper(GRIPPER_OPEN)
                else:
                    # 접근 실패 시 다음 오프셋이나 후보군으로 넘어감
                    pass

        self.get_logger().error("All candidates failed.")
        return False

# =========================================================================
# [4] Convey
# =========================================================================
    def handle_convey(self, convey_pos, grasp_quat=np.array([0.0,1.0,0.0,0.0])):
        self.get_logger().info("[Convey] Strategy Start")

        if not self.helper.move_cartesian(convey_pos, grasp_quat,collision=False):
            return False

        if not self.helper.move_cartesian(convey_pos, grasp_quat, collision=False):
            return False
        
        if self.last_status == "Flipped" or self.last_status == "Standing":
            self.operate_gripper(GRIPPER_CLOSE)
            time.sleep(0.4)
        else:
            self.operate_gripper(GRIPPER_OPEN)
            time.sleep(0.4)

        return True
    
    def operate_gripper(self, pos):
        if not self.gripper_ac.wait_for_server(timeout_sec=1.0):
            return
        goal = GripperCommand.Goal()
        goal.command.position = float(pos)
        goal.command.max_effort = 100.0
        self.gripper_ac.send_goal_async(goal)


def main():
    rclpy.init()
    node = SmartGraspNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()