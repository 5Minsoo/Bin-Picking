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
from geometry_msgs.msg import PoseStamped, Point, Quaternion

from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import RobotState, MoveItErrorCodes 

from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker, MarkerArray

# ==========================================
# [설정]
# ==========================================
LINK_NAME = "link_6"
BASE_FRAME = "base_link" 
GRIPPER_TOPIC = "/gripper_controller/gripper_cmd"

GRIPPER_OPEN  = 0.08
GRIPPER_CLOSE = 0.00
OFFSET_DIST   = 0.3

# ==========================================
# [DEBUG 설정]
# ==========================================
DEBUG_ENABLE = True
DEBUG_TOPIC  = "/smart_grasp/debug_markers"
DEBUG_LIFETIME_SEC = 5.0 

DEBUG_LYING_TOPK = 50       
DEBUG_DRAW_LYING_CIRCLE = True
DEBUG_DRAW_OBJECT_AXES = True
DEBUG_OBJECT_AXIS_LEN = 0.15
DEBUG_GRIPPER_AXIS_LEN = 0.15

# ==========================================
# [Point 헬퍼함수]
# ==========================================
def make_pose(header, position_xyz: np.ndarray, quat_msg) -> PoseStamped:
    """header + (x,y,z) + quaternion 으로 PoseStamped 생성"""
    p = PoseStamped()
    p.header = header
    p.pose.position = Point(
        x=float(position_xyz[0]),
        y=float(position_xyz[1]),
        z=float(position_xyz[2]),
    )
    p.pose.orientation = quat_msg
    return p

def make_waypoints_along_vec(header, center_pos: np.ndarray, target_vec: np.ndarray,
                             offsets, quat_msg):
    """
    center_pos + target_vec*offset 을 여러 개 만들어 waypoints list 반환
    offsets: [0.25, 0.17, 0.25] 같은 iterable
    """
    poses = []
    for off in offsets:
        pt = center_pos + (target_vec * float(off))
        poses.append(make_pose(header, pt, quat_msg).pose)  # GetCartesianPath 는 Pose 리스트를 원함
    return poses

# ==========================================
# [Main Class]
# ==========================================
class SmartGraspNode(Node):
    def __init__(self):
        super().__init__("smart_grasp_node")
        self.cb_group = ReentrantCallbackGroup()

        self.sub = self.create_subscription(
            String, "/grasp_planner/target_info", self.on_target, 10, callback_group=self.cb_group
        )
        self.cart_cli = self.create_client(GetCartesianPath, "/compute_cartesian_path", callback_group=self.cb_group)
        self.exec_cli = ActionClient(self, ExecuteTrajectory, "/execute_trajectory", callback_group=self.cb_group)
        self.gripper_ac = ActionClient(self, GripperCommand, GRIPPER_TOPIC, callback_group=self.cb_group)

        self.debug_pub = self.create_publisher(MarkerArray, DEBUG_TOPIC, 10)

        self.lock = threading.Lock()
        self.is_busy = False
        self.get_logger().info(">>> Smart Grasp Node Ready (DEBUG MODE + EXECUTION + SPEED CONTROL) <<<")

    def on_target(self, msg):
        with self.lock:
            if self.is_busy: return
            self.is_busy = True
        threading.Thread(target=self.process_request, args=(msg.data,), daemon=True).start()

    def process_request(self, json_data):
        try:
            data = json.loads(json_data)
            pos_dict = data["pose"]["position"]
            ori_dict = data["pose"]["orientation"]
            status = data.get("status", "Lying") 

            center_pos = np.array([pos_dict["x"], pos_dict["y"], pos_dict["z"]], dtype=float)
            rot_mat = R.from_quat([ori_dict["x"], ori_dict["y"], ori_dict["z"], ori_dict["w"]]).as_matrix()

            self.get_logger().info(f"Target Received: {status}")

            if status == "Flipped":
                self.handle_flipped(center_pos, rot_mat)
            elif status == "Standing":
                self.handle_standing(center_pos, rot_mat)
            elif status == "Lying":
                self.handle_lying(center_pos, rot_mat)
            else:
                self.handle_standing(center_pos, rot_mat)

        except Exception as e:
            self.get_logger().error(f"Logic Error: {e}")
        finally:
            with self.lock:
                self.is_busy = False

    # =========================================================================
    # [Helper] 속도 조절 함수 (새로 추가됨)
    # =========================================================================
    def scale_trajectory_speed(self, trajectory, speed_factor):
        """
        trajectory: moveit_msgs/RobotTrajectory
        speed_factor: 0.0 ~ 1.0 (예: 0.2 = 20% 속도)
        """
        if speed_factor >= 1.0 or speed_factor <= 0.0:
            return trajectory

        # 시간 배율 (속도가 0.5배면 시간은 2배 걸려야 함)
        time_scaling = 1.0 / speed_factor

        for point in trajectory.joint_trajectory.points:
            # 1. 시간(time_from_start) 늘리기
            total_nanos = point.time_from_start.sec * 1_000_000_000 + point.time_from_start.nanosec
            new_nanos = int(total_nanos * time_scaling)
            
            point.time_from_start.sec = new_nanos // 1_000_000_000
            point.time_from_start.nanosec = new_nanos % 1_000_000_000

            # 2. 속도(velocity) 줄이기
            if point.velocities:
                point.velocities = [v * speed_factor for v in point.velocities]

            # 3. 가속도(acceleration) 줄이기
            if point.accelerations:
                point.accelerations = [a * speed_factor for a in point.accelerations]

        return trajectory

    # =========================================================================
    # [핵심] 그리퍼 정렬 로직
    # =========================================================================
    def make_grasp_quat_for_approach(self, approach_vec: np.ndarray, 
                                     ref_up: Optional[np.ndarray] = None) -> Quaternion:
        look_dir = -approach_vec 
        if ref_up is None:
            up_vec = np.array([0.0, 0.0, 1.0])
        else:
            up_vec = ref_up

        return self.quat_gripper_z_align(look_dir, up_vec)

    def quat_gripper_z_align(self, vz_target: np.ndarray, ref_up: np.ndarray) -> Quaternion:
        z = np.array(vz_target, dtype=float)
        zn = np.linalg.norm(z)
        if zn < 1e-9: return Quaternion(w=1.0)
        z /= zn

        up = np.array(ref_up, dtype=float)
        upn = np.linalg.norm(up)
        if upn < 1e-9: up = np.array([0.0, 0.0, 1.0])
        else: up /= upn

        if abs(np.dot(up, z)) > 0.99:
            up = np.array([1.0, 0.0, 0.0])

        x = np.cross(up, z)
        xn = np.linalg.norm(x)
        if xn < 1e-9: x = np.array([1.0, 0.0, 0.0])
        else: x /= xn
        
        y = np.cross(z, x)
        R_ee = np.column_stack((x, y, z))
        q = R.from_matrix(R_ee).as_quat()
        return Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))

    # =========================================================================
    # [Handlers]
    # =========================================================================

    def handle_lying(self, center_pos, rot_mat):
        """
        누워있는 물체 잡기: 접근 가능한 벡터 후보를 모두 생성하고, 
        성공할 때까지 순차적으로 경로 계획 및 실행을 시도합니다.
        """
        self.get_logger().info("[Lying] Start Handling (Planning & Execution)...")
        
        # --- 0. 실행을 위한 Action Client 확인 ---
        if not self.exec_cli.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("ExecuteTrajectory Action Server not available!")
            return

        # --- 1. 후보군 생성 (모든 각도 탐색) ---
        candidates: List[Tuple[float, np.ndarray, int]] = []
        
        # Lying: 물체의 X축(길이 방향)을 Roll 정렬 기준으로 사용
        obj_x_axis = rot_mat[:, 0]

        # 0~360도를 5도 간격으로 순회
        for deg in range(0, 360, 5): 
            rad = np.deg2rad(deg)
            # v_local: YZ 평면 상의 벡터
            v_local = np.array([0.0, np.cos(rad), np.sin(rad)], dtype=float) 
            v_world = rot_mat @ v_local
            
            # [충돌 방지] 지면(World Z)과 너무 수평이면 바닥 충돌 위험이 크므로 제외
            if abs(v_world[2]) < np.sin(np.deg2rad(15)): 
                continue
            
            # [점수 계산] 수직(Top-down)에 가까울수록 높은 점수
            score = abs(v_world[2]) 
            candidates.append((score, v_world, deg))

        # 점수 높은 순(수직 접근 우선)으로 정렬
        candidates.sort(key=lambda x: x[0], reverse=True)

        if not candidates: 
            self.get_logger().warn("No valid approach vectors found (All collide with floor).")
            return

        self.get_logger().info(f"Generated {len(candidates)} candidates. Trying best ones...")

        # --- 2. 후보군 순회하며 계획 및 실행 시도 ---
        plan_found = False
        outer_offsets=0.25
        inner_offsets = np.arange(0.15, 0.190, 0.005)

        for score, target_vec, deg in candidates:
            if plan_found:break
            # (1) 쿼터니언 생성
            target_quat = self.make_grasp_quat_for_approach(target_vec, ref_up=obj_x_axis)
            for dist_val in inner_offsets:
                current_offsets = [outer_offsets, dist_val, outer_offsets]
                req = GetCartesianPath.Request()
                req.header.frame_id = BASE_FRAME
                req.header.stamp = self.get_clock().now().to_msg()
                req.group_name = "manipulator"
                req.link_name = LINK_NAME
                req.max_step = 0.01       
                req.jump_threshold = 2.0  
                req.avoid_collisions = True # 충돌 무시 테스트 중
                req.start_state = RobotState()
                req.start_state.is_diff = True 
                req.waypoints = make_waypoints_along_vec(
                    req.header,
                    center_pos=center_pos,
                    target_vec=target_vec,
                    offsets=current_offsets,
                    quat_msg=target_quat,
                )
            # (4) 서비스 호출
            future = self.cart_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0) 
            
            if not future.done():
                continue 

            resp = future.result()

            if resp.fraction >= 0.90: 
                self.get_logger().info(f"Plan SUCCESS! Deg: {deg}, Offset: {dist_val}, Score: {score:.3f}")
                
                # (6) 실행 로직 (Action Goal 전송)
                # 여기서 resp.solution.joint_trajectory를 사용하여 ExecuteTrajectory Action 호출
                
                goal_msg = ExecuteTrajectory.Goal()
                goal_msg.trajectory =self.scale_trajectory_speed(resp.solution,0.2)
                
                send_goal_future=self.exec_cli.send_goal_async(goal_msg)
                # 실제로는 여기서 wait_for_result 등을 통해 실행 완료를 기다려야 할 수 있음
                rclpy.spin_until_future_complete(self, send_goal_future)
                goal_handle = send_goal_future.result()
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future)
                plan_found = True # 플래그 설정
                break # Offset 루프 탈출 -> 외부 루프 탈출로 이어짐
                
            else:
                # 실패 시 로그 찍고 다음 Offset 시도
                self.get_logger().debug(f"Plan failed (frac={resp.fraction:.2f}) at Deg={deg}, Offset={dist_val}. Retrying...")

        if not plan_found:
            self.get_logger().error("All candidates and offsets failed.")
        else:
            self.get_logger().info("Grasp execution initiated.")
            return True

    def handle_standing(self, center_pos, rot_mat):
        self.get_logger().info("[Standing] Debugging Visualization...")
        
        obj_x_axis = rot_mat[:, 0]
        approach_vec = obj_x_axis 
        grasp_quat = self.make_grasp_quat_for_approach(-approach_vec, ref_up=np.array([0.0, 0.0, 1.0]))
        
        if DEBUG_ENABLE:
            self._debug_publish_static_grasp(
                center_pos, rot_mat, grasp_quat, approach_vec, 
                ns_prefix="st", label="Standing Side Grasp"
            )

    def handle_flipped(self, center_pos, rot_mat):
        self.get_logger().info("[Flipped] Debugging Visualization...")

        obj_x_axis = rot_mat[:, 0]
        approach_vec=obj_x_axis
        grasp_quat = self.make_grasp_quat_for_approach(-obj_x_axis, ref_up=obj_x_axis)

        if DEBUG_ENABLE:
            self._debug_publish_static_grasp(
                center_pos, rot_mat, grasp_quat, approach_vec, 
                ns_prefix="fl", label="Flipped Top Grasp"
            )

    # =========================================================================
    # [DEBUG Visualization Implementation]
    # =========================================================================
    
    def _debug_publish_static_grasp(self, center_pos, obj_rot_mat, grasp_quat, approach_vec, ns_prefix="static", label="Target"):
        ma = MarkerArray()
        ma.markers.append(self._debug_deleteall_marker())
        c = np.array(center_pos, dtype=float)

        # 1. Center Point
        ma.markers.append(self._make_sphere(1, f"{ns_prefix}_pt", c, (1.0, 0.5, 0.0, 1.0), d=0.03))
        
        # 2. Object Axes
        self._debug_add_object_axes(ma, c, obj_rot_mat, base_id=10, ns=f"{ns_prefix}_obj_axes")

        # 3. Approach Line
        p1 = c + approach_vec * OFFSET_DIST
        ma.markers.append(self._make_arrow(100, f"{ns_prefix}_vec", c, p1, (1.0, 1.0, 1.0, 0.5), shaft_d=0.005))

        # 4. Gripper Axes
        desired_minus_z = -approach_vec
        ma.markers.append(self._make_text(200, f"{ns_prefix}_txt", c + np.array([0,0,0.15]), label, (1,1,1,1)))
        self._debug_add_gripper_axes(ma, c, grasp_quat, desired_minus_z, base_id=300, ns=f"{ns_prefix}_grip_axes")

        self.debug_pub.publish(ma)
        self.get_logger().info(f">>> {label} Visualization Published <<<")

    def _debug_publish_lying(self, center_pos, obj_rot_mat, candidates, grasp_quat_preview, preview_vec):
        ma = MarkerArray()
        ma.markers.append(self._debug_deleteall_marker())
        c = np.array(center_pos, dtype=float)

        # 1. Center Point
        ma.markers.append(self._make_sphere(1, "ly_pts", c, (1.0, 1.0, 0.0, 1.0), d=0.03))
        
        # 2. Object Axes
        self._debug_add_object_axes(ma, c, obj_rot_mat, base_id=10, ns="ly_obj_axes")

        # 3. Reference Circle
        if DEBUG_DRAW_LYING_CIRCLE:
            pts = []
            for deg in range(0, 361, 5):
                rad = np.deg2rad(deg)
                v_local = np.array([0.0, np.cos(rad), np.sin(rad)], dtype=float)
                v_world = obj_rot_mat @ v_local
                pts.append(c + v_world * OFFSET_DIST)
            ma.markers.append(self._make_linestrip(20, "ly_circle", pts, (1.0, 1.0, 1.0, 0.5), width=0.002))

        # 4. Candidates Vectors
        topk = min(DEBUG_LYING_TOPK, len(candidates))
        for i in range(topk):
            score, v, deg = candidates[i]
            p1 = c + v * OFFSET_DIST
            ma.markers.append(self._make_arrow(100+i, "ly_cand", c, p1, (0.7, 0.7, 0.7, 0.3), shaft_d=0.003, head_d=0.006))

        # 5. Preview Gripper Axes (Winner)
        pv = np.array(preview_vec, dtype=float)
        desired_minus_z = -pv
        ma.markers.append(self._make_text(550, "ly_sel", c + np.array([0.0, 0.0, 0.1]), "EXECUTING BEST PATH", (1.0, 1.0, 1.0, 1.0)))
        self._debug_add_gripper_axes(ma, c, grasp_quat_preview, desired_minus_z, base_id=700, ns="ly_grip_axes")

        self.debug_pub.publish(ma)

    # =========================================================================
    # [Helper Functions]
    # =========================================================================
    def _debug_deleteall_marker(self):
        m = Marker()
        m.action = Marker.DELETEALL
        return m

    def _lifetime(self, m: Marker):
        m.lifetime.sec = int(DEBUG_LIFETIME_SEC)
        m.lifetime.nanosec = int((DEBUG_LIFETIME_SEC - int(DEBUG_LIFETIME_SEC)) * 1e9)

    def _make_arrow(self, mid, ns, p0, p1, rgba, shaft_d=0.01, head_d=0.02, head_len=0.03):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(mid)
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.points = [Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                    Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))]
        m.scale.x = float(shaft_d); m.scale.y = float(head_d); m.scale.z = float(head_len)
        r, g, b, a = rgba
        m.color.r = float(r); m.color.g = float(g); m.color.b = float(b); m.color.a = float(a)
        self._lifetime(m)
        return m

    def _make_sphere(self, mid, ns, p, rgba, d=0.03):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(mid)
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
        m.pose.orientation.w = 1.0
        m.scale.x = float(d); m.scale.y = float(d); m.scale.z = float(d)
        r, g, b, a = rgba
        m.color.r = float(r); m.color.g = float(g); m.color.b = float(b); m.color.a = float(a)
        self._lifetime(m)
        return m

    def _make_text(self, mid, ns, p, text, rgba=(1.0, 1.0, 1.0, 1.0), h=0.04):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(mid)
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
        m.pose.orientation.w = 1.0
        m.scale.z = float(h)
        m.text = str(text)
        r, g, b, a = rgba
        m.color.r = float(r); m.color.g = float(g); m.color.b = float(b); m.color.a = float(a)
        self._lifetime(m)
        return m

    def _make_linestrip(self, mid, ns, pts, rgba, width=0.005):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(mid)
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = float(width)
        r, g, b, a = rgba
        m.color.r = float(r); m.color.g = float(g); m.color.b = float(b); m.color.a = float(a)
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
        self._lifetime(m)
        return m

    def _quat_to_rotmat(self, q: Quaternion) -> np.ndarray:
        return R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    def _debug_add_object_axes(self, ma: MarkerArray, origin, rot_mat, base_id=10, ns="obj_axes"):
        if not DEBUG_DRAW_OBJECT_AXES: return
        x, y, z = rot_mat[:, 0], rot_mat[:, 1], rot_mat[:, 2]
        p0 = np.array(origin, dtype=float)
        ma.markers.append(self._make_arrow(base_id+0, ns, p0, p0 + x*DEBUG_OBJECT_AXIS_LEN, (1.0, 0.0, 0.0, 1.0)))
        ma.markers.append(self._make_arrow(base_id+1, ns, p0, p0 + y*DEBUG_OBJECT_AXIS_LEN, (0.0, 1.0, 0.0, 1.0)))
        ma.markers.append(self._make_arrow(base_id+2, ns, p0, p0 + z*DEBUG_OBJECT_AXIS_LEN, (0.0, 0.5, 1.0, 1.0)))
        ma.markers.append(self._make_text(base_id+3, ns, p0 + x*(DEBUG_OBJECT_AXIS_LEN+0.02), "Xo", (1,0.2,0.2,1)))
        ma.markers.append(self._make_text(base_id+4, ns, p0 + y*(DEBUG_OBJECT_AXIS_LEN+0.02), "Yo", (0.2,1,0.2,1)))
        ma.markers.append(self._make_text(base_id+5, ns, p0 + z*(DEBUG_OBJECT_AXIS_LEN+0.02), "Zo", (0.2,0.7,1,1)))

    def _debug_add_gripper_axes(self, ma: MarkerArray, origin, grasp_quat: Quaternion, desired_minus_z: np.ndarray, base_id=600, ns="grip_axes"):
        Rg = self._quat_to_rotmat(grasp_quat)
        gx, gy, gz = Rg[:, 0], Rg[:, 1], Rg[:, 2]
        p0 = np.array(origin, dtype=float)
        
        ma.markers.append(self._make_arrow(base_id+0, ns, p0, p0 + gx*DEBUG_GRIPPER_AXIS_LEN, (1.0, 0.3, 0.3, 0.9), shaft_d=0.006))
        ma.markers.append(self._make_arrow(base_id+1, ns, p0, p0 + gy*DEBUG_GRIPPER_AXIS_LEN, (0.3, 1.0, 0.3, 0.9), shaft_d=0.006))
        ma.markers.append(self._make_arrow(base_id+2, ns, p0, p0 + gz*DEBUG_GRIPPER_AXIS_LEN, (0.3, 0.6, 1.0, 0.9), shaft_d=0.006))
        
        g_minus_z = -gz
        ma.markers.append(self._make_arrow(base_id+3, ns, p0, p0 + g_minus_z*(DEBUG_GRIPPER_AXIS_LEN*1.25), (1.0, 0.0, 1.0, 1.0), shaft_d=0.01, head_d=0.02))
        ma.markers.append(self._make_text(base_id+4, ns, p0 + g_minus_z*(DEBUG_GRIPPER_AXIS_LEN*1.3), "-Zg (Approach)", (1,0,1,1)))

def main():
    rclpy.init()
    node = SmartGraspNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()