#!/usr/bin/env python3
import json
import threading
import time
import numpy as np
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import String
from geometry_msgs.msg import Point, Quaternion, PoseStamped

from sensor_msgs.msg import JointState

from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import RobotState

from scipy.spatial.transform import Rotation as R

# ✅ RViz 마커
from visualization_msgs.msg import Marker, MarkerArray

# ==========================================
# [설정]
# ==========================================
BASE_FRAME = "base_link"

# MoveIt 설정 (환경에 맞게 수정)
MOVEIT_CART_SERVICE = "/compute_cartesian_path"
MOVE_GROUP = "manipulator"
EE_LINK = "link_6"

# 경로 탐색 설정
OFFSET_START = 0.15
OFFSET_END   = 0.3
OFFSET_STEP  = 0.05

# “양옆 3개” = 각도 step(기존 5deg 스캔) 기준으로 ±(1~3)step
ANGLE_STEP_DEG = 5
SIDE_STEPS = 10

# grasp 쪽으로 얼마나 더 들어갈지(원하시면 0으로 둬도 됨)
GRASP_INSET = 0.00

# CartesianPath 파라미터
CART_MAX_STEP = 0.1
CART_JUMP_THRESH = 0.0
AVOID_COLLISIONS = True

# ✅ MoveIt 요청 템포 조절 (너무 빠른 연속 요청 방지)
PLAN_WARMUP_SEC = 0.15        # target 받은 직후 첫 요청 전에 한 번 대기
PLAN_CALL_DELAY_SEC = 0.08    # 각 CartesianPath 요청 사이 최소 간격(추천 0.05~0.15)

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


class SmartGraspNode(Node):
    def __init__(self):
        super().__init__("smart_grasp_node")
        self.cb_group = ReentrantCallbackGroup()

        self.sub = self.create_subscription(
            String,
            "/grasp_planner/target_info",
            self.on_target,
            10,
            callback_group=self.cb_group
        )

        # 현재 로봇 상태(관절) 확보용
        self.js_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.on_joint_state,
            50,
            callback_group=self.cb_group
        )

        # ✅ (유지) joint_states 정상이라고 하셨으니 그대로 사용
        self.last_js: Optional[JointState] = None

        # MoveIt CartesianPath 서비스
        self.cart_cli = self.create_client(GetCartesianPath, MOVEIT_CART_SERVICE, callback_group=self.cb_group)
        if not self.cart_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"[MoveIt] Service not ready: {MOVEIT_CART_SERVICE} (나중에 뜨면 자동 사용됩니다)")

        self.debug_pub = self.create_publisher(MarkerArray, DEBUG_TOPIC, 10)

        self.lock = threading.Lock()
        self.is_busy = False
        self.get_logger().info(">>> Smart Grasp Node Ready (LYING ONLY / MOVEIT PLAN + DEBUG) <<<")

    def on_joint_state(self, msg: JointState):
        self.last_js = msg

    def on_target(self, msg: String):
        with self.lock:
            if self.is_busy:
                return
            self.is_busy = True
        threading.Thread(target=self.process_request, args=(msg.data,), daemon=True).start()

    def process_request(self, json_data: str):
        try:
            data = json.loads(json_data)
            pos_dict = data["pose"]["position"]
            ori_dict = data["pose"]["orientation"]

            center_pos = np.array([pos_dict["x"], pos_dict["y"], pos_dict["z"]], dtype=float)
            rot_mat = R.from_quat([ori_dict["x"], ori_dict["y"], ori_dict["z"], ori_dict["w"]]).as_matrix()

            # ✅ status 무시하고 Lying만 처리
            self.get_logger().info("Target Received -> FORCING [Lying] logic only")
            self.handle_lying_with_moveit(center_pos, rot_mat)

        except Exception as e:
            self.get_logger().error(f"Logic Error: {e}")
        finally:
            with self.lock:
                self.is_busy = False

    # =========================================================================
    # [핵심] 그리퍼 정렬 로직 (그리퍼 Z축 정렬)
    # =========================================================================
    def make_grasp_quat_for_approach(
        self,
        approach_vec: np.ndarray,
        ref_up: Optional[np.ndarray] = None
    ) -> Quaternion:
        look_dir = -approach_vec

        if ref_up is None:
            up_vec = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            up_vec = np.array(ref_up, dtype=float)

        return self.quat_gripper_z_align(look_dir, up_vec)

    def quat_gripper_z_align(self, vz_target: np.ndarray, ref_up: np.ndarray) -> Quaternion:
        z = np.array(vz_target, dtype=float)
        zn = np.linalg.norm(z)
        if zn < 1e-9:
            return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        z /= zn

        up = np.array(ref_up, dtype=float)
        upn = np.linalg.norm(up)
        if upn < 1e-9:
            up = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            up /= upn

        if abs(np.dot(up, z)) > 0.99:
            up = np.array([1.0, 0.0, 0.0], dtype=float)

        x = np.cross(up, z)
        xn = np.linalg.norm(x)
        if xn < 1e-9:
            x = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            x /= xn

        y = np.cross(z, x)
        R_ee = np.column_stack((x, y, z))

        q = R.from_matrix(R_ee).as_quat()  # [x,y,z,w]
        return Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))

    # =========================================================================
    # [MoveIt] Cartesian Path 호출
    # =========================================================================
    def _plan_cartesian(self, waypoints: List[PoseStamped]) -> Tuple[float, Optional[GetCartesianPath.Response]]:
        if not self.cart_cli.service_is_ready():
            return 0.0, None
        if self.last_js is None:
            return 0.0, None

        req = GetCartesianPath.Request()
        req.header.frame_id = BASE_FRAME
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = MOVE_GROUP
        req.link_name = EE_LINK
        req.max_step = float(CART_MAX_STEP)
        req.jump_threshold = float(CART_JUMP_THRESH)
        req.avoid_collisions = bool(AVOID_COLLISIONS)

        # ✅ 현재 상태를 start_state로 반영 (유지)
        rs = RobotState()
        rs.joint_state = self.last_js
        req.start_state = rs

        # PoseStamped -> Pose
        req.waypoints = [wp.pose for wp in waypoints]

        fut = self.cart_cli.call_async(req)
        t0 = time.time()
        while rclpy.ok() and (not fut.done()):
            time.sleep(0.002)
            if time.time() - t0 > 3.0:
                return 0.0, None

        resp = fut.result()
        if resp is None:
            return 0.0, None

        return float(resp.fraction), resp

    def _make_pose_stamped(self, pos_xyz: np.ndarray, quat: Quaternion) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = BASE_FRAME
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position = Point(x=float(pos_xyz[0]), y=float(pos_xyz[1]), z=float(pos_xyz[2]))
        ps.pose.orientation = quat
        return ps

    # =========================================================================
    # [Lying + MoveIt 탐색]
    # =========================================================================
    def handle_lying_with_moveit(self, center_pos: np.ndarray, rot_mat: np.ndarray):
        self.get_logger().info("[Lying] MoveIt Cartesian planning with debug markers...")

        # ✅ 추가: target 받은 직후 첫 요청 전에 워밍업 대기
        time.sleep(PLAN_WARMUP_SEC)

        obj_x_axis = rot_mat[:, 0]

        cand_by_deg = {}
        scored: List[Tuple[float, int, np.ndarray]] = []

        for deg in range(0, 360, ANGLE_STEP_DEG):
            rad = np.deg2rad(deg)
            v_local = np.array([0.0, np.cos(rad), np.sin(rad)], dtype=float)
            v_world = rot_mat @ v_local

            if abs(v_world[2]) < np.sin(np.deg2rad(15)):
                continue

            score = abs(v_world[2])
            cand_by_deg[deg] = (v_world, score)
            scored.append((score, deg, v_world))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            self.get_logger().warn("[Lying] No candidates found.")
            return

        best_score, best_deg, best_vec = scored[0]
        self.get_logger().info(f"[Lying] Best score={best_score:.3f}, best_deg={best_deg} deg")

        offsets = []
        o = OFFSET_START
        while o <= OFFSET_END + 1e-9:
            offsets.append(round(o, 3))
            o += OFFSET_STEP

        degs_to_try = [best_deg]
        for k in range(1, SIDE_STEPS + 1):
            degs_to_try.append((best_deg + k * ANGLE_STEP_DEG) % 360)
            degs_to_try.append((best_deg - k * ANGLE_STEP_DEG) % 360)

        seen = set()
        ordered_degs = []
        for d in degs_to_try:
            if d in seen:
                continue
            seen.add(d)
            ordered_degs.append(d)

        attempt_records = []
        success_info = None

        for off in offsets:
            for deg in ordered_degs:
                if deg not in cand_by_deg:
                    continue
                v_world, score = cand_by_deg[deg]

                pre_pos = center_pos + v_world * float(off)
                grasp_pos = center_pos + v_world * float(GRASP_INSET)

                quat = self.make_grasp_quat_for_approach(-v_world, ref_up=obj_x_axis)

                pre = self._make_pose_stamped(pre_pos, quat)
                grasp = self._make_pose_stamped(grasp_pos, quat)

                # ✅ pre -> grasp 를 8개 waypoint로 보간해서 요청 (유지)
                wps = []
                N = 8
                for t in np.linspace(0.0, 1.0, N):
                    p = (1.0 - t) * pre_pos + t * grasp_pos
                    wps.append(self._make_pose_stamped(p, quat))

                fraction, resp = self._plan_cartesian(wps)

                # ✅ 추가: 연속 요청 템포 늦춤
                time.sleep(PLAN_CALL_DELAY_SEC)

                attempt_records.append((off, deg, v_world, fraction))

                self.get_logger().info(f"[Try] off={off:.3f}, deg={deg:3d}, score={score:.3f}, fraction={fraction:.3f}")

                if fraction >= 0.90:
                    success_info = (off, deg, v_world, quat, pre, grasp, fraction)
                    break
            if success_info is not None:
                break

        if DEBUG_ENABLE:
            self._debug_publish_planning_attempts(center_pos, rot_mat, scored, best_deg, attempt_records, success_info)

        if success_info is None:
            self.get_logger().warn("[Result] 모든 시도에서 경로 탐색 실패했습니다.")
        else:
            off, deg, v, quat, pre, grasp, fraction = success_info
            self.get_logger().info(f"[Result] ✅ SUCCESS: off={off:.3f}, deg={deg}, fraction={fraction:.3f}")

    # =========================================================================
    # [DEBUG] 시도 결과 시각화
    # =========================================================================
    def _debug_publish_planning_attempts(
        self,
        center_pos: np.ndarray,
        obj_rot_mat: np.ndarray,
        scored_candidates: List[Tuple[float, int, np.ndarray]],
        best_deg: int,
        attempt_records: List[Tuple[float, int, np.ndarray, float]],
        success_info
    ):
        ma = MarkerArray()
        ma.markers.append(self._debug_deleteall_marker())
        c = np.array(center_pos, dtype=float)

        ma.markers.append(self._make_sphere(1, "ly_pt", c, (1.0, 1.0, 0.0, 1.0), d=0.03))
        self._debug_add_object_axes(ma, c, obj_rot_mat, base_id=10, ns="ly_obj_axes")

        if DEBUG_DRAW_LYING_CIRCLE:
            pts = []
            for deg in range(0, 361, ANGLE_STEP_DEG):
                rad = np.deg2rad(deg)
                v_local = np.array([0.0, np.cos(rad), np.sin(rad)], dtype=float)
                v_world = obj_rot_mat @ v_local
                pts.append(c + v_world * float(OFFSET_START))
            ma.markers.append(self._make_linestrip(20, "ly_circle", pts, (1.0, 1.0, 1.0, 0.35), width=0.002))

        topk = min(DEBUG_LYING_TOPK, len(scored_candidates))
        for i in range(topk):
            score, deg, v = scored_candidates[i]
            p1 = c + v * float(OFFSET_START)
            ma.markers.append(self._make_arrow(
                100 + i, "ly_cand_all", c, p1,
                (0.7, 0.7, 0.7, 0.18),
                shaft_d=0.003, head_d=0.006
            ))

        best_vec = None
        for s, d, v in scored_candidates:
            if d == best_deg:
                best_vec = v
                break
        if best_vec is not None:
            ma.markers.append(self._make_arrow(
                400, "ly_best", c, c + best_vec * float(OFFSET_START),
                (0.2, 0.7, 1.0, 0.9),
                shaft_d=0.006, head_d=0.012
            ))
            ma.markers.append(self._make_text(
                401, "ly_best", c + best_vec * (float(OFFSET_START) + 0.03),
                f"BEST deg={best_deg}", (0.2, 0.7, 1.0, 1.0)
            ))

        for i, (off, deg, v, fraction) in enumerate(attempt_records[:300]):
            p1 = c + v * float(off)
            if fraction >= 0.90:
                rgba = (0.2, 1.0, 0.2, 0.95)
                shaft = 0.010
                head = 0.020
            else:
                rgba = (1.0, 0.2, 0.2, 0.25)
                shaft = 0.005
                head = 0.010
            ma.markers.append(self._make_arrow(
                800 + i, "ly_attempts", c, p1, rgba,
                shaft_d=shaft, head_d=head
            ))

        if success_info is not None:
            off, deg, v, quat, pre, grasp, fraction = success_info
            ma.markers.append(self._make_text(
                1200, "ly_success", c + np.array([0.0, 0.0, 0.12]),
                f"SUCCESS off={off:.3f}, deg={deg}, frac={fraction:.2f}",
                (1.0, 1.0, 1.0, 1.0), h=0.05
            ))

            pre_p = np.array([pre.pose.position.x, pre.pose.position.y, pre.pose.position.z], dtype=float)
            gr_p  = np.array([grasp.pose.position.x, grasp.pose.position.y, grasp.pose.position.z], dtype=float)

            ma.markers.append(self._make_sphere(1210, "ly_success", pre_p, (0.2, 1.0, 0.2, 0.9), d=0.03))
            ma.markers.append(self._make_sphere(1211, "ly_success", gr_p,  (0.2, 0.7, 1.0, 0.9), d=0.03))

            desired_minus_z = -np.array(v, dtype=float)
            self._debug_add_gripper_axes(ma, pre_p, quat, desired_minus_z, base_id=1300, ns="ly_pre_grip")
            self._debug_add_gripper_axes(ma, gr_p,  quat, desired_minus_z, base_id=1400, ns="ly_grasp_grip")

        self.debug_pub.publish(ma)
        self.get_logger().info(">>> [Lying] MoveIt attempt visualization published <<<")

    # ---- Marker Helpers ----
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
        m.points = [
            Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
            Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))
        ]
        m.scale.x = float(shaft_d)
        m.scale.y = float(head_d)
        m.scale.z = float(head_len)
        r, g, b, a = rgba
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.color.a = float(a)
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
        m.scale.x = float(d)
        m.scale.y = float(d)
        m.scale.z = float(d)
        r, g, b, a = rgba
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.color.a = float(a)
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
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.color.a = float(a)
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
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.color.a = float(a)
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
        self._lifetime(m)
        return m

    def _quat_to_rotmat(self, q: Quaternion) -> np.ndarray:
        return R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    def _debug_add_object_axes(self, ma: MarkerArray, origin, rot_mat, base_id=10, ns="obj_axes"):
        if not DEBUG_DRAW_OBJECT_AXES:
            return
        x, y, z = rot_mat[:, 0], rot_mat[:, 1], rot_mat[:, 2]
        p0 = np.array(origin, dtype=float)
        ma.markers.append(self._make_arrow(base_id + 0, ns, p0, p0 + x * DEBUG_OBJECT_AXIS_LEN, (1.0, 0.0, 0.0, 1.0)))
        ma.markers.append(self._make_arrow(base_id + 1, ns, p0, p0 + y * DEBUG_OBJECT_AXIS_LEN, (0.0, 1.0, 0.0, 1.0)))
        ma.markers.append(self._make_arrow(base_id + 2, ns, p0, p0 + z * DEBUG_OBJECT_AXIS_LEN, (0.0, 0.5, 1.0, 1.0)))
        ma.markers.append(self._make_text(base_id + 3, ns, p0 + x * (DEBUG_OBJECT_AXIS_LEN + 0.02), "Xo", (1, 0.2, 0.2, 1)))
        ma.markers.append(self._make_text(base_id + 4, ns, p0 + y * (DEBUG_OBJECT_AXIS_LEN + 0.02), "Yo", (0.2, 1, 0.2, 1)))
        ma.markers.append(self._make_text(base_id + 5, ns, p0 + z * (DEBUG_OBJECT_AXIS_LEN + 0.02), "Zo", (0.2, 0.7, 1, 1)))

    def _debug_add_gripper_axes(self, ma: MarkerArray, origin, grasp_quat: Quaternion, desired_minus_z: np.ndarray,
                               base_id=600, ns="grip_axes"):
        Rg = self._quat_to_rotmat(grasp_quat)
        gx, gy, gz = Rg[:, 0], Rg[:, 1], Rg[:, 2]
        p0 = np.array(origin, dtype=float)

        ma.markers.append(self._make_arrow(base_id + 0, ns, p0, p0 + gx * DEBUG_GRIPPER_AXIS_LEN, (1.0, 0.3, 0.3, 0.9), shaft_d=0.006))
        ma.markers.append(self._make_arrow(base_id + 1, ns, p0, p0 + gy * DEBUG_GRIPPER_AXIS_LEN, (0.3, 1.0, 0.3, 0.9), shaft_d=0.006))
        ma.markers.append(self._make_arrow(base_id + 2, ns, p0, p0 + gz * DEBUG_GRIPPER_AXIS_LEN, (0.3, 0.6, 1.0, 0.9), shaft_d=0.006))

        g_minus_z = -gz
        ma.markers.append(self._make_arrow(base_id + 3, ns, p0, p0 + g_minus_z * (DEBUG_GRIPPER_AXIS_LEN * 1.25),
                                           (1.0, 0.0, 1.0, 1.0), shaft_d=0.01, head_d=0.02))
        ma.markers.append(self._make_text(base_id + 4, ns, p0 + g_minus_z * (DEBUG_GRIPPER_AXIS_LEN * 1.3),
                                          "-Zg (Approach)", (1, 0, 1, 1)))


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
