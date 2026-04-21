#!/usr/bin/env python3
from pathlib import Path

import json
from typing import List, Dict
import yaml
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from simulation_interfaces.srv import GetEntitiesStates

def short_name(entity_path: str) -> str:
    if not entity_path:
        return "unknown"
    return entity_path.split("/")[-1]

BASE_DIR = Path(__file__).resolve().parent.parent
path=BASE_DIR/ "handeye_result.yaml"

CONFIG_PATH = BASE_DIR / "cfg" / "config.yaml"
with open(CONFIG_PATH, "r") as _f:
    CFG = yaml.safe_load(_f)

class PerceptionBridge(Node):
    def __init__(self):
        super().__init__("perception_bridge")

        # ===== Params =====
        self.entities_srv_name = "/get_entities_states"
        self.rate_hz           = 10.0
        self.world_frame       = "base_link"
        self.filter_regex      = r"^/World/obj_[0-9]+$|^/World/bin/wall_[pn][xy]$"
        self.wall_thick        = 0.02

        # 퍼블리시 토픽 (카메라/Isaac Sim 공통 출력)
        self.pose_topic  = "/perception_bridge/poses"
        self.names_topic = "/perception_bridge/names"
        self.bin_topic   = "/perception_bridge/bin_info"    

        # 카메라 서브스크립션 토픽
        self.camera_poses_topic  = "/camera/poses"
        self.camera_errors_topic = "/camera/errors"
        self.camera_bin_topic    = "/camera/bin_info"

        self.yaml_file_path=path
        with open(self.yaml_file_path, "r") as f:
            data = yaml.safe_load(f)
        self.hand_eye_R = np.array(data["rotation"]).reshape(3, 3)
        self.hand_eye_t = np.array(data["translation"])
        self.T_base_camera=np.eye(4)
        self.T_base_camera[:3,:3]=self.hand_eye_R
        self.T_base_camera[:3,3]=self.hand_eye_t
        qos = QoSProfile(depth=10)

        # ===== Publishers (3개 공통) =====
        self.pub_poses = self.create_publisher(PoseArray, self.pose_topic,  qos)
        self.pub_names = self.create_publisher(String,    self.names_topic, qos)
        self.pub_bin   = self.create_publisher(String,    self.bin_topic,   qos)

        # ===== Camera Subscribers =====
        self.sub_camera_poses  = self.create_subscription(
            PoseArray, self.camera_poses_topic,  self.camera_pose_callback,  qos)
        self.sub_camera_errors = self.create_subscription(
            String,    self.camera_errors_topic, self.camera_error_callback, qos)
        self.sub_camera_bin    = self.create_subscription(
            String,    self.camera_bin_topic,    self.camera_bin_callback,   qos)

        # ===== Isaac Sim Service Client =====
        self.client = self.create_client(GetEntitiesStates, self.entities_srv_name)
        self.get_logger().info(f"Waiting for service: {self.entities_srv_name}")
        if not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Service not available yet...")

        self._inflight = False
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)

        self.get_logger().info(
            f"PerceptionBridge Ready | Regex: {self.filter_regex}\n"
            f"Bin Info -> {self.bin_topic} (Assumed Thickness: {self.wall_thick}m)"
        )
        
        # ============ tf 관련 ====================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer=self.create_timer(0.1,self.tf_publisher)


    # =============================================
    # 카메라 콜백 → 동일한 3개 토픽으로 발행
    # =============================================

    def camera_pose_callback(self, msg: PoseArray):
        offset = np.array(CFG["pose_offset"], dtype=float)
        transformed = PoseArray()
        transformed.header= msg.header
        transformed.header.frame_id = self.world_frame

        now = self.get_clock().now().to_msg()
        tfs = []

        for i, pose in enumerate(msg.poses):

            p_cam = np.array([pose.position.x, pose.position.y, pose.position.z])
            q_cam = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

            R_cam = Rotation.from_quat(q_cam).as_matrix()
            T_cam = np.eye(4)
            T_cam[:3, :3] = R_cam
            T_cam[:3, 3]  = p_cam

            T_world = self.T_base_camera @ T_cam

            new_pose = Pose()
            pos = T_world[:3, 3] + offset
            new_pose.position.x, new_pose.position.y, new_pose.position.z = pos
            q_world = Rotation.from_matrix(T_world[:3, :3]).as_quat()
            new_pose.orientation.x, new_pose.orientation.y, new_pose.orientation.z, new_pose.orientation.w = q_world
            transformed.poses.append(new_pose)

            # TF 추가
            t = TransformStamped()
            t.header.stamp    = now
            t.header.frame_id = self.world_frame
            t.child_frame_id  = f"obj_{i+1}"
            t.transform.translation.x = T_world[:3, 3][0] 
            t.transform.translation.y = T_world[:3, 3][1]
            t.transform.translation.z = T_world[:3, 3][2]
            self.get_logger().info(f'{t.transform.translation}')
            t.transform.rotation.x = q_world[0]
            t.transform.rotation.y = q_world[1]
            t.transform.rotation.z = q_world[2]
            t.transform.rotation.w = q_world[3]
            tfs.append(t)

        self.pub_poses.publish(transformed)
        if tfs:
            self.tf_broadcaster.sendTransform(tfs)

    def camera_error_callback(self, msg: String):
        """
        카메라 에러 수신 (JSON 리스트, 예: [0.012, 0.034, 0.008])
        → 에러 개수만큼 obj_1, obj_2... 생성
        → {"obj_1": 0.012, "obj_2": 0.034, ...} 형식으로 /perception_bridge/names 발행
        """
        try:
            errors: list = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"camera_error_callback JSON 파싱 실패: {e}")
            return

        obj_names = [f"obj_{i}" for i in range(1, len(errors) + 1)]
        paired    = {name: error for name, error in zip(obj_names, errors)}

        out      = String()
        out.data = json.dumps(paired, ensure_ascii=False)
        self.pub_names.publish(out)

        self.get_logger().debug(f"camera names+errors published: {paired}")

    def camera_bin_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            pts_cam = np.array(data["points"], dtype=float)  # shape: (4, 2)

            # --- 4점 전부 로봇 좌표계로 변환 ---
            pts_world = []
            for pt in pts_cam:
                pt_3d = np.array([pt[0], pt[1], 0.0])
                pw = self.T_base_camera @ pt_3d
                pts_world.append(pw[:2])
            pts_world = np.array(pts_world)  # shape: (4, 2)

            # --- center ---
            center = np.mean(pts_world, axis=0)

            # --- yaw (긴 축 기준) ---
            edge1 = pts_world[1] - pts_world[0]
            edge2 = pts_world[3] - pts_world[0]
            len1 = np.linalg.norm(edge1)
            len2 = np.linalg.norm(edge2)
            long_axis = edge1 if len1 >= len2 else edge2
            yaw = np.arctan2(long_axis[1], long_axis[0])  # rad

            # --- size ---
            width  = max(len1, len2)
            height = min(len1, len2)

            transformed = {
                "center": [float(center[0]), float(center[1])],
                "yaw":    float(yaw),
                "size":   [float(width), float(height)]
            }

            out      = String()
            out.data = json.dumps(transformed)
            self.pub_bin.publish(out)

        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().error(f"camera_bin_callback 파싱 실패: {e}")

    # =============================================
    # Isaac Sim 서비스 루프 → 동일한 3개 토픽으로 발행
    # =============================================

    def _make_request(self):
        req = GetEntitiesStates.Request()
        try:
            req.filters.filter = self.filter_regex
        except AttributeError:
            pass
        return req

    def _tick(self):
        if not self.client.service_is_ready() or self._inflight:
            return
        self._inflight = True
        future = self.client.call_async(self._make_request())
        future.add_done_callback(self._on_response)

    def _on_response(self, future):
        self._inflight = False
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            return

        if hasattr(resp, "result") and hasattr(resp.result, "result"):
            if int(resp.result.result) != 1:
                return

        entities = getattr(resp, "entities", [])
        states   = getattr(resp, "states",   [])
        n = min(len(entities), len(states))
        if n == 0:
            return

        now = self.get_clock().now().to_msg()

        pa = PoseArray()
        pa.header.stamp    = now
        pa.header.frame_id = self.world_frame

        names_list: List[str]        = []
        walls_pos:  Dict[str, float] = {}

        for i in range(n):
            e_path = entities[i]
            st     = states[i]
            if not e_path:
                continue

            name = short_name(e_path)

            if name.startswith("wall_"):
                if "px" in name or "nx" in name:
                    walls_pos[name] = st.pose.position.x
                elif "py" in name or "ny" in name:
                    walls_pos[name] = st.pose.position.y

            elif name.startswith("obj_"):
                names_list.append(name)
                pa.poses.append(st.pose)

        # 상자 크기 계산
        bin_info = {}
        if "wall_px" in walls_pos and "wall_nx" in walls_pos:
            px, nx = walls_pos["wall_px"], walls_pos["wall_nx"]
            bin_info["center_x"]    = (px + nx) / 2.0
            bin_info["inner_width"] = abs(px - nx) - self.wall_thick + 0.005

        if "wall_py" in walls_pos and "wall_ny" in walls_pos:
            py, ny = walls_pos["wall_py"], walls_pos["wall_ny"]
            bin_info["center_y"]     = (py + ny) / 2.0
            bin_info["inner_height"] = abs(py - ny) - self.wall_thick + 0.005

        # 퍼블리시 1: poses
        self.pub_poses.publish(pa)

        # 퍼블리시 2: names (단순 리스트)
        msg_names      = String()
        msg_names.data = json.dumps(names_list, ensure_ascii=False)
        self.pub_names.publish(msg_names)

        # 퍼블리시 3: bin_info
        if "center_x" in bin_info and "center_y" in bin_info:
            msg_bin      = String()
            msg_bin.data = json.dumps({
                "center": [bin_info["center_x"], bin_info["center_y"]],
                "size":   [bin_info["inner_width"], bin_info["inner_height"]],
                "yaw": 0
            })
            self.pub_bin.publish(msg_bin)

    def tf_publisher(self):
        cam_tf = TransformStamped()
        cam_tf.header.stamp = self.get_clock().now().to_msg()
        cam_tf.header.frame_id = self.world_frame
        cam_tf.child_frame_id = "camera"
        cam_tf.transform.translation.x = self.T_base_camera[:3,3][0] 
        cam_tf.transform.translation.y = self.T_base_camera[:3,3][1] 
        cam_tf.transform.translation.z = self.T_base_camera[:3,3][2]
        q = Rotation.from_matrix(self.T_base_camera[:3,:3]).as_quat()
        cam_tf.transform.rotation.x, cam_tf.transform.rotation.y = q[0], q[1]
        cam_tf.transform.rotation.z, cam_tf.transform.rotation.w = q[2], q[3]
        self.tf_broadcaster.sendTransform(cam_tf)


def main():
    rclpy.init()
    node = PerceptionBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()