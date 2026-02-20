#!/usr/bin/env python3
import json
import math
from typing import List, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_msgs.msg import String
from geometry_msgs.msg import PoseArray

from simulation_interfaces.srv import GetEntitiesStates


def short_name(entity_path: str) -> str:
    # "/World/obj_3" -> "obj_3"
    # "/World/bin/wall_px" -> "wall_px"
    if not entity_path:
        return "unknown"
    return entity_path.split("/")[-1]


class PerceptionBridge(Node):
    def __init__(self):
        super().__init__("perception_bridge")

        # ===== Params =====
        self.declare_parameter("entities_srv", "/get_entities_states")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("world_frame", "world")
        
        # [중요] Regex 수정: 물체(obj_*)와 상자의 벽(wall_*)을 모두 찾음
        # wall_px, wall_nx, wall_py, wall_ny 등을 찾기 위함
        self.declare_parameter("filter_regex", r"^/World/obj_[0-9]+$|^/World/bin/wall_[pn][xy]$")

        # 토픽 설정
        self.declare_parameter("pose_topic", "/perception_bridge/poses")     # 물체 위치
        self.declare_parameter("names_topic", "/perception_bridge/names")    # 물체 이름
        self.declare_parameter("bin_topic", "/perception_bridge/bin_info")   # [NEW] 상자 정보(JSON)

        # [NEW] 상자 벽 두께 (미터 단위, 시뮬레이션 설정에 맞춰 수정 필요)
        # 예: 벽 두께가 2cm라면 0.02
        self.declare_parameter("wall_thickness", 0.02) 

        self.entities_srv_name = self.get_parameter("entities_srv").value
        self.rate_hz      = float(self.get_parameter("rate_hz").value)
        self.world_frame  = self.get_parameter("world_frame").value
        self.filter_regex = self.get_parameter("filter_regex").value
        self.pose_topic   = self.get_parameter("pose_topic").value
        self.names_topic  = self.get_parameter("names_topic").value
        self.bin_topic    = self.get_parameter("bin_topic").value
        self.wall_thick   = self.get_parameter("wall_thickness").value

        qos = QoSProfile(depth=10)

        # ===== Publishers =====
        self.pub_poses = self.create_publisher(PoseArray, self.pose_topic, qos)
        self.pub_names = self.create_publisher(String, self.names_topic, qos)
        self.pub_bin   = self.create_publisher(String, self.bin_topic, qos) # JSON 발행용

        # ===== Service Client =====
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
        states   = getattr(resp, "states", [])
        n = min(len(entities), len(states))
        
        if n == 0:
            return

        now = self.get_clock().now().to_msg()

        # 데이터 컨테이너
        pa = PoseArray()
        pa.header.stamp = now
        pa.header.frame_id = self.world_frame
        names_list: List[str] = []

        # 상자 벽 좌표를 임시 저장할 딕셔너리
        # 예: {'wall_px': 1.2, 'wall_nx': 0.8, ...}
        walls_pos: Dict[str, float] = {}

        for i in range(n):
            e_path = entities[i]
            st = states[i]
            if not e_path: continue

            name = short_name(e_path)
            
            # 1. 상자 벽(wall_*)인 경우
            if name.startswith("wall_"):
                # 좌표 수집 (X축 벽은 x좌표, Y축 벽은 y좌표만 중요)
                if "px" in name or "nx" in name:
                    walls_pos[name] = st.pose.position.x
                elif "py" in name or "ny" in name:
                    walls_pos[name] = st.pose.position.y
            
            # 2. 일반 물체(obj_*)인 경우
            elif name.startswith("obj_"):
                names_list.append(name)
                pa.poses.append(st.pose)

        # === 상자 크기 및 중심 계산 로직 ===
        bin_info = {}
        
        # X축 계산 (Left/Right)
        if "wall_px" in walls_pos and "wall_nx" in walls_pos:
            px = walls_pos["wall_px"]
            nx = walls_pos["wall_nx"]
            
            center_x = (px + nx) / 2.0
            # 벽 중심 간 거리 = Inner + Thickness
            dist_x = abs(px - nx)
            inner_w = dist_x - self.wall_thick
            
            bin_info["center_x"] = center_x
            bin_info["inner_width"] = inner_w +0.005

        # Y축 계산 (Top/Bottom)
        if "wall_py" in walls_pos and "wall_ny" in walls_pos:
            py = walls_pos["wall_py"]
            ny = walls_pos["wall_ny"]
            
            center_y = (py + ny) / 2.0
            dist_y = abs(py - ny)
            inner_h = dist_y - self.wall_thick
            
            bin_info["center_y"] = center_y
            bin_info["inner_height"] = inner_h + 0.005

        # 퍼블리시 1: 일반 물체 PoseArray
        self.pub_poses.publish(pa)

        # 퍼블리시 2: 일반 물체 이름 JSON
        msg_names = String()
        msg_names.data = json.dumps(names_list, ensure_ascii=False)
        self.pub_names.publish(msg_names)

        # 퍼블리시 3: 상자 정보 JSON (데이터가 충분할 때만)
        if "center_x" in bin_info and "center_y" in bin_info:
            # 최종 JSON 구조 생성
            # 예: {"center": [0.5, 0.5], "size": [0.4, 0.6]}
            final_bin_data = {
                "center": [bin_info["center_x"], bin_info["center_y"]],
                "size":   [bin_info["inner_width"], bin_info["inner_height"]]
            }
            msg_bin = String()
            msg_bin.data = json.dumps(final_bin_data)
            self.pub_bin.publish(msg_bin)

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