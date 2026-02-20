#!/usr/bin/env python3
import zlib
import math
import sys
from typing import Dict, Optional
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, PointStamped
from simulation_interfaces.srv import GetEntitiesStates

# [추가] 행렬 연산을 위한 라이브러리
import numpy as np
import trimesh.transformations as tf 

def stable_marker_id(name: str) -> int:
    return (zlib.adler32(name.encode("utf-8")) & 0x7FFFFFFF)

def quat_rotate_vec(q, v):
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx)
    )

def quat_to_rpy_deg(q):
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

class IsaacClickMarker(Node):
    def __init__(self):
        super().__init__("isaac_click_marker")

        self.declare_parameter("service_name", "/get_entities_states")
        self.declare_parameter("marker_topic", "/isaac_entities/markers")
        self.declare_parameter("clicked_point_topic", "/clicked_point")
        self.declare_parameter("name_regex", r"^/World/obj_[0-9]+$")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("mesh_resource", "file:///root/bin_picking/src/utils/object.stl")
        self.declare_parameter("mesh_scale", [0.001, 0.001, 0.001]) 
        self.declare_parameter("color_rgba", [0.8, 0.8, 0.8, 1.0])

        self.service_name = self.get_parameter("service_name").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.regex = self.get_parameter("name_regex").value
        self.frame_id = self.get_parameter("world_frame").value
        self.mesh_path = self.get_parameter("mesh_resource").value
        self.mesh_scale = self.get_parameter("mesh_scale").value
        self.rgba = self.get_parameter("color_rgba").value

        # [추가] 회전 보정 행렬 정의 (Z축 90도 회전 예시)
        # 상황에 맞춰 np.pi/2 (90도) 혹은 -np.pi/2 (-90도) 등으로 조절하세요.
        self.correction_matrix = tf.euler_matrix(0, np.pi/2, 0)

        self.pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.cli = self.create_client(GetEntitiesStates, self.service_name)
        self.sub_click = self.create_subscription(
            PointStamped, 
            self.get_parameter("clicked_point_topic").value, 
            self._on_click, 
            10
        )

        self._cache: Dict[str, Pose] = {}
        self._selected_name: Optional[str] = None
        self._inflight = False
        self._select_radius = 0.1

        self.create_timer(0.1, self._tick)

        print("-" * 60, flush=True)
        print(f"Node Ready. RViz에서 'Publish Point'로 물체를 클릭하세요.", flush=True)
        print("-" * 60, flush=True)

    def _on_click(self, msg: PointStamped):
        p_click = msg.point
        best_name = None
        min_dist = float('inf')

        for name, pose in self._cache.items():
            dx = pose.position.x - p_click.x
            dy = pose.position.y - p_click.y
            dz = pose.position.z - p_click.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < min_dist:
                min_dist = dist
                best_name = name

        if best_name and min_dist <= self._select_radius:
            self._selected_name = best_name
            self._print_clean_code(best_name, self._cache[best_name])
        else:
            print(f">> 허공 클릭됨 (가장 가까운 물체: {best_name}, 거리: {min_dist:.3f}m)", flush=True)
            self._selected_name = None
        
        self._publish_markers()

    def _print_clean_code(self, name, pose):
        """복붙하기 좋은 파이썬 코드 출력"""
        p = pose.position
        q = pose.orientation
        r, pt, y = quat_to_rpy_deg(q)

        print("\n" + "="*20 + " [ 복사하세요 ] " + "="*20, flush=True)
        print(f"# Selected Object: {name}", flush=True)
        print(f"# RPY(deg): ({r:.1f}, {pt:.1f}, {y:.1f})", flush=True)
        print("from geometry_msgs.msg import Pose", flush=True)
        print("", flush=True)
        print("target_pose = Pose()", flush=True)
        print(f"target_pose.position.x = {p.x:.5f}", flush=True)
        print(f"target_pose.position.y = {p.y:.5f}", flush=True)
        print(f"target_pose.position.z = {p.z:.5f}", flush=True)
        print(f"target_pose.orientation.x = {q.x:.5f}", flush=True)
        print(f"target_pose.orientation.y = {q.y:.5f}", flush=True)
        print(f"target_pose.orientation.z = {q.z:.5f}", flush=True)
        print(f"target_pose.orientation.w = {q.w:.5f}", flush=True)
        print("="*56 + "\n", flush=True)

    def _tick(self):
        if not self.cli.service_is_ready() or self._inflight: return
        req = GetEntitiesStates.Request()
        req.filters.filter = self.regex
        self._inflight = True
        future = self.cli.call_async(req)
        future.add_done_callback(self._on_response)

    def _on_response(self, future):
        self._inflight = False
        try:
            resp = future.result()
        except Exception: return
        
        self._cache.clear()
        entities = getattr(resp, "entities", [])
        states = getattr(resp, "states", [])
        
        for i in range(min(len(entities), len(states))):
            short_name = entities[i].split("/")[-1]
            raw_pose = states[i].pose

            # === [수정] 좌표 변환 적용 (Pose -> Matrix -> Rotated -> Pose) ===
            
            # 1. Pose를 Matrix로 (Quaternion 순서: w, x, y, z)
            q_list = [raw_pose.orientation.w, raw_pose.orientation.x, raw_pose.orientation.y, raw_pose.orientation.z]
            mat_pose = tf.quaternion_matrix(q_list)
            mat_pose[0:3, 3] = [raw_pose.position.x, raw_pose.position.y, raw_pose.position.z]

            # 2. 회전 행렬 곱셈 (Local Frame 기준 회전)
            new_mat = np.dot(mat_pose, self.correction_matrix)

            # 3. Matrix를 다시 Pose로 변환
            new_q = tf.quaternion_from_matrix(new_mat) # [w, x, y, z]

            corrected_pose = Pose()
            corrected_pose.position.x = new_mat[0, 3]
            corrected_pose.position.y = new_mat[1, 3]
            corrected_pose.position.z = new_mat[2, 3]
            corrected_pose.orientation.w = new_q[0]
            corrected_pose.orientation.x = new_q[1]
            corrected_pose.orientation.y = new_q[2]
            corrected_pose.orientation.z = new_q[3]

            # 4. 변환된 Pose를 저장 (마커와 텍스트 출력 모두 이 값을 사용함)
            self._cache[short_name] = corrected_pose
            
        self._publish_markers()

    def _publish_markers(self):
        marr = MarkerArray()
        now = self.get_clock().now().to_msg()
        for name, pose in self._cache.items():
            is_selected = (name == self._selected_name)
            # 1. Mesh
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = now
            m.ns = "obj_mesh"
            m.id = stable_marker_id(name)
            m.type = Marker.MESH_RESOURCE
            m.action = Marker.ADD
            m.pose = pose # 이미 회전된 Pose 사용
            m.scale.x, m.scale.y, m.scale.z = self.mesh_scale
            m.mesh_resource = self.mesh_path
            if is_selected:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 0.0, 1.0
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = self.rgba
            marr.markers.append(m)
            # 2. Selected UI
            if is_selected:
                t = Marker()
                t.header.frame_id = self.frame_id
                t.header.stamp = now
                t.ns = "obj_name"
                t.id = stable_marker_id(name + "_txt")
                t.type = Marker.TEXT_VIEW_FACING
                t.action = Marker.ADD
                t.text = name
                t.pose.position.x = pose.position.x
                t.pose.position.y = pose.position.y
                t.pose.position.z = pose.position.z + 0.08
                t.scale.z = 0.03
                t.color.r, t.color.g, t.color.b, t.color.a = 1.0, 1.0, 1.0, 1.0
                marr.markers.append(t)
                axis_len = 0.15
                vecs = {'x': (1,0,0,(1.0,0.0,0.0)), 'y': (0,1,0,(0.0,1.0,0.0)), 'z': (0,0,1,(0.0,0.0,1.0))}
                for axis_name, (vx, vy, vz, color) in vecs.items():
                    arrow = Marker()
                    arrow.header.frame_id = self.frame_id
                    arrow.header.stamp = now
                    arrow.ns = f"obj_axis_{axis_name}"
                    arrow.id = stable_marker_id(name + axis_name)
                    arrow.type = Marker.ARROW
                    arrow.action = Marker.ADD
                    p1 = Point(x=pose.position.x, y=pose.position.y, z=pose.position.z)
                    rvx, rvy, rvz = quat_rotate_vec(pose.orientation, (vx, vy, vz))
                    p2 = Point(x=p1.x + rvx*axis_len, y=p1.y + rvy*axis_len, z=p1.z + rvz*axis_len)
                    arrow.points = [p1, p2]
                    arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.005, 0.01, 0.0
                    arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = (*color, 1.0)
                    marr.markers.append(arrow)
        self.pub.publish(marr)

def main():
    rclpy.init()
    node = IsaacClickMarker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()