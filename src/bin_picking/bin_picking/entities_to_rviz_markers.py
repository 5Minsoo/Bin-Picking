#!/usr/bin/env python3
import zlib
import math
from typing import Dict, Tuple, Set, Optional, List

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, PointStamped, Point

from simulation_interfaces.srv import GetEntitiesStates


def stable_marker_id(name: str) -> int:
    return (zlib.adler32(name.encode("utf-8")) & 0x7FFFFFFF)


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x
 

def quat_to_rpy(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    # ROS quaternion (x,y,z,w) -> roll, pitch, yaw (rad)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = clamp(sinp, -1.0, 1.0)
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def quat_rotate_vec(x: float, y: float, z: float, w: float, vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
    # Rotate vector v by quaternion q (x,y,z,w) (ROS standard)
    # t = 2 * cross(q_vec, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # v' = v + w*t + cross(q_vec, t)
    vpx = vx + w * tx + (y * tz - z * ty)
    vpy = vy + w * ty + (z * tx - x * tz)
    vpz = vz + w * tz + (x * ty - y * tx)
    return vpx, vpy, vpz


class IsaacEntitiesMarkers(Node):
    def __init__(self):
        super().__init__("isaac_entities_markers")

        # ---- 기본값: ros2 run 때 인자 없이 동작하도록 여기서 default 세팅 ----
        self.declare_parameter("service_name", "/get_entities_states")
        self.declare_parameter("marker_topic", "/isaac_entities/markers")

        # RViz Publish Point 툴 토픽
        self.declare_parameter("clicked_point_topic", "/clicked_point")

        # obj_35, obj_36 ... 만 잡고 싶으면 정규식 사용 (비우면 전체 엔티티)
        self.declare_parameter("name_regex", r"^/World/obj_[0-9]+$")

        # Isaac에서 오는 pose가 월드좌표라고 가정하고 RViz fixed frame에 맞출 프레임
        self.declare_parameter("world_frame", "world")

        # STL 경로 (RViz가 실행되는 머신 기준 경로여야 함)
        self.declare_parameter("mesh_resource", "file:///root/bin_picking/utils/object.stl")

        # 스케일 조정용 파라미터
        self.declare_parameter("mesh_scale", [0.001, 0.001, 0.001])

        self.declare_parameter("use_embedded_materials", True)
        self.declare_parameter("color_rgba", [0.8, 0.8, 0.8, 1.0])

        self.declare_parameter("poll_hz", 10.0)
        self.declare_parameter("stale_sec", 0.5)  # 응답에서 사라지거나 오래 안 보이면 DELETE

        # ---- 클릭 선택 기능 관련 ----
        self.declare_parameter("select_radius", 0.08)  # (m) 클릭 지점과 엔티티 중심 거리 임계값
        self.declare_parameter("axis_length", 0.06)    # (m) 축 화살표 길이
        # ARROW(points 기반): scale.x=shaft_diameter, scale.y=head_diameter, scale.z=head_length
        self.declare_parameter("axis_arrow_scale", [0.004, 0.008, 0.012])
        self.declare_parameter("use_tf_for_clicked_point", True)  # 클릭 프레임 != world_frame이면 TF 변환 시도

        # ---- (추가) 클릭한 RViz 좌표 표시 기능 ----
        self.declare_parameter("show_clicked_point", True)
        self.declare_parameter("clicked_point_ns", "isaac_clicked_point")
        self.declare_parameter("clicked_point_color_rgba", [1.0, 1.0, 0.2, 1.0])  # 점 색상 (RGBA)
        self.declare_parameter("clicked_point_sphere_diameter", 0.015)            # (m) 점 크기
        self.declare_parameter("clicked_point_text_size", 0.03)                   # (m) TEXT scale.z
        self.declare_parameter("clicked_point_text_z_offset", 0.03)               # (m) 텍스트 z 오프셋
        self.declare_parameter("clicked_point_precision", 4)                      # 소수점 자리수
        self.declare_parameter("clicked_point_lifetime_sec", 0.0)                 # 0=영구(다음 클릭으로 갱신)

        self.service_name = self.get_parameter("service_name").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.clicked_point_topic = self.get_parameter("clicked_point_topic").value

        self.name_regex = self.get_parameter("name_regex").value
        self.world_frame = self.get_parameter("world_frame").value
        self.mesh_resource = self.get_parameter("mesh_resource").value
        self.mesh_scale = list(self.get_parameter("mesh_scale").value)
        self.use_embedded_materials = bool(self.get_parameter("use_embedded_materials").value)
        self.color_rgba = list(self.get_parameter("color_rgba").value)
        self.poll_hz = float(self.get_parameter("poll_hz").value)
        self.stale_sec = float(self.get_parameter("stale_sec").value)

        self.select_radius = float(self.get_parameter("select_radius").value)
        self.axis_length = float(self.get_parameter("axis_length").value)
        self.axis_arrow_scale = list(self.get_parameter("axis_arrow_scale").value)
        self.use_tf_for_clicked_point = bool(self.get_parameter("use_tf_for_clicked_point").value)

        # clicked point display
        self.show_clicked_point = bool(self.get_parameter("show_clicked_point").value)
        self.clicked_point_ns = str(self.get_parameter("clicked_point_ns").value)
        self.clicked_point_color_rgba = list(self.get_parameter("clicked_point_color_rgba").value)
        self.clicked_point_sphere_diameter = float(self.get_parameter("clicked_point_sphere_diameter").value)
        self.clicked_point_text_size = float(self.get_parameter("clicked_point_text_size").value)
        self.clicked_point_text_z_offset = float(self.get_parameter("clicked_point_text_z_offset").value)
        self.clicked_point_precision = int(self.get_parameter("clicked_point_precision").value)
        self.clicked_point_lifetime_sec = float(self.get_parameter("clicked_point_lifetime_sec").value)

        self.pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.cli = self.create_client(GetEntitiesStates, self.service_name)

        self.sub_clicked = self.create_subscription(
            PointStamped, self.clicked_point_topic, self._on_clicked_point, 10
        )

        self._cache: Dict[str, Tuple[Pose, float]] = {}  # name -> (pose, last_seen_sec)
        self._last_entities: Set[str] = set()
        self._inflight = False

        # 선택 상태
        self._selected: Optional[str] = None
        self._axis_visible: bool = False

        # 마지막 클릭 좌표(월드 기준)
        self._last_clicked_point_world: Optional[Point] = None

        # TF (선택적으로)
        self._tf_ok = False
        self._tf_buffer = None
        self._tf_listener = None
        self._do_transform_point = None
        if self.use_tf_for_clicked_point:
            try:
                import tf2_ros
                from tf2_geometry_msgs import do_transform_point  # type: ignore
                self._do_transform_point = do_transform_point
                self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
                self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
                self._tf_ok = True
            except Exception as e:
                self.get_logger().warn(
                    f"TF disabled (import failed). clicked_point는 world_frame과 동일하다고 가정합니다. err={e}"
                )
                self._tf_ok = False

        period = 1.0 / max(self.poll_hz, 1e-6)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(f"[OK] service={self.service_name}, markers={self.marker_topic}")
        self.get_logger().info(f"[OK] mesh={self.mesh_resource}, scale={self.mesh_scale}, regex={self.name_regex}")
        self.get_logger().info(f"[OK] world_frame={self.world_frame}")
        self.get_logger().info(f"[OK] clicked_point_topic={self.clicked_point_topic}, select_radius={self.select_radius}m")
        self.get_logger().info(f"[OK] show_clicked_point={self.show_clicked_point}, clicked_point_ns={self.clicked_point_ns}")

    # --------------------------
    # RViz 클릭 처리
    # --------------------------
    def _on_clicked_point(self, msg: PointStamped):
        # 1) 클릭 포인트를 world_frame으로 맞추기(가능하면 TF)
        pt_msg = msg
        if msg.header.frame_id and msg.header.frame_id != self.world_frame:
            if self._tf_ok and self._tf_buffer is not None and self._do_transform_point is not None:
                try:
                    trans = self._tf_buffer.lookup_transform(
                        self.world_frame,
                        msg.header.frame_id,
                        Time()
                    )
                    pt_msg = self._do_transform_point(msg, trans)
                except Exception as e:
                    self.get_logger().warn(
                        f"clicked_point TF 변환 실패: {msg.header.frame_id} -> {self.world_frame}, err={e}"
                    )
                    return
            else:
                self.get_logger().warn(
                    f"clicked_point frame({msg.header.frame_id}) != world_frame({self.world_frame})인데 TF가 꺼져있습니다."
                )
                return

        now_msg = self.get_clock().now().to_msg()
        marr = MarkerArray()

        # (추가) 클릭한 점/좌표를 RViz에 표시 (월드 기준 숫자)
        if self.show_clicked_point:
            self._last_clicked_point_world = pt_msg.point
            marr.markers.extend(self._make_clicked_point_markers(pt_msg.point, now_msg, action=Marker.ADD))

        # 2) 가장 가까운 엔티티 찾기
        target, dist = self._find_nearest_entity(pt_msg.point)
        if target is None:
            if marr.markers:
                self.pub.publish(marr)
            return

        if dist > self.select_radius:
            self.get_logger().info(
                f"[CLICK] nearest={target} dist={dist:.3f}m > select_radius={self.select_radius:.3f}m (무시)"
            )
            if marr.markers:
                self.pub.publish(marr)
            return

        # 3) 토글 로직
        if self._selected == target:
            self._axis_visible = not self._axis_visible
        else:
            self._selected = target
            self._axis_visible = True

        pose, _ = self._cache[target]
        self._log_pose(target, pose, dist)

        # 4) 축 마커 반영(ADD 또는 DELETE) + 클릭 마커와 함께 한번에 publish
        if self._axis_visible:
            marr.markers.extend(self._make_axis_markers(target, pose, now_msg, action=Marker.ADD))
        else:
            marr.markers.extend(self._make_axis_markers(target, pose, now_msg, action=Marker.DELETE))

        if marr.markers:
            self.pub.publish(marr)

    def _find_nearest_entity(self, p: Point) -> Tuple[Optional[str], float]:
        if not self._cache:
            return None, float("inf")

        best_name = None
        best_d2 = float("inf")
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        for name, (pose, last_seen) in self._cache.items():
            # stale는 선택 후보에서 제외(원하시면 제거 가능)
            if (now_sec - last_seen) > self.stale_sec:
                continue

            dx = pose.position.x - p.x
            dy = pose.position.y - p.y
            dz = pose.position.z - p.z
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < best_d2:
                best_d2 = d2
                best_name = name

        if best_name is None:
            return None, float("inf")
        return best_name, math.sqrt(best_d2)

    def _log_pose(self, name: str, pose: Pose, dist: float):
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w
        r, p, y = quat_to_rpy(qx, qy, qz, qw)
        self.get_logger().info(
            f"[SELECT] {name} (dist={dist:.3f}m)\n"
            f"  pos: x={pose.position.x:.4f}, y={pose.position.y:.4f}, z={pose.position.z:.4f}\n"
            f"  quat: x={qx:.5f}, y={qy:.5f}, z={qz:.5f}, w={qw:.5f}\n"
            f"  rpy(rad): r={r:.4f}, p={p:.4f}, y={y:.4f}\n"
            f"  rpy(deg): r={math.degrees(r):.1f}, p={math.degrees(p):.1f}, y={math.degrees(y):.1f}\n"
            f"  axis_visible={self._axis_visible}"
        )

    # --------------------------
    # 주기적으로 서비스 폴링 + 마커 퍼블리시
    # --------------------------
    def _tick(self):
        if not self.cli.service_is_ready():
            return
        if self._inflight:
            return

        req = GetEntitiesStates.Request()
        req.filters.filter = self.name_regex  # 비우면 전체 엔티티

        self._inflight = True
        fut = self.cli.call_async(req)
        fut.add_done_callback(self._on_response)

    def _on_response(self, future):
        self._inflight = False
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        now_msg = self.get_clock().now().to_msg()

        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f"GetEntitiesStates call failed: {e}")
            return

        # result 코드가 구현마다 0/1로 다를 수 있어, 실패만 확실히 거르는 방식
        if hasattr(resp, "result"):
            code = getattr(resp.result, "result", 0)
            err = getattr(resp.result, "error_message", "")
            if err and code not in (0, 1):
                self.get_logger().warn(f"GetEntitiesStates failed: code={code}, msg={err}")
                return

        entities = list(getattr(resp, "entities", []))
        states = list(getattr(resp, "states", []))
        n = min(len(entities), len(states))

        current: Set[str] = set()
        for i in range(n):
            name = entities[i]
            st = states[i]
            current.add(name)
            self._cache[name] = (st.pose, now_sec)

        removed = self._last_entities - current
        self._last_entities = current

        stale = set()
        for name, (_, last_seen) in list(self._cache.items()):
            if (now_sec - last_seen) > self.stale_sec:
                stale.add(name)

        # 선택된 엔티티가 사라졌으면 축도 같이 내리기
        if self._selected is not None and (self._selected in removed or self._selected in stale):
            if self._axis_visible:
                marr_del = MarkerArray()
                pose, _ = self._cache.get(self._selected, (Pose(), now_sec))
                marr_del.markers.extend(self._make_axis_markers(self._selected, pose, now_msg, action=Marker.DELETE))
                if marr_del.markers:
                    self.pub.publish(marr_del)
            self._selected = None
            self._axis_visible = False

        marr = MarkerArray()

        # ADD/UPDATE (mesh)
        for name in current:
            pose, _ = self._cache[name]

            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = now_msg

            m.ns = "isaac_entities"
            m.id = stable_marker_id(name)

            m.type = Marker.MESH_RESOURCE
            m.action = Marker.ADD

            m.pose = pose
            m.scale.x = float(self.mesh_scale[0])
            m.scale.y = float(self.mesh_scale[1])
            m.scale.z = float(self.mesh_scale[2])

            m.mesh_resource = self.mesh_resource
            m.mesh_use_embedded_materials = self.use_embedded_materials

            # embedded material이 안 먹는 상황 대비
            m.color.r = float(self.color_rgba[0])
            m.color.g = float(self.color_rgba[1])
            m.color.b = float(self.color_rgba[2])
            m.color.a = float(self.color_rgba[3])

            marr.markers.append(m)

        # 선택 축(업데이트 포함): 선택되어 있고 보이는 상태면 매 tick마다 따라가게 ADD
        if self._selected is not None and self._axis_visible and self._selected in self._cache:
            pose, last_seen = self._cache[self._selected]
            if (now_sec - last_seen) <= self.stale_sec:
                marr.markers.extend(self._make_axis_markers(self._selected, pose, now_msg, action=Marker.ADD))

        # DELETE(removed + stale) (mesh)
        for name in (removed | stale):
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = now_msg
            m.ns = "isaac_entities"
            m.id = stable_marker_id(name)
            m.action = Marker.DELETE
            marr.markers.append(m)
            self._cache.pop(name, None)

        if marr.markers:
            self.pub.publish(marr)

    # --------------------------
    # 클릭 포인트(점 + 텍스트) 마커 생성/삭제
    # --------------------------
    def _make_clicked_point_markers(self, p: Point, stamp_msg, action: int) -> List[Marker]:
        r, g, b, a = [float(x) for x in self.clicked_point_color_rgba]

        lifetime_msg = None
        if self.clicked_point_lifetime_sec > 0.0:
            lifetime_msg = Duration(seconds=self.clicked_point_lifetime_sec).to_msg()

        # Sphere marker
        ms = Marker()
        ms.header.frame_id = self.world_frame
        ms.header.stamp = stamp_msg
        ms.ns = self.clicked_point_ns
        ms.id = stable_marker_id("__clicked_point_sphere__")
        ms.type = Marker.SPHERE
        ms.action = action
        ms.pose.position.x = float(p.x)
        ms.pose.position.y = float(p.y)
        ms.pose.position.z = float(p.z)
        ms.pose.orientation.w = 1.0
        d = float(self.clicked_point_sphere_diameter)
        ms.scale.x = d
        ms.scale.y = d
        ms.scale.z = d
        ms.color.r = r
        ms.color.g = g
        ms.color.b = b
        ms.color.a = a
        if lifetime_msg is not None:
            ms.lifetime = lifetime_msg

        # Text marker
        mt = Marker()
        mt.header.frame_id = self.world_frame
        mt.header.stamp = stamp_msg
        mt.ns = self.clicked_point_ns
        mt.id = stable_marker_id("__clicked_point_text__")
        mt.type = Marker.TEXT_VIEW_FACING
        mt.action = action
        mt.pose.position.x = float(p.x)
        mt.pose.position.y = float(p.y)
        mt.pose.position.z = float(p.z + self.clicked_point_text_z_offset)
        mt.pose.orientation.w = 1.0
        mt.scale.z = float(self.clicked_point_text_size)

        prec = max(0, int(self.clicked_point_precision))
        mt.text = f"x={p.x:.{prec}f}, y={p.y:.{prec}f}, z={p.z:.{prec}f} ({self.world_frame})"

        # 텍스트는 가독성 우선으로 흰색
        mt.color.r = 1.0
        mt.color.g = 1.0
        mt.color.b = 1.0
        mt.color.a = 1.0
        if lifetime_msg is not None:
            mt.lifetime = lifetime_msg

        return [ms, mt]

    # --------------------------
    # 축 마커 생성/삭제
    # --------------------------
    def _make_axis_markers(self, name: str, pose: Pose, stamp_msg, action: int) -> List[Marker]:
        # 엔티티 pose 기준으로 x/y/z 방향 벡터(쿼터니언 회전 적용)
        px = pose.position.x
        py = pose.position.y
        pz = pose.position.z

        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w

        # rotated unit axes
        ux = quat_rotate_vec(qx, qy, qz, qw, 1.0, 0.0, 0.0)
        uy = quat_rotate_vec(qx, qy, qz, qw, 0.0, 1.0, 0.0)
        uz = quat_rotate_vec(qx, qy, qz, qw, 0.0, 0.0, 1.0)

        L = self.axis_length
        sx = float(self.axis_arrow_scale[0])
        sy = float(self.axis_arrow_scale[1])
        sz = float(self.axis_arrow_scale[2])

        def make_arrow(axis_key: str, vx: float, vy: float, vz: float, r: float, g: float, b: float) -> Marker:
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = stamp_msg

            m.ns = "isaac_entities_axis"
            m.id = stable_marker_id(f"{name}::{axis_key}")

            m.type = Marker.ARROW
            m.action = action

            # points 기반 ARROW: start -> end
            start = Point(x=px, y=py, z=pz)
            end = Point(x=px + L * vx, y=py + L * vy, z=pz + L * vz)
            m.points = [start, end]

            # ARROW(points): scale.x=shaft_d, scale.y=head_d, scale.z=head_len
            m.scale.x = sx
            m.scale.y = sy
            m.scale.z = sz

            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 1.0

            return m

        mx = make_arrow("axis_x", ux[0], ux[1], ux[2], 1.0, 0.0, 0.0)
        my = make_arrow("axis_y", uy[0], uy[1], uy[2], 0.0, 1.0, 0.0)
        mz = make_arrow("axis_z", uz[0], uz[1], uz[2], 0.0, 0.0, 1.0)

        return [mx, my, mz]


def main():
    rclpy.init()
    node = IsaacEntitiesMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
