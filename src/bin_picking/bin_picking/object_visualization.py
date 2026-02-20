import rclpy
from rclpy.node import Node
from std_msgs.msg import String, ColorRGBA
from geometry_msgs.msg import PoseArray, Point, Pose, Vector3
from visualization_msgs.msg import Marker, MarkerArray 
from simulation_interfaces.srv import GetEntitiesStates 
import json
import trimesh
import trimesh.transformations as tf
import numpy as np
import os
import time

class VisualObjectSpawner(Node):

    def __init__(self):
        super().__init__('visual_object_spawner')

        # 1. 기본 설정
        self.stl_path = '/root/bin_picking/src/utils/object.stl'
        self.mesh_scale = 0.001
        self.target_frame = 'world' 
        self.correction_matrix = tf.euler_matrix(0, 0, 0) 
        
        # 2. Bin 설정
        self.bin_service_name = '/get_entities_states'
        self.bin_thick = 0.015
        self.bin_parts = [
            '/World/bin/floor', 
            '/World/bin/wall_px', 
            '/World/bin/wall_nx', 
            '/World/bin/wall_py', 
            '/World/bin/wall_ny'
        ]
        self.bin_spawned = False

        # 3. 통신 설정
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)

        self.sub_names = self.create_subscription(String, '/perception_bridge/names', self.names_callback, 10)
        self.sub_poses = self.create_subscription(PoseArray, '/perception_bridge/poses', self.poses_callback, 10)
        self.cli_bin_state = self.create_client(GetEntitiesStates, self.bin_service_name)
        
        # 타이머 설정
        self.clean_timer = self.create_timer(0.5, self.force_clear_markers)
        self.bin_timer = self.create_timer(1.0, self.request_bin_info)
        
        # [추가] 마커 발행용 타이머 (10Hz = 0.1초) -> 50Hz 경고 해결
        self.vis_timer = self.create_timer(0.1, self.publish_markers_loop)

        # 4. 상태 변수
        self.object_names = []      
        self.marker_points = [] 
        self.known_object_ids = set() 
        
        # [추가] 최신 데이터를 담아둘 변수
        self.latest_poses_msg = None 
        
        self.load_mesh()

    def load_mesh(self):
        if not os.path.exists(self.stl_path): 
            self.get_logger().error(f"STL file not found: {self.stl_path}")
            return
        try:
            mesh_data = trimesh.load(self.stl_path)
            matrix = np.eye(4)
            matrix[:3, :3] *= self.mesh_scale
            mesh_data.apply_transform(matrix)
            
            self.marker_points = []
            for face in mesh_data.faces:
                for idx in face:
                    v = mesh_data.vertices[idx]
                    p = Point()
                    p.x, p.y, p.z = v[0], v[1], v[2]
                    self.marker_points.append(p)
            
            self.get_logger().info(f"Visual Mesh loaded. Points: {len(self.marker_points)}")

        except Exception as e: 
            self.get_logger().error(f"Failed to load mesh: {e}")

    def force_clear_markers(self):
        ma = MarkerArray()
        marker_del = Marker()
        marker_del.action = Marker.DELETEALL 
        ma.markers.append(marker_del)
        self.marker_pub.publish(ma)
        self.clean_timer.cancel()

    def request_bin_info(self):
        if self.bin_spawned: return
        if not self.cli_bin_state.service_is_ready(): return
        
        req = GetEntitiesStates.Request()
        req.filters.filter = "|".join(self.bin_parts)
        future = self.cli_bin_state.call_async(req)
        future.add_done_callback(self.bin_response_callback)

    def bin_response_callback(self, future):
        try:
            resp = future.result()
            if not resp or resp.result.result != 1 or len(resp.states) < 5: return

            poses = {name: state.pose for name, state in zip(resp.entities, resp.states)}
            p_floor = poses['/World/bin/floor']
            p_px = poses['/World/bin/wall_px']
            p_nx = poses['/World/bin/wall_nx']
            p_py = poses['/World/bin/wall_py']
            p_ny = poses['/World/bin/wall_ny']

            dist_x = abs(p_px.position.x - p_nx.position.x)
            dist_y = abs(p_py.position.y - p_ny.position.y)
            if dist_x < 0.001: return

            full_x = dist_x + self.bin_thick
            full_y = dist_y + self.bin_thick
            height_wall = 2 * (p_px.position.z - p_floor.position.z) - self.bin_thick
            if height_wall < 0: height_wall = 0.1

            marker_array = MarkerArray()
            wall_configs = [
                (p_px, [self.bin_thick, full_y, height_wall], 100), 
                (p_nx, [self.bin_thick, full_y, height_wall], 101),
                (p_py, [full_x, self.bin_thick, height_wall], 102),
                (p_ny, [full_x, self.bin_thick, height_wall], 103)
            ]

            for pose, dim, m_id in wall_configs:
                marker = Marker()
                marker.header.frame_id = self.target_frame
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "bin_visual"
                marker.id = m_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose = pose
                marker.scale.x = dim[0]
                marker.scale.y = dim[1]
                marker.scale.z = dim[2]
                marker.color = ColorRGBA(r=0.6, g=0.4, b=0.2, a=0.5)
                marker.lifetime.sec = 0 
                marker_array.markers.append(marker)

            self.marker_pub.publish(marker_array)
            self.bin_spawned = True
            self.bin_timer.cancel()
            self.get_logger().info(">>> BIN VISUALIZED.")

        except Exception as e:
            self.get_logger().error(f"Error in bin callback: {e}")

    def names_callback(self, msg):
        try:
            self.object_names = json.loads(msg.data)
        except json.JSONDecodeError:
            self.object_names = []

    # [수정] 콜백에서는 데이터 저장만 수행 (부하 없음)
    def poses_callback(self, msg):
        self.latest_poses_msg = msg

    # [추가] 10Hz 주기로 실행되는 실제 발행 함수
    def publish_markers_loop(self):
        msg = self.latest_poses_msg
        if msg is None: return
        if not self.marker_points: return 
        if len(msg.poses) != len(self.object_names): return

        current_frame_id = msg.header.frame_id or self.target_frame
        current_ids = set(self.object_names)
        ids_to_remove = self.known_object_ids - current_ids

        # 1. 사라진 물체 삭제
        if ids_to_remove:
            ma_delete = MarkerArray()
            for obj_id in ids_to_remove:
                m_del = Marker()
                m_del.header.frame_id = current_frame_id
                m_del.ns = "detected_objects"
                m_del.id = abs(hash(obj_id)) % 2147483647
                m_del.action = Marker.DELETE
                ma_delete.markers.append(m_del)
            
            if ma_delete.markers:
                self.marker_pub.publish(ma_delete)

        # 2. 업데이트 (0.1초마다 수행)
        ma_update = MarkerArray()
        
        for i, (name, pose) in enumerate(zip(self.object_names, msg.poses)):
            # 좌표 보정
            current_quat = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
            mat_pose = tf.quaternion_matrix(current_quat)
            mat_pose[0:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
            new_mat = np.dot(mat_pose, self.correction_matrix)
            new_quat = tf.quaternion_from_matrix(new_mat) 
            
            pose.position.x, pose.position.y, pose.position.z = new_mat[0, 3], new_mat[1, 3], new_mat[2, 3]
            pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = new_quat

            marker = Marker()
            marker.header.frame_id = current_frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "detected_objects"
            marker.id = abs(hash(name)) % 2147483647 
            marker.type = Marker.TRIANGLE_LIST
            marker.action = Marker.ADD
            marker.pose = pose
            marker.scale = Vector3(x=1.0, y=1.0, z=1.0)
            marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.8)
            # 수명 0.12초 (타이머가 0.1초이므로 깜빡임 방지용으로 약간 여유 둠)
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 120000000 
            
            marker.points = self.marker_points
            ma_update.markers.append(marker)

        if ma_update.markers:
            self.marker_pub.publish(ma_update)

        self.known_object_ids = current_ids

def main(args=None):
    rclpy.init(args=args)
    node = VisualObjectSpawner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()