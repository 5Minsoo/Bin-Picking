import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Point, Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive

# 서비스 인터페이스
from simulation_interfaces.srv import GetEntitiesStates 

import json
import trimesh
import trimesh.transformations as tf
import numpy as np
import os

class ObjectSpawner(Node):

    def __init__(self):
        super().__init__('object_spawner')

        # =========================================================
        # 1. 기본 설정
        # =========================================================
        self.stl_path = '/root/bin_picking/src/utils/object.stl'
        self.mesh_scale = 0.0008
        self.target_frame = 'world' 
        
        # 좌표계 보정 (필요시 수정)
        self.correction_matrix = tf.euler_matrix(0, 0, 0) 
        
        # =========================================================
        # 2. Bin (상자) 설정
        # =========================================================
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

        # =========================================================
        # 3. 통신 설정
        # =========================================================
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)

        self.sub_names = self.create_subscription(String, '/perception_bridge/names', self.names_callback, 10)
        self.sub_poses = self.create_subscription(PoseArray, '/perception_bridge/poses', self.poses_callback, 10)
        
        self.cli_bin_state = self.create_client(GetEntitiesStates, self.bin_service_name)
        
        # 1.0초마다 Bin 정보 요청 (생성될 때까지)
        self.bin_timer = self.create_timer(1.0, self.request_bin_info)

        # =========================================================
        # 4. 상태 변수
        # =========================================================
        self.object_names = []      
        self.mesh_msg = None        
        self.known_object_ids = set() 

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

            self.mesh_msg = Mesh()
            for v in mesh_data.vertices:
                p = Point()
                p.x, p.y, p.z = v[0], v[1], v[2]
                self.mesh_msg.vertices.append(p)
            for f in mesh_data.faces:
                t = MeshTriangle()
                t.vertex_indices = [int(f[0]), int(f[1]), int(f[2])]
                self.mesh_msg.triangles.append(t)
            self.get_logger().info("Mesh loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load mesh: {e}")

    # =========================================================
    # [유지] 일반 바닥(General Floor) 생성 함수
    # =========================================================
    def spawn_floor(self):
        floor_obj = CollisionObject()
        floor_obj.header.frame_id = self.target_frame
        floor_obj.header.stamp = self.get_clock().now().to_msg()
        floor_obj.id = "floor"
        floor_obj.operation = CollisionObject.ADD
        
        # 3m x 3m 크기, 두께 1cm 박스 생성
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [3.0, 3.0, 0.01] 
        
        pose = Pose()
        # 바닥 상단면이 z=0에 오도록, 중심을 -0.005 (두께의 절반)로 설정
        pose.position.z = -0.005 
        pose.orientation.w = 1.0
        
        floor_obj.primitives = [prim]
        floor_obj.primitive_poses = [pose]
        
        self.collision_pub.publish(floor_obj)
        self.get_logger().info("Floor spawned (3m x 3m).")

    # =========================================================
    # [Section A] Bin (환경) 생성 로직
    # =========================================================
    def request_bin_info(self):
        if self.bin_spawned:
            return

        if not self.cli_bin_state.service_is_ready():
            self.get_logger().warn(f"Waiting for service '{self.bin_service_name}'...", throttle_duration_sec=5.0)
            return
        
        req = GetEntitiesStates.Request()
        req.filters.filter = "|".join(self.bin_parts)
        future = self.cli_bin_state.call_async(req)
        future.add_done_callback(self.bin_response_callback)

    def bin_response_callback(self, future):
        try:
            resp = future.result()
            
            if not resp or resp.result.result != 1:
                return

            if len(resp.states) < 5:
                self.get_logger().info(f"Waiting for bin parts... Found {len(resp.states)}/5")
                return

            poses = {name: state.pose for name, state in zip(resp.entities, resp.states)}
            
            missing = [k for k in self.bin_parts if k not in poses]
            if missing:
                return

            p_floor = poses['/World/bin/floor']
            p_px = poses['/World/bin/wall_px']
            p_nx = poses['/World/bin/wall_nx']
            p_py = poses['/World/bin/wall_py']
            p_ny = poses['/World/bin/wall_ny']

            dist_x = abs(p_px.position.x - p_nx.position.x)
            dist_y = abs(p_py.position.y - p_ny.position.y)
            
            if dist_x < 0.001 or dist_y < 0.001:
                self.get_logger().info("Bin dimensions too small. Retrying...")
                return

            full_x = dist_x + self.bin_thick
            full_y = dist_y + self.bin_thick
            height_wall = 2 * (p_px.position.z - p_floor.position.z) - self.bin_thick
            if height_wall < 0: height_wall = 0.1

            self.get_logger().info(f" >>> Spawning Bin Walls Only: {full_x:.3f} x {full_y:.3f} x {height_wall:.3f}")

            bin_obj = CollisionObject()
            bin_obj.header.frame_id = self.target_frame 
            bin_obj.header.stamp = self.get_clock().now().to_msg()
            bin_obj.id = "World/bin"
            bin_obj.operation = CollisionObject.ADD 

            # [수정됨] 상자 바닥(prim_floor)은 생성하지 않음.
            # prim_floor = SolidPrimitive() ... (삭제됨)

            def make_wall(dims):
                p = SolidPrimitive()
                p.type = SolidPrimitive.BOX
                p.dimensions = dims
                return p

            # [수정됨] primitives 리스트에 바닥면 제외, 벽 4개만 추가
            bin_obj.primitives = [
                make_wall([self.bin_thick, full_y, height_wall]), # px
                make_wall([self.bin_thick, full_y, height_wall]), # nx
                make_wall([full_x, self.bin_thick, height_wall]), # py
                make_wall([full_x, self.bin_thick, height_wall])  # ny
            ]
            
            # [수정됨] primitive_poses 리스트에서도 바닥(p_floor) 제외
            bin_obj.primitive_poses = [p_px, p_nx, p_py, p_ny]

            self.collision_pub.publish(bin_obj)
            
            # [유지] 일반 바닥(General Floor)은 생성
            # self.spawn_floor()
            
            self.bin_spawned = True
            self.bin_timer.cancel()
            self.get_logger().info("Bin Walls and General Floor created successfully.")

        except Exception as e:
            self.get_logger().error(f"Error inside bin_response_callback: {e}")

    # =========================================================
    # [Section B] Object (물체) 동기화 로직
    # =========================================================
    def names_callback(self, msg):
        try:
            self.object_names = json.loads(msg.data)
        except json.JSONDecodeError:
            self.object_names = []

    def poses_callback(self, msg):
        if self.mesh_msg is None:
            return

        if len(msg.poses) != len(self.object_names):
            return

        current_frame_id = msg.header.frame_id or self.target_frame

        current_ids = set(self.object_names)
        ids_to_remove = self.known_object_ids - current_ids

        if ids_to_remove:
            for obj_id in ids_to_remove:
                remove_obj = CollisionObject()
                remove_obj.header.frame_id = current_frame_id
                remove_obj.header.stamp = self.get_clock().now().to_msg()
                remove_obj.id = obj_id
                remove_obj.operation = CollisionObject.REMOVE
                remove_obj.meshes = []      
                remove_obj.primitives = []  
                self.collision_pub.publish(remove_obj)

        for i, (name, pose) in enumerate(zip(self.object_names, msg.poses)):
            current_quat = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
            mat_pose = tf.quaternion_matrix(current_quat)
            mat_pose[0:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
            
            new_mat = np.dot(mat_pose, self.correction_matrix)
            new_quat = tf.quaternion_from_matrix(new_mat) 

            pose.position.x = new_mat[0, 3]
            pose.position.y = new_mat[1, 3]
            pose.position.z = new_mat[2, 3]
            pose.orientation.w = new_quat[0]
            pose.orientation.x = new_quat[1]
            pose.orientation.y = new_quat[2]
            pose.orientation.z = new_quat[3]

            col_obj = CollisionObject()
            col_obj.header.frame_id = current_frame_id
            col_obj.header.stamp = self.get_clock().now().to_msg()
            col_obj.id = name
            col_obj.meshes = [self.mesh_msg]
            col_obj.mesh_poses = [pose]
            col_obj.operation = CollisionObject.ADD 
            
            self.collision_pub.publish(col_obj)

        self.known_object_ids = current_ids

def main(args=None):
    rclpy.init(args=args)
    node = ObjectSpawner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()