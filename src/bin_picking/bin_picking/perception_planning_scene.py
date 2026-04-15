import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Point, Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive

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
        self.bin_thick = 0.015

        # =========================================================
        # 3. 통신 설정
        # =========================================================
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)

        self.sub_names   = self.create_subscription(String,    '/perception_bridge/names',    self.names_callback,    10)
        self.sub_bin     = self.create_subscription(String,    '/perception_bridge/bin_info', self.bin_callback,      10)
        self.sub_poses   = self.create_subscription(PoseArray, '/perception_bridge/poses',    self.poses_callback,    10)

        # =========================================================
        # 4. 상태 변수
        # =========================================================
        self.object_names    = []
        self.mesh_msg        = None
        self.known_object_ids = set()

        self.load_mesh()

    # =========================================================
    # Mesh 로드
    # =========================================================
    def load_mesh(self):
        if not os.path.exists(self.stl_path):
            self.stl_path = '/home/minsoo/bin_picking/src/utils/object.stl'
        try:
            mesh_data = trimesh.load(self.stl_path)
            matrix = np.eye(4)
            matrix[:3, :3] *= self.mesh_scale
            mesh_data.apply_transform(matrix)

            self.mesh_msg = Mesh()
            for v in mesh_data.vertices:
                p = Point()
                p.x, p.y, p.z = float(v[0]), float(v[1]), float(v[2])
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

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [3.0, 3.0, 0.01]

        pose = Pose()
        pose.position.z = -0.005
        pose.orientation.w = 1.0

        floor_obj.primitives = [prim]
        floor_obj.primitive_poses = [pose]

        self.collision_pub.publish(floor_obj)
        self.get_logger().info("Floor spawned (3m x 3m).")

    # =========================================================
    # [Section A] Bin 토픽 콜백 (서비스 대체)
    #
    # 수신 JSON 형식:
    # {
    #   "center": [cx, cy],   # world 좌표 (m)
    #   "yaw":    <rad>,       # bin 회전각
    #   "size":   [width, height]  # bin 내부 치수 (m)
    # }
    # =========================================================
    def bin_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            center = data["center"]   # [cx, cy]
            yaw    = float(data["yaw"])
            size   = data["size"]     # [width, height]
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().error(f"bin_callback 파싱 실패: {e}")
            return

        cx, cy   = float(center[0]), float(center[1])
        full_x   = float(size[0]) + self.bin_thick   # 외벽 포함 X 폭
        full_y   = float(size[1]) + self.bin_thick   # 외벽 포함 Y 폭
        height_wall = 0.15                            # 벽 높이 (필요 시 JSON에서 받아오도록 수정)

        self.get_logger().info(
            f"Spawning Bin: center=({cx:.3f},{cy:.3f}), "
            f"size=({full_x:.3f}x{full_y:.3f}), yaw={np.degrees(yaw):.1f}°"
        )

        # ── yaw 회전을 쿼터니언으로 변환 ──
        # yaw: world Z축 기준 회전
        quat = tf.quaternion_from_euler(0.0, 0.0, yaw)  # [w, x, y, z]

        def make_pose(local_x, local_y, local_z=height_wall / 2.0):
            """bin 중심 기준 로컬 오프셋을 yaw 회전 적용해 world 포즈로 변환"""
            # 로컬 → world 회전
            rot = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw),  np.cos(yaw)]
            ])
            wx, wy = rot @ np.array([local_x, local_y])
            p = Pose()
            p.position.x = cx + wx
            p.position.y = cy + wy
            p.position.z = local_z
            p.orientation.w = float(quat[0])
            p.orientation.x = float(quat[1])
            p.orientation.y = float(quat[2])
            p.orientation.z = float(quat[3])
            return p

        half_x = full_x / 2.0
        half_y = full_y / 2.0

        # 벽 4개의 로컬 오프셋 (px, nx, py, ny)
        wall_offsets = [
            ( half_x,    0.0 ),   # +X 벽
            (-half_x,    0.0 ),   # -X 벽
            (   0.0,   half_y),   # +Y 벽
            (   0.0,  -half_y),   # -Y 벽
        ]

        def make_wall(dims):
            p = SolidPrimitive()
            p.type = SolidPrimitive.BOX
            p.dimensions = list(dims)
            return p

        # 기존 bin 제거 후 재생성 (실시간 갱신)
        remove_obj = CollisionObject()
        remove_obj.header.frame_id = self.target_frame
        remove_obj.header.stamp    = self.get_clock().now().to_msg()
        remove_obj.id              = "World/bin"
        remove_obj.operation       = CollisionObject.REMOVE
        self.collision_pub.publish(remove_obj)

        bin_obj = CollisionObject()
        bin_obj.header.frame_id = self.target_frame
        bin_obj.header.stamp    = self.get_clock().now().to_msg()
        bin_obj.id              = "World/bin"
        bin_obj.operation       = CollisionObject.ADD

        bin_obj.primitives = [
            make_wall([self.bin_thick, full_y, height_wall]),  # +X 벽
            make_wall([self.bin_thick, full_y, height_wall]),  # -X 벽
            make_wall([full_x, self.bin_thick, height_wall]),  # +Y 벽
            make_wall([full_x, self.bin_thick, height_wall]),  # -Y 벽
        ]

        bin_obj.primitive_poses = [
            make_pose(ox, oy) for ox, oy in wall_offsets
        ]

        self.collision_pub.publish(bin_obj)

        self.get_logger().info("Bin walls updated successfully (with yaw).")

    # =========================================================
    # [Section B] Object (물체) 동기화 로직
    # =========================================================
    def names_callback(self, msg: String):
        try:
            self.object_names = json.loads(msg.data)
        except json.JSONDecodeError:
            self.object_names = []

    def poses_callback(self, msg: PoseArray):
        if self.mesh_msg is None:
            return

        if len(msg.poses) != len(self.object_names):
            return

        current_frame_id = msg.header.frame_id or self.target_frame

        current_ids  = set(self.object_names)
        ids_to_remove = self.known_object_ids - current_ids

        if ids_to_remove:
            for obj_id in ids_to_remove:
                remove_obj = CollisionObject()
                remove_obj.header.frame_id = current_frame_id
                remove_obj.header.stamp    = self.get_clock().now().to_msg()
                remove_obj.id              = obj_id
                remove_obj.operation       = CollisionObject.REMOVE
                remove_obj.meshes          = []
                remove_obj.primitives      = []
                self.collision_pub.publish(remove_obj)

        for name, pose in zip(self.object_names, msg.poses):
            current_quat = [
                pose.orientation.w, pose.orientation.x,
                pose.orientation.y, pose.orientation.z
            ]
            mat_pose = tf.quaternion_matrix(current_quat)
            mat_pose[0:3, 3] = [pose.position.x, pose.position.y, pose.position.z]

            new_mat  = np.dot(mat_pose, self.correction_matrix)
            new_quat = tf.quaternion_from_matrix(new_mat)  # [w, x, y, z]

            pose.position.x    = new_mat[0, 3]
            pose.position.y    = new_mat[1, 3]
            pose.position.z    = new_mat[2, 3]
            pose.orientation.w = new_quat[0]
            pose.orientation.x = new_quat[1]
            pose.orientation.y = new_quat[2]
            pose.orientation.z = new_quat[3]

            col_obj = CollisionObject()
            col_obj.header.frame_id = current_frame_id
            col_obj.header.stamp    = self.get_clock().now().to_msg()
            col_obj.id              = name
            col_obj.meshes          = [self.mesh_msg]
            col_obj.mesh_poses      = [pose]
            col_obj.operation       = CollisionObject.ADD

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