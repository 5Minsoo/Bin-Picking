#!/usr/bin/env python3
"""
MoveIt2 Planning Scene - 4 Walls + Ceiling + Floor (ROS2)
/apply_planning_scene 서비스를 사용하여 ACM을 보존합니다.

[충돌 뚫림 방지]
  1. WALL_THICKNESS를 충분히 두껍게 설정 (기본 0.05m)
  2. longest_valid_segment_fraction 파라미터를 낮춰 충돌 체크 해상도를 높임

Usage:
  ros2 run your_package add_walls_ros2.py
  ros2 run your_package add_walls_ros2.py --remove
"""

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
import sys


# ============================================================
#  설정 (여기서 수정하세요!)
# ============================================================

# 두께: 너무 얇으면 플래너가 뚫고 지나감!
# 0.02m → 뚫림 자주 발생
# 0.05m 이상 권장
WALL_THICKNESS = 0.1  # (m)

# --- 수직 벽 4개 ---
WALLS = [
{
"name": "wall_front",
"x": 0.95,
"y": 0.0,
"z": 0.5,
"length": 1.0,
"height": 1.0,
"orientation": "y",
},
{
"name": "wall_back",
"x": -0.9,
"y": 0.0,
"z": 0.5,
"length": 1.0,
"height": 1.0,
"orientation": "y",
},
{
"name": "wall_left",
"x": 0.0,
"y": 0.85,
"z": 0.5,
"length": 1.0,
"height": 1.0,
"orientation": "x",
},
{
"name": "wall_right",
"x": 0.0,
"y": -0.35,
"z": 0.5,
"length": 1.0,
"height": 1.0,
"orientation": "x",
},
]

# --- 바닥 & 천장 ---
FLOOR = {
"name": "floor",
"x": 0.7, # 중심 x
"y": 0.0, # 중심 y
"z": 0.1, # 바닥 z 위치
"size_x": 0.6, # x방향 크기 (m)
"size_y": 0.8, # y방향 크기 (m)
}

FLOOR1 = {
"name": "floor1",
"x": -0.5, # 중심 x
"y": 0.0, # 중심 y
"z": -0.1, # 바닥 z 위치
"size_x": 1.0, # x방향 크기 (m)
"size_y": 0.8, # y방향 크기 (m)
}

CEILING = {
"name": "ceiling",
"x": 0.0, # 중심 x
"y": 0.0, # 중심 y
"z": 1.35, # 천장 z 위치
"size_x": 2.0, # x방향 크기 (m)
"size_y": 2.0, # y방향 크기 (m)
}



class WallCollisionNode(Node):
    def __init__(self):
        super().__init__("collision_wall_node")

        self.apply_scene_cli = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )

        self.get_logger().info("Waiting for /apply_planning_scene service...")
        if not self.apply_scene_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                "/apply_planning_scene service not available! "
                "Is move_group running?"
            )
            raise RuntimeError("Service not available")
        self.get_logger().info("Service connected.")

        # ──────────────────────────────────────────────
        # 충돌 체크 해상도 높이기 (뚫림 방지 핵심)
        # longest_valid_segment_fraction: 기본값 0.05 (5%)
        #   → 0.005 (0.5%)로 낮추면 경로를 훨씬 촘촘히 검사
        #   → 플래닝 시간이 약간 늘어날 수 있음
        # ──────────────────────────────────────────────
        self.declare_parameter(
            "move_group.longest_valid_segment_fraction", 0.005
        )
        self._set_collision_resolution()

    def _set_collision_resolution(self):
        """
        move_group 노드의 longest_valid_segment_fraction 파라미터를 설정합니다.
        이 파라미터가 작을수록 경로 상의 충돌 체크 포인트가 많아져서
        얇은 오브젝트를 뚫고 지나가는 것을 방지합니다.
        """
        try:
            from rcl_interfaces.srv import SetParameters
            from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

            set_param_cli = self.create_client(
                SetParameters, "/move_group/set_parameters"
            )

            if not set_param_cli.wait_for_service(timeout_sec=5.0):
                self.get_logger().warn(
                    "Could not reach /move_group/set_parameters. "
                    "Collision resolution not updated. "
                    "Consider setting longest_valid_segment_fraction "
                    "in your ompl_planning.yaml instead."
                )
                return

            param = Parameter()
            param.name = "longest_valid_segment_fraction"
            param.value = ParameterValue()
            param.value.type = ParameterType.PARAMETER_DOUBLE
            param.value.double_value = 0.005

            request = SetParameters.Request()
            request.parameters = [param]

            future = set_param_cli.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

            if future.result() is not None:
                self.get_logger().info(
                    "longest_valid_segment_fraction set to 0.005 "
                    "(collision check resolution increased)"
                )
            else:
                self.get_logger().warn("Failed to set collision resolution parameter.")

        except Exception as e:
            self.get_logger().warn(
                f"Could not set collision resolution dynamically: {e}. "
                f"Set it in ompl_planning.yaml instead."
            )

    def _make_wall_co(self, wall_cfg, operation=CollisionObject.ADD, frame_id="world"):
        """수직 벽 CollisionObject 생성"""
        co = CollisionObject()
        co.header.frame_id = frame_id
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = wall_cfg["name"]
        co.operation = operation

        if operation == CollisionObject.REMOVE:
            return co

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX

        length = wall_cfg["length"]
        height = wall_cfg["height"]

        if wall_cfg["orientation"] == "x":
            box.dimensions = [length, WALL_THICKNESS, height]
        else:
            box.dimensions = [WALL_THICKNESS, length, height]

        pose = Pose()
        pose.position.x = float(wall_cfg["x"])
        pose.position.y = float(wall_cfg["y"])
        pose.position.z = float(wall_cfg["z"])
        pose.orientation.w = 1.0

        co.primitives.append(box)
        co.primitive_poses.append(pose)
        return co

    def _make_plane_co(self, plane_cfg, operation=CollisionObject.ADD, frame_id="world"):
        """수평면(바닥/천장) CollisionObject 생성"""
        co = CollisionObject()
        co.header.frame_id = frame_id
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = plane_cfg["name"]
        co.operation = operation

        if operation == CollisionObject.REMOVE:
            return co

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [plane_cfg["size_x"], plane_cfg["size_y"], WALL_THICKNESS]

        pose = Pose()
        pose.position.x = float(plane_cfg["x"])
        pose.position.y = float(plane_cfg["y"])
        pose.position.z = float(plane_cfg["z"])
        pose.orientation.w = 1.0

        co.primitives.append(box)
        co.primitive_poses.append(pose)
        return co

    def apply_scene(self, collision_objects: list):
        """is_diff=True로 서비스 호출 → ACM 보존"""
        scene_msg = PlanningScene()
        scene_msg.is_diff = True
        scene_msg.world.collision_objects = collision_objects

        request = ApplyPlanningScene.Request()
        request.scene = scene_msg

        future = self.apply_scene_cli.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is not None and future.result().success:
            self.get_logger().info("PlanningScene applied successfully.")
            return True
        else:
            self.get_logger().error("Failed to apply PlanningScene!")
            return False

    def add_all(self):
        """벽 4개 + 바닥 + 천장 추가"""
        cos = []

        for w in WALLS:
            cos.append(self._make_wall_co(w, CollisionObject.ADD))
            self.get_logger().info(
                f"  Wall '{w['name']}' at ({w['x']}, {w['y']}, {w['z']}) "
                f"len={w['length']} h={w['height']} orient={w['orientation']}"
            )

        cos.append(self._make_plane_co(FLOOR, CollisionObject.ADD))
        self.get_logger().info(
            f"  Floor '{FLOOR['name']}' at z={FLOOR['z']} "
            f"size=({FLOOR['size_x']} x {FLOOR['size_y']})"
        )

        cos.append(self._make_plane_co(CEILING, CollisionObject.ADD))
        self.get_logger().info(
            f"  Ceiling '{CEILING['name']}' at z={CEILING['z']} "
            f"size=({CEILING['size_x']} x {CEILING['size_y']})"
        )

        cos.append(self._make_plane_co(FLOOR1, CollisionObject.ADD))
        self.get_logger().info(
            f"  Floor '{FLOOR1['name']}' at z={FLOOR1['z']} "
            f"size=({FLOOR1['size_x']} x {FLOOR1['size_y']})"
        )

        if self.apply_scene(cos):
            self.get_logger().info(
                f"=== 6 collision objects added "
                f"(4 walls + floor + ceiling, thickness={WALL_THICKNESS}m) ==="
            )

    def remove_all(self):
        """모두 제거"""
        cos = []
        for w in WALLS:
            cos.append(self._make_wall_co(w, CollisionObject.REMOVE))
        cos.append(self._make_plane_co(FLOOR, CollisionObject.REMOVE))
        cos.append(self._make_plane_co(CEILING, CollisionObject.REMOVE))

        if self.apply_scene(cos):
            self.get_logger().info("=== All collision objects removed ===")


def main():
    rclpy.init(args=sys.argv)
    try:
        node = WallCollisionNode()
        if "--remove" in sys.argv:
            node.remove_all()
        else:
            node.add_all()
    except RuntimeError as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()