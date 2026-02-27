"""
Auto Hand-Eye Calibration Runner (Eye-to-Hand)
- 카메라가 고정, ChArUco 보드가 로봇 EE에 부착된 구조
- 기준 자세(보드가 카메라에 잘 보이는 자세)를 중심으로 4가지 회전 오프셋 적용
  1. 월드 Z 공전 (그리퍼 Z가 월드 XY 평면에서 회전)
  2. 그리퍼 Z 자전 (joint6 yaw)
  3. Roll  (그리퍼 로컬 X)
  4. Pitch (그리퍼 로컬 Y)
- UnifiedCalibrationNode와 함께 사용 (keypress_topic으로 트리거)
"""
import sys
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import yaml
import time
from scipy.spatial.transform import Rotation as R_scipy
from geometry_msgs.msg import TransformStamped
import tf2_ros

current_dir = os.path.dirname(os.path.abspath(__file__))
helper_path = os.path.abspath(os.path.join(current_dir, '../../../bin_picking'))
sys.path.append(helper_path)

from bin_picking.moveit_helper_functions import MoveItMoveHelper


class AutoEyeToHandRunner(MoveItMoveHelper):
    def __init__(self):
        super().__init__()

        config_path = os.path.join(current_dir, '..', 'config.yaml')
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.trigger_pub = self.create_publisher(String, 'keypress_topic', 10)
    #========================================================================================
        config_path = 'src/handeye_calibration_ros2/eye_to_hand_spinnaker/config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.handeye_result_file_name = config["handeye_result_file_name"]
        with open(self.handeye_result_file_name, 'r') as file:
            config=yaml.safe_load(file)
        self.camera_trans=np.array(config['translation'])
        self.camera_rotation=np.array(config['rotation']).reshape(3,3)
        quat=R_scipy.from_matrix(self.camera_rotation).as_quat()
        broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        t=TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_link'
        t.transform.translation.x = self.camera_trans[0]
        t.transform.translation.y = self.camera_trans[1]
        t.transform.translation.z = self.camera_trans[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        broadcaster.sendTransform(t)
    #==============================================================================================
    @staticmethod
    def generate_random_calibration_poses(num_poses: int, seed: int = None,
                                           world_z_range=(-90, 0)):
        """
        Hand-Eye 캘리브레이션용 랜덤 포즈 생성

        Position 범위:
            x: [-0.06, 0.00] m
            y: [-0.03, 0.06] m
            z: [-0.05, 0.05] m

        Orientation 범위:
            world_z:     world_z_range° — 월드 Z축 공전
            gripper_yaw: [-20, 20]° — 그리퍼 Z축 자전 (joint6)
            roll:        [-20, 20]° — 그리퍼 로컬 X
            pitch:       [-20, 20]° — 그리퍼 로컬 Y

        Returns:
            position_offsets: list of [x, y, z]
            orientation_offsets_deg: list of (world_z, gripper_yaw, roll, pitch)
        """
        if seed is not None:
            np.random.seed(seed)

        position_offsets = np.column_stack([
            np.random.uniform(-0.06, 0.03,  num_poses),
            np.random.uniform(0.0, 0.06, num_poses),
            np.random.uniform(-0.05, 0.05, num_poses),
        ])

        world_z_degs     = np.random.uniform(world_z_range[0], world_z_range[1], num_poses)
        gripper_yaw_degs = np.random.uniform(-20, 20, num_poses)
        roll_degs        = np.random.uniform(-20, 20, num_poses)
        pitch_degs       = np.random.uniform(-20, 20, num_poses)

        orientation_offsets_deg = np.column_stack([
            world_z_degs, gripper_yaw_degs, roll_degs, pitch_degs
        ])

        # 첫 번째는 항상 원점 (기준 포즈 확인용)
        position_offsets[0] = [0.0, 0.0, 0.0]
        orientation_offsets_deg[0] = [0.0, 0.0, 0.0, 0.0]

        return (
            position_offsets.tolist(),
            [tuple(row) for row in orientation_offsets_deg.tolist()],
        )

    def run_auto_calib(self, ref_pos, ref_quat_xyzw, num_poses=1, seed=42,
                       world_z_range=(-90, 0), sort_ascending=True):
        self.move_to_joint_values(joint_goal = {
                "joint_1": 0.0,
                "joint_2": 0.0,
                "joint_3": 1.57,
                "joint_4": 0.0,
                "joint_5": 1.57,
                "joint_6": 0.0
            })
        """
        Eye-to-Hand 캘리브레이션 자동 실행

        Args:
            ref_pos: 보드가 카메라에 잘 보이는 기준 위치 [x, y, z]
            ref_quat_xyzw: 해당 위치에서의 기준 자세 [qx, qy, qz, qw]
            num_poses: 생성할 포즈 수
            seed: 랜덤 시드 (재현성)
            world_z_range: 월드 Z 공전 범위 (min_deg, max_deg)
            sort_ascending: True면 오름차순, False면 내림차순 정렬
        """
        self.get_logger().info("=== Eye-to-Hand 자동 캘리브레이션 시작 ===")
        

        ref_pos = np.array(ref_pos)
        R_ref = R_scipy.from_quat(ref_quat_xyzw)

        # ── 포즈 생성 ──
        position_offsets, orientation_offsets_deg = \
            self.generate_random_calibration_poses(num_poses, seed=seed,
                                                    world_z_range=world_z_range)

        # ── 월드 Z 공전 기준 정렬 (부드러운 경로) ──
        poses = list(zip(position_offsets, orientation_offsets_deg))
        poses.sort(key=lambda p: p[1][0], reverse=(not sort_ascending))
        position_offsets, orientation_offsets_deg = zip(*poses)

        total = len(position_offsets)
        pose_count = 0
        success_count = 0

        self.get_logger().info(f"  기준 위치: {ref_pos.tolist()}")
        self.get_logger().info(f"  기준 자세(xyzw): {ref_quat_xyzw}")
        self.get_logger().info(f"  계획 포즈 수: {total}")
        self.get_logger().info(
            f"  월드Z 범위: {world_z_range}, "
            f"정렬: {'오름차순' if sort_ascending else '내림차순'}"
        )

        for pos_offset, (wz, gyaw, roll, pitch) in zip(position_offsets, orientation_offsets_deg):
            pose_count += 1

            # ── 위치 = 기준 + 오프셋 ──
            pos = ref_pos + np.array(pos_offset)

            # ── 회전 합성 ──
            # 1. 월드 Z 공전
            R_world_z = R_scipy.from_euler('z', wz, degrees=True)
            # 2. 그리퍼 Z 자전 (joint6)
            R_local_z = R_scipy.from_euler('z', gyaw, degrees=True)
            # 3. Roll(X) + Pitch(Y) 로컬
            R_rp = R_scipy.from_euler('xy', [roll, pitch], degrees=True)

            # 합성: 월드Z공전 × 기준자세 × 로컬Z자전 × Roll/Pitch
            R_final = R_world_z * R_ref * R_local_z * R_rp
            q_final = R_final.as_quat().tolist()  # [x, y, z, w]

            self.get_logger().info(
                f"[{pose_count}/{total}] "
                f"pos=({pos_offset[0]:+.3f}, {pos_offset[1]:+.3f}, {pos_offset[2]:+.3f}) "
                f"wZ={wz:+.1f}, gYaw={gyaw:+.1f}, "
                f"roll={roll:+.1f}, pitch={pitch:+.1f}"
            )

            if self.move_cartesian(pos.tolist(), q_final):
                self._wait_with_spin(4.0)
                self.trigger_pub.publish(String(data='r'))
                success_count += 1
                self.get_logger().info(f"  → saved ({success_count}번째)")
                self._wait_with_spin(1.0)
            else:
                self.get_logger().warn(f"  → move failed, skip")
                return False

        self.get_logger().info(
            f"=== 완료: {success_count}/{total} 포즈 저장 ==="
        )

    def _wait_with_spin(self, duration):
        start = time.time()
        while time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.1)


def main():
    rclpy.init()
    runner = AutoEyeToHandRunner()

    # ── 1차: 기준 자세 A, world_z -90→0 (오름차순) ──
    runner.get_logger().info("===== Phase 1: ref_quat A =====")
    runner.run_auto_calib(
        ref_pos=[0.55, 0.2, 0.525],
        ref_quat_xyzw=[0.5, 0.5, 0.5, 0.5],
        num_poses=20,
        seed=42,
        world_z_range=(-90, 0),
        sort_ascending=False,
    )

    # ── 2차: 기준 자세 B, world_z 90→0 (내림차순) ──
    runner.get_logger().info("===== Phase 2: ref_quat B =====")
    runner.run_auto_calib(
        ref_pos=[0.55, 0.2, 0.525],
        ref_quat_xyzw=[0.7071, 0.0, 0.0, 0.7071],
        num_poses=20,
        seed=43,
        world_z_range=(0, 90),
        sort_ascending=True,
    )

    rclpy.shutdown()


if __name__ == '__main__':
    main()