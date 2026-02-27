import sys
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Quaternion
import numpy as np
import yaml
import time
import math
import random

# moveit_helper_functions 위치를 path에 추가 (상대 경로 임포트)
current_dir = os.path.dirname(os.path.abspath(__file__))
# ../../../../bin_picking/bin_picking 경로 계산
helper_path = os.path.abspath(os.path.join(current_dir, '../../../bin_picking'))
sys.path.append(helper_path)

from bin_picking.moveit_helper_functions import MoveItMoveHelper # 클래스명이 MoveItHelper라고 가정

class AutoHandEyeRunner(MoveItMoveHelper): # 상속해서 함수들 사용
    def __init__(self):
        # MoveItHelper의 __init__이 있다면 super().__init__() 호출 필요
        super().__init__() 
        
        # 로봇 데이터 저장 파일 경로 (상대 경로)
        config_path = os.path.join(current_dir, '..', 'config.yaml')
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        self.robot_data_path = os.path.join(current_dir, '..', self.cfg["robot_data_file_name"])
        self.trigger_pub = self.create_publisher(String, 'keypress_topic', 10)

# 그리퍼 원점 → 카메라 렌즈까지의 오프셋 (로봇 기준 측정 필요)
# 예: 카메라가 그리퍼 기준으로 x +0.03, y 0, z +0.05 에 있다면


    def run_auto_calib(self, marker_xyz, num_shots=40):
        self.get_logger().info(f"Starting random calibration: {num_shots} shots")
        target = np.array(marker_xyz)
        camera_offset = np.array([-0.08, -0.01, 0.04])
        dist_range = (0.35, 0.55)
        tilt_range = (0, 45)
        pan_range = (0, 360)

        count = 0
        while count < num_shots:
            dist = random.uniform(*dist_range)
            t = random.uniform(*tilt_range)
            p = random.uniform(*pan_range)

            rad_t, rad_p = math.radians(t), math.radians(p)

            # 카메라 렌즈가 이 위치에 오도록 계산
            cam_pos = [
                target[0] + dist * math.sin(rad_t) * math.cos(rad_p),
                target[1] + dist * math.sin(rad_t) * math.sin(rad_p),
                target[2] + dist * math.cos(rad_t)
            ]

            # 카메라가 마커를 바라보는 방향
            direction = target - np.array(cam_pos)
            q_obj = self.make_grasp_quat_for_approach(
                -direction, np.array([1, 0, 0])
            )

            # 카메라 위치 → 그리퍼 위치로 역변환
            # 그리퍼 회전 행렬 구해서 오프셋을 빼줌
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_quat([q_obj.x, q_obj.y, q_obj.z, q_obj.w])
            gripper_pos = np.array(cam_pos) - rot.apply(camera_offset)

            q_list = [q_obj.x, q_obj.y, q_obj.z, q_obj.w]
            cp = gripper_pos.tolist()

            if self.move_cartesian(cp, q_list):
                self._wait_with_spin(2.0)
                self.trigger_pub.publish(String(data='r'))
                count += 1
                self.get_logger().info(
                    f"[{count}/{num_shots}] dist={dist:.3f}, tilt={t:.1f}, pan={p:.1f}"
                )
                self._wait_with_spin(1.0)
            else:
                self.get_logger().warn(
                    f"Move failed: dist={dist:.3f}, tilt={t:.1f}, pan={p:.1f} — retrying"
                )

    def _wait_with_spin(self, duration):
        start = time.time()
        while time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.1)


def main():
    rclpy.init()
    runner = AutoHandEyeRunner()
    runner.run_auto_calib(marker_xyz=[-0.699, -0.067, 0.1])
    runner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()