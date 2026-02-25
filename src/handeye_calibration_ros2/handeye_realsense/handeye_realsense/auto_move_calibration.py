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

    def run_auto_calib(self, marker_xyz):
        self.get_logger().info("Starting automatic trajectory...")
        target = np.array(marker_xyz)

        # 거리(Z축 깊이) 변화 추가: 예를 들어 35cm, 45cm, 55cm
        distances = [0.55, 0.60, 0.65] 
        tilts = [15, 20, 25, 30, 35, 45]
        pans = [0, 90,  180,  240]

        # 1. 최상위 루프: 거리(Z) 변화
        for dist in distances:
            # 2. 중간 루프: 기울기(Tilt) 변화
            for t in tilts:
                # 3. 최하위 루프: 회전(Pan) 변화
                for p in pans:
                    rad_t, rad_p = math.radians(t), math.radians(p)
                    
                    # 구면 좌표계를 이용한 Cartesian Position(cp) 계산
                    cp = [
                        target[0] + dist * math.sin(rad_t) * math.cos(rad_p),
                        target[1] + dist * math.sin(rad_t) * math.sin(rad_p),
                        target[2] + dist * math.cos(rad_t)
                    ]
                    
                    # 마커를 바라보도록 그리퍼(카메라) 자세 방향 쿼터니언 계산
                    q_obj = self.make_grasp_quat_for_approach(
                        -(target - np.array(cp)), np.array([1, 0, 0])
                    )
                    q_list = [q_obj.x, q_obj.y, q_obj.z, q_obj.w]

                    # MoveIt을 통한 로봇 이동 명령
                    if self.move_cartesian(cp, q_list):
                        # 2. 로봇이 완전히 멈추고 진동이 잦아들 시간 대기 (매우 중요)
                        self._wait_with_spin(2.0)

                        # 이미지 캡처 트리거 신호 전송
                        self.trigger_pub.publish(String(data='r'))
                        self.get_logger().info(f"Published 'r' for dist={dist}, tilt={t}, pan={p}")
                        
                        # 3. ArucoNode가 이미지를 처리하고 저장할 시간 확보
                        self._wait_with_spin(1.0)
                    else:
                        self.get_logger().warn(f"Move failed for dist={dist}, tilt={t}, pan={p}")

        self.get_logger().info("Calibration trajectory complete!")

    def _wait_with_spin(self, duration):
        start = time.time()
        while time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.1)


def main():
    rclpy.init()
    runner = AutoHandEyeRunner()

    # 1. DDS discovery 대기 - subscriber(ArucoNode)와 연결 확인
    runner.get_logger().info("Waiting for ArucoNode subscriber...")
    while runner.trigger_pub.get_subscription_count() == 0:
        rclpy.spin_once(runner, timeout_sec=0.1)
    runner.get_logger().info(f"Connected! ({runner.trigger_pub.get_subscription_count()} subscribers)")

    runner.run_auto_calib(marker_xyz=[-0.133, -0.514, 0.15])
    runner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()