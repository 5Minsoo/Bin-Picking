"""
Copyright © 2024 Shengyang Zhuang. All rights reserved.

Contact: https://shengyangzhuang.github.io/
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# tf2 관련 모듈 추가
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from scipy.spatial.transform import Rotation as R
import yaml
import numpy as np

class RobotTransformNode(Node):
    def __init__(self):
        super().__init__('robot_transform_node')
        
        # 키보드 입력 토픽 구독 (그대로 유지)
        self.subscription_keypress = self.create_subscription(String, 'keypress_topic', self.keypress_callback, 10)
        self.pose_count = 0
        
        # config.yaml 읽기 (그대로 유지)
        with open('src/handeye_calibration_ros2/handeye_realsense/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.robot_data_file_name = config["robot_data_file_name"]
        self.base_link = config["base_link"]
        self.ee_link = config["ee_link"]

        # ---------------- 변경된 부분 ----------------
        # 수동 구독(/tf, /tf_static) 대신 TF Buffer와 Listener 초기화
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info("TF2 Listener initialized. Waiting for transforms...")
        # ---------------------------------------------

    def quaternion_to_rotation_matrix(self, x, y, z, w):
        """ Convert a quaternion into a full three-dimensional rotation matrix. """
        return R.from_quat([x, y, z, w]).as_matrix()

    def save_transformation_to_yaml(self, rotation_matrix, translation_vector):
        """ Append the rotation matrix and translation vector to a YAML file and print them. """
        yaml_file_path = self.robot_data_file_name
        try:
            with open(yaml_file_path, 'r') as file:
                data = yaml.safe_load(file) or {'poses': []}
        except FileNotFoundError:
            data = {'poses': []}

        data['poses'].append({
            'rotation': rotation_matrix.tolist(),
            'translation': translation_vector.tolist()
        })

        with open(yaml_file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

        self.pose_count += 1
        print(f"Pose {self.pose_count}:")
        print("Rotation Matrix:")
        print(rotation_matrix)
        print("Translation Vector:")
        print(translation_vector)
        self.get_logger().info(f'Transformation for Pose {self.pose_count} appended to {self.robot_data_file_name}')
    
    def keypress_callback(self, msg):
        key = msg.data
        if key == 'r':
            # ---------------- 변경된 부분 ----------------
            try:
                # Buffer를 통해 base_link에서 ee_link까지의 현재 변환 정보를 한 번에 조회
                now = rclpy.time.Time()
                trans = self.tf_buffer.lookup_transform(
                    self.base_link,    # Target frame (기준 좌표계)
                    self.ee_link,      # Source frame (알고자 하는 좌표계)
                    now)               # 조회할 시간 (Time()은 가장 최신 데이터를 의미함)
                
                # Translation vector (1x3)
                translation = [
                    trans.transform.translation.x, 
                    trans.transform.translation.y, 
                    trans.transform.translation.z
                ]
                
                # Rotation quaternion (x, y, z, w)
                rotation = [
                    trans.transform.rotation.x, 
                    trans.transform.rotation.y, 
                    trans.transform.rotation.z, 
                    trans.transform.rotation.w
                ]

                # 변환 수행 (기존 로직과 동일)
                R_gripper2base = self.quaternion_to_rotation_matrix(*rotation)
                t_gripper2base = np.array(translation)
                
                self.save_transformation_to_yaml(R_gripper2base, t_gripper2base)

            except (LookupException, ConnectivityException, ExtrapolationException) as ex:
                self.get_logger().error(f'Could not transform {self.base_link} to {self.ee_link}: {ex}')
            # ---------------------------------------------
                
        elif key == 'q':
            self.get_logger().info("Ending program...")
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    robot_transform_node = RobotTransformNode()

    try:
        rclpy.spin(robot_transform_node)
    finally:
        robot_transform_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()