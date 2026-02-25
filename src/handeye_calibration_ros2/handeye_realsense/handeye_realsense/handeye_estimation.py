"""
Copyright © 2024 Shengyang Zhuang. All rights reserved.

Contact: https://shengyangzhuang.github.io/
"""
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
import os

class HandEyeCalibrationNode(Node):
    def __init__(self):
        super().__init__('hand_eye_calibration_node')
        self.get_logger().info("Starting Hand-Eye Calibration Node")
        
        # with open('src/handeye_calibration_ros2/handeye_realsense/config.yaml', 'r') as file:
        with open('src/handeye_calibration_ros2/eye_to_hand_spinnaker/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.robot_data_file_name = config["robot_data_file_name"]
        self.marker_data_file_name = config["marker_data_file_name"]
        self.handeye_result_file_name = config["handeye_result_file_name"]
        self.handeye_result_profile_file_name = config["handeye_result_profile_file_name"]

        # Load transformation data from YAML files
        self.R_gripper2base, self.t_gripper2base = self.load_transformations(self.robot_data_file_name)
        self.R_target2cam, self.t_target2cam = self.load_transformations(self.marker_data_file_name)

        # Compute the hand-eye transformation matrix
        self.compute_hand_eye()

    def load_transformations(self, file_path):
        import yaml
        import numpy as np

        print(f"[INFO] '{file_path}' 파일 읽기를 시작합니다...")
        
        with open(file_path, 'r') as file:
            try:
                data = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(f"[ERROR] YAML 파일 파싱 에러 (파일이 깨졌을 수 있습니다):\n{exc}")
                return [], []

            # 'poses' 키가 없으면 빈 리스트 반환
            poses = data.get('poses', []) 
            print(f"[INFO] 총 {len(poses)}개의 포즈 데이터를 찾았습니다.")

        R = []
        t = []

        # enumerate를 사용해 몇 번째 데이터인지 추적합니다.
        for i, pose in enumerate(poses):
            # 1. pose가 딕셔너리 형태인지 확인 (파일 형식 오류 방지)
            if not isinstance(pose, dict):
                print(f"[ERROR] 인덱스 {i}: 데이터가 딕셔너리 형태가 아닙니다. (타입: {type(pose)})\n값: {pose}")
                continue

            # 2. 필수 키가 존재하는지 확인
            if 'rotation' not in pose or 'translation' not in pose:
                print(f"[ERROR] 인덱스 {i}: 'rotation' 또는 'translation' 키가 누락되었습니다!")
                print(f"        가지고 있는 키: {list(pose.keys())}")
                print(f"        실제 데이터: {pose}")
                continue

            # 3. 데이터 변환 및 추가
            try:
                rotation = np.array(pose['rotation'], dtype=np.float32)
                translation = np.array(pose['translation'], dtype=np.float32)

                R.append(rotation)
                t.append(translation)
            except Exception as e:
                print(f"[FATAL] 인덱스 {i} 배열 변환 중 에러 발생: {e}")
                print(f"        문제의 데이터: {pose}")

        print(f"[INFO] 성공적으로 변환된 데이터: {len(R)}/{len(poses)} 개")
        return R, t

    def compute_hand_eye(self):
        self.get_logger().info(f"Loaded {len(self.R_gripper2base)} rotations and {len(self.t_gripper2base)} translations for gripper to base")
        self.get_logger().info(f"Loaded {len(self.R_target2cam)} rotations and {len(self.t_target2cam)} translations for target to camera")
        rotations = [r.reshape(3, 3) for r in self.R_gripper2base]
        translations = [t.reshape(3, 1) for t in self.t_gripper2base]
        obj_rotations = [r.reshape(3, 3) for r in self.R_target2cam]
        obj_translations = [t.reshape(3, 1) for t in self.t_target2cam]


        # Perform hand-eye calibration
        R, t = cv2.calibrateHandEye(
            rotations, translations, obj_rotations, obj_translations,
            method=4)

        # Save results to YAML
        # Output: camera relative to gripper frame (eye to hand)
        self.save_yaml(R, t)
        #self.save_yaml_profile(R_qua, t)
    
    def rotation_matrix_to_quaternion(self, matrix):
        """Convert a 3x3 rotation matrix into a quaternion."""
        rotation = R.from_matrix(matrix)
        return rotation.as_quat()

    def save_yaml(self, R, t):
        '''This function will always show only the updated result'''
        new_data = {'rotation': R.flatten().tolist(), 'translation': t.flatten().tolist()}

        # Write the new data to the YAML file, overwriting any existing content
        with open(self.handeye_result_file_name, 'w') as file:
            yaml.safe_dump(new_data, file)

        self.get_logger().info("Simulated hand-eye calibration results saved.")
        print(f"Rotation matrix: {R}")
        print(f"Translation vector: {t}")
        
    def save_yaml_profile(self, R, t):
        '''This function saves the rotation and translation data in the correct format.'''
        new_data = {'rotation': R.flatten().tolist(), 'translation': t.flatten().tolist()}

        # Check if the file exists and is not empty
        if os.path.exists(self.handeye_result_profile_file_name) and os.path.getsize(self.handeye_result_profile_file_name) > 0:
            # Load the existing data from the file
            with open(self.handeye_result_profile_file_name, 'r') as file:
                existing_data = yaml.safe_load(file)

            # If the file contains data, append the new transform
            if 'transforms' in existing_data:
                existing_data['transforms'].append(new_data)
            else:
                existing_data = {'transforms': [new_data]}
        else:
            # If the file does not exist or is empty, start with a new structure
            existing_data = {'transforms': [new_data]}

        # Save the updated structure back to the file
        with open(self.handeye_result_profile_file_name, 'w') as file:
            yaml.safe_dump(existing_data, file)

        self.get_logger().info("Simulated hand-eye calibration results saved.")
        print(f"Rotation matrix quaternion: {R}")
        print(f"Translation vector: {t}")


def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
