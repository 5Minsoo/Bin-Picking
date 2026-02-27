import sys
import os
import time
import math
import numpy as np
import cv2
import yaml
import random
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Quaternion, TransformStamped
from cv_bridge import CvBridge

# tf2 임포트
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# 사용자 헬퍼 함수 임포트
current_dir = os.path.dirname(os.path.abspath(__file__))
helper_path = os.path.abspath(os.path.join(current_dir, '../../../bin_picking'))
sys.path.append(helper_path)
from bin_picking.moveit_helper_functions import MoveItMoveHelper

class AutoHandEyeVerifier(MoveItMoveHelper):
    def __init__(self):
        super().__init__()
        
        # 1. Config YAML 로드
        config_path = os.path.join(current_dir, '..', 'config.yaml')
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        # 2. Config 기반 파라미터 설정
        self.base_frame = self.cfg.get('base_link', 'base_link')
        self.tcp_frame = self.cfg.get('ee_link', 'link_6')
        self.camera_topic = self.cfg.get('image_topic', '/camera/camera/color/image_raw')
        self.marker_length = float(self.cfg.get('aruco_marker_side_length', 0.20)) # 14.5cm
        
        # ArUco 딕셔너리 동적 매핑
        dict_name = self.cfg.get('aruco_dictionary_name', 'DICT_4X4_1000')
        self.aruco_dict = self.get_aruco_dictionary(dict_name)

        # 3. 카메라 파라미터 및 핸드아이 결과 로드 (ROS2 워크스페이스 루트 기준 경로라고 가정)
        # 작업 디렉토리에 맞춰 상대경로를 조정해야 할 수 있습니다.
        self.camera_matrix, self.dist_coeffs = self.load_camera_info(self.cfg.get('camera_calibration_parameters_filename'))
        self.T_ee_cam = self.load_handeye_result(self.cfg.get('handeye_result_file_name'))
        Rot=self.T_ee_cam[:3,:3]
        T=self.T_ee_cam[:3,3]
        quat=R.from_matrix(Rot).as_quat()
        self.tf_broadcaster=tf2_ros.StaticTransformBroadcaster(self)
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'link_6'
        tf_msg.child_frame_id = 'camera_link'
        tf_msg.transform.translation.x = float(T[0])
        tf_msg.transform.translation.y = float(T[1])
        tf_msg.transform.translation.z = float(T[2])
        tf_msg.transform.rotation.x = float(quat[0])
        tf_msg.transform.rotation.y = float(quat[1])
        tf_msg.transform.rotation.z = float(quat[2])
        tf_msg.transform.rotation.w = float(quat[3])
        self.tf_broadcaster.sendTransform(tf_msg)

        # ROS 2 셋업
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 검증 결과 저장용 리스트
        self.calculated_marker_positions = []

    def get_aruco_dictionary(self, dict_name):
        """문자열 이름으로 OpenCV ArUco Dictionary 가져오기"""
        aruco_dict_mapping = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        }
        if dict_name in aruco_dict_mapping:
            return cv2.aruco.getPredefinedDictionary(aruco_dict_mapping[dict_name])
        else:
            self.get_logger().warn(f"알 수 없는 딕셔너리 {dict_name}, 기본값 DICT_4X4_1000 사용")
            return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

    def load_camera_info(self, filepath):
        """realsense_info.yaml 등에서 카메라 내부 파라미터 로드"""
        # OpenCV의 FileStorage로 읽기
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode('K').mat()
        dist_coeffs = fs.getNode('D').mat()
        fs.release()
        self.get_logger().info("카메라 캘리브레이션 파라미터 로드 완료")
        return camera_matrix, dist_coeffs

    def load_handeye_result(self, filepath):
        with open(filepath, 'r') as f:
            he_data = yaml.safe_load(f)
        
        R = np.array(he_data['rotation']).reshape(3, 3)
        t = np.array(he_data['translation']).reshape(3, 1)
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        
        self.get_logger().info("Hand-Eye 캘리브레이션 파라미터 로드 완료")
        return T

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def get_current_tcp_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, self.tcp_frame, rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0)
            )
            q = trans.transform.rotation
            t = trans.transform.translation
            
            R = np.array([
                [1 - 2*(q.y**2 + q.z**2), 2*(q.x*q.y - q.z*q.w), 2*(q.x*q.z + q.y*q.w)],
                [2*(q.x*q.y + q.z*q.w), 1 - 2*(q.x**2 + q.z**2), 2*(q.y*q.z - q.x*q.w)],
                [2*(q.x*q.z - q.y*q.w), 2*(q.y*q.z + q.x*q.w), 1 - 2*(q.x**2 + q.y**2)]
            ])
            T_base_ee = np.eye(4)
            T_base_ee[:3, :3] = R
            T_base_ee[:3, 3] = [t.x, t.y, t.z]
            return T_base_ee
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f"TF Lookup Error: {e}")
            return None

    def get_marker_pose_from_image(self):
        if self.latest_image is None:
            return None

        # 1. 파라미터 생성 (구버전 방식)
        parameters = cv2.aruco.DetectorParameters_create()
        
        gray = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2GRAY)
        
        # 2. ArucoDetector 클래스를 쓰지 않고, 모듈 내 함수를 직접 호출 (구버전 방식)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=parameters
        )

        # ids가 None이 아니고, 하나 이상의 마커가 감지되었을 때만 처리
        if ids is not None and len(ids) > 0:
            # 마커 코너의 3D 좌표 (Z=0)
            obj_points = np.array([[-self.marker_length/2, self.marker_length/2, 0],
                                   [self.marker_length/2, self.marker_length/2, 0],
                                   [self.marker_length/2, -self.marker_length/2, 0],
                                   [-self.marker_length/2, -self.marker_length/2, 0]], dtype=np.float32)
            
            # 자세 추정 (첫 번째로 감지된 마커 기준)
            success, rvec, tvec = cv2.solvePnP(obj_points, corners[0][0], self.camera_matrix, self.dist_coeffs)
            
            if success:
                # 회전 벡터를 회전 행렬로 변환
                R_cam_marker, _ = cv2.Rodrigues(rvec)
                
                # 4x4 동차 변환 행렬(Homogeneous Transformation Matrix) 생성
                T_cam_marker = np.eye(4)
                T_cam_marker[:3, :3] = R_cam_marker
                T_cam_marker[:3, 3] = tvec.flatten()
                
                return T_cam_marker
            
        return None

    def run_auto_verify(self, marker_xyz):
        self.get_logger().info("자동 캘리브레이션 검증을 시작합니다...")
        target = np.array(marker_xyz)
        camera_offset = np.array([-0.08, -0.01, 0.04])
        tilts = [20, 40]
        pans = [0, 120, 240]
        dist = 0.35

        from scipy.spatial.transform import Rotation

        for t in tilts:
            for p in pans:
                rclpy.spin_once(self, timeout_sec=0.1)
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
                rot = Rotation.from_quat([q_obj.x, q_obj.y, q_obj.z, q_obj.w])
                gripper_pos = np.array(cam_pos) - rot.apply(camera_offset)

                q_list = [q_obj.x, q_obj.y, q_obj.z, q_obj.w]
                cp = gripper_pos.tolist()

                if self.move_cartesian(cp, q_list):
                    time.sleep(1.0)
                    rclpy.spin_once(self, timeout_sec=0.5)
                    T_base_ee = self.get_current_tcp_pose()
                    T_cam_marker = self.get_marker_pose_from_image()
                    if T_base_ee is not None and T_cam_marker is not None:
                        T_base_marker = T_base_ee @ self.T_ee_cam @ T_cam_marker
                        marker_base_xyz = T_base_marker[:3, 3]
                        self.calculated_marker_positions.append(marker_base_xyz)
                        self.get_logger().info(
                            f"계산된 마커 위치: X={marker_base_xyz[0]:.4f}, "
                            f"Y={marker_base_xyz[1]:.4f}, Z={marker_base_xyz[2]:.4f}"
                        )
                    else:
                        self.get_logger().warn("마커 검출 실패 또는 TF 수신 실패로 스킵합니다.")
                else:
                    self.get_logger().warn(
                        f"Move failed: tilt={t}, pan={p} — skipping"
                    )

        self.evaluate_calibration()

    def evaluate_calibration(self):
        if not self.calculated_marker_positions:
            self.get_logger().error("수집된 마커 데이터가 없습니다.")
            return

        positions = np.array(self.calculated_marker_positions)
        mean_pos = np.mean(positions, axis=0)
        std_dev = np.std(positions, axis=0)
        distances = np.linalg.norm(positions - mean_pos, axis=1)
        rmse = np.sqrt(np.mean(distances**2))

        print("\n" + "="*50)
        print("🎯 핸드아이 캘리브레이션 검증 결과 🎯")
        print("="*50)
        print(f"Base 기준 마커 평균 위치 (m): X={mean_pos[0]:.4f}, Y={mean_pos[1]:.4f}, Z={mean_pos[2]:.4f}")
        print(f"축별 표준편차 (m): X={std_dev[0]:.4f}, Y={std_dev[1]:.4f}, Z={std_dev[2]:.4f}")
        print(f"전체 평균 제곱근 오차 (RMSE): {rmse:.4f} m ({rmse*1000:.2f} mm)")
        print("="*50)

def main(args=None):
    rclpy.init(args=args)
    verifier = AutoHandEyeVerifier()
    
    # 아루코 마커가 놓여있는 실제 공간의 대략적인 위치 (예: Base 기준 X 0.5m, Y 0.0m, Z 0.0m)
    verifier.run_auto_verify(marker_xyz=[-0.699, -0.067, 0.1]) 
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()