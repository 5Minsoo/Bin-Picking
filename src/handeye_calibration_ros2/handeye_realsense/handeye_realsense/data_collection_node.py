"""
Copyright © 2024 Shengyang Zhuang. All rights reserved.
Modified: Unified Hand-Eye Calibration Node (ArUco + Robot Transform)
- 두 노드를 하나로 합쳐 동시성 문제 해결
- 저장 시 타임스탬프 기록 (디버그용)
- 실행 시 기존 YAML 초기화 후 저장
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from scipy.spatial.transform import Rotation as R_scipy
from cv_bridge import CvBridge
from std_msgs.msg import String

import cv2
import numpy as np
import yaml
from datetime import datetime

# ArUco dictionary lookup
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class UnifiedCalibrationNode(Node):
    def __init__(self):
        super().__init__('unified_calibration_node')

        # ── Config ──
        with open('src/handeye_calibration_ros2/handeye_realsense/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        aruco_dictionary_name = config["aruco_dictionary_name"]
        self.aruco_marker_name = config["aruco_marker_name"]
        self.aruco_marker_side_length = float(config["aruco_marker_side_length"])
        self.camera_calibration_parameters_filename = config["camera_calibration_parameters_filename"]
        self.image_topic = config["image_topic"]
        self.calculated_camera_optical_frame_name = config["calculated_camera_optical_frame_name"]
        self.marker_data_file_name = config["marker_data_file_name"]
        self.robot_data_file_name = config["robot_data_file_name"]
        self.image_filename = config["image_filename"]
        self.base_link = config["base_link"]
        self.ee_link = config["ee_link"]

        # ── ArUco 설정 ──
        if ARUCO_DICT.get(aruco_dictionary_name) is None:
            self.get_logger().error(f"ArUCo tag '{aruco_dictionary_name}' is not supported")
            return

        cv_file = cv2.FileStorage(self.camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ)
        self.mtx = cv_file.getNode('K').mat()
        self.dst = cv_file.getNode('D').mat()
        node_R = cv_file.getNode('R')
        self.R_mtx = node_R.mat() if not node_R.empty() else None
        cv_file.release()

        self.this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dictionary_name])
        self.this_aruco_parameters = cv2.aruco.DetectorParameters_create()

        # ── TF2 Buffer & Listener (로봇 변환 조회용) ──
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribers / Publishers ──
        self.subscription_img = self.create_subscription(Image, self.image_topic, self.listener_callback, 10)
        self.subscription_keypress = self.create_subscription(String, 'keypress_topic', self.keypress_callback, 10)
        self.keypress_publisher = self.create_publisher(String, 'keypress_topic', 10)

        # ── 상태 변수 ──
        self.pose_count = 0
        self.tfbroadcaster = TransformBroadcaster(self)
        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_rvecs = None
        self.latest_tvecs = None

        # ── 실행 시 기존 YAML 초기화 ──
        self._init_yaml(self.marker_data_file_name)
        self._init_yaml(self.robot_data_file_name)
        self.get_logger().info("=== Unified Calibration Node Started ===")
        self.get_logger().info(f"  Marker YAML : {self.marker_data_file_name}  (초기화됨)")
        self.get_logger().info(f"  Robot  YAML : {self.robot_data_file_name}  (초기화됨)")
        self.get_logger().info(f"  TF Lookup   : {self.base_link} -> {self.ee_link}")
        self.get_logger().info("  'r' = save | 'q' = quit")

    # ─────────────────────── YAML 초기화 ───────────────────────
    @staticmethod
    def _init_yaml(path):
        """실행 시 YAML 파일을 빈 상태로 초기화"""
        with open(path, 'w') as f:
            yaml.dump({'poses': []}, f, default_flow_style=False)

    # ─────────────────────── 카메라 콜백 ───────────────────────
    def listener_callback(self, data):
        current_frame = self.bridge.imgmsg_to_cv2(data)
        corners, marker_ids, _ = cv2.aruco.detectMarkers(
            current_frame, self.this_aruco_dictionary, parameters=self.this_aruco_parameters
        )
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(current_frame, corners, marker_ids)

            # 서브픽셀 보정 (포즈 추정 전에)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = []
            for c in corners:
                refined = cv2.cornerSubPix(gray, c.reshape(-1, 1, 2), (5, 5), (-1, -1),
            criteria)
                corners_refined.append(refined.reshape(1, 4, 2))
            corners = tuple(corners_refined)

            # 보정된 코너로 포즈 추정
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.aruco_marker_side_length, self.mtx, self.dst
            )

            self.latest_frame = current_frame.copy()
            self.latest_rvecs = rvecs
            self.latest_tvecs = tvecs


            for i, marker_id in enumerate(marker_ids):
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = self.calculated_camera_optical_frame_name
                t.child_frame_id = self.aruco_marker_name
                t.transform.translation.x = tvecs[i][0][0]
                t.transform.translation.y = tvecs[i][0][1]
                t.transform.translation.z = tvecs[i][0][2]

                rot_mat = cv2.Rodrigues(rvecs[i][0])[0]
                quat = R_scipy.from_matrix(rot_mat).as_quat()
                t.transform.rotation.x = quat[0]
                t.transform.rotation.y = quat[1]
                t.transform.rotation.z = quat[2]
                t.transform.rotation.w = quat[3]

                cv2.drawFrameAxes(current_frame, self.mtx, self.dst, rvecs[i], tvecs[i], 0.05)
                self.tfbroadcaster.sendTransform(t)

        else:
            self.latest_rvecs = None
            self.latest_tvecs = None
        cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("camera", 700, 500)
        cv2.imshow("camera", current_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            self.keypress_publisher.publish(String(data='q'))
        elif key == ord('r'):
            self.keypress_publisher.publish(String(data='r'))

    # ─────────────────────── 저장 트리거 ───────────────────────
    def keypress_callback(self, msg):
        if msg.data == 'r':
            self._save_both()
        elif msg.data == 'q':
            self.get_logger().info("Quit signal received. Shutting down...")
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def _save_both(self):
        """마커 데이터와 로봇 데이터를 동시에(원자적으로) 저장"""
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

        # ── 1) ArUco 마커 확인 ──
        if self.latest_rvecs is None or self.latest_frame is None:
            self.get_logger().warn(f"[{now_str}] Save failed: ArUco marker not detected!")
            return

        # ── 2) 로봇 TF 조회 ──
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_link, self.ee_link, rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as ex:
            self.get_logger().error(f"[{now_str}] Save failed: TF lookup error — {ex}")
            return

        # ── 여기까지 왔으면 양쪽 데이터 모두 유효 → 저장 진행 ──
        self.pose_count += 1

        # --- 마커 데이터 저장 ---
        marker_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self._append_marker_data(self.latest_rvecs, self.latest_tvecs, marker_time)

        # --- 로봇 데이터 저장 ---
        robot_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        translation = [
            trans.transform.translation.x,
            trans.transform.translation.y,
            trans.transform.translation.z,
        ]
        rotation = [
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w,
        ]
        R_gripper2base = R_scipy.from_quat(rotation).as_matrix()
        t_gripper2base = np.array(translation)
        self._append_robot_data(R_gripper2base, t_gripper2base, robot_time)

        # --- 이미지 저장 ---
        image_filename = self.image_filename.format(pose_count=self.pose_count)
        cv2.imwrite(image_filename, self.latest_frame)

        # --- 로그 출력 ---
        self.get_logger().info(
            f"[Pose {self.pose_count}] Saved successfully\n"
            f"  Marker saved at : {marker_time}\n"
            f"  Robot  saved at : {robot_time}\n"
            f"  Image  : {image_filename}"
        )

    # ─────────────────────── YAML 저장 헬퍼 ───────────────────────
    def _append_marker_data(self, rvecs, tvecs, timestamp):
        with open(self.marker_data_file_name, 'r') as f:
            data = yaml.safe_load(f) or {'poses': []}

        for rvec, tvec in zip(rvecs, tvecs):
            R_mat = cv2.Rodrigues(rvec)[0]
            data['poses'].append({
                'rotation': R_mat.tolist(),
                'translation': tvec[0].tolist(),
                'saved_at': timestamp,
            })

        with open(self.marker_data_file_name, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        with open(self.marker_data_file_name, 'r') as f:
            saved = yaml.safe_load(f)
        last_pose = saved['poses'][-1]
        print(f"  [DEBUG] ARUCO YAML 검증 (Pose {self.pose_count}/{len(saved['poses'])}개 저장됨)")
        print(f"    Saved at: {last_pose['saved_at']}")

    def _append_robot_data(self, rotation_matrix, translation_vector, timestamp):
        with open(self.robot_data_file_name, 'r') as f:
            data = yaml.safe_load(f) or {'poses': []}

        data['poses'].append({
            'rotation': rotation_matrix.tolist(),
            'translation': translation_vector.tolist(),
            'saved_at': timestamp,
        })

        with open(self.robot_data_file_name, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        # 디버그: 저장된 YAML 파일을 다시 읽어서 출력
        with open(self.robot_data_file_name, 'r') as f:
            saved = yaml.safe_load(f)
        last_pose = saved['poses'][-1]
        print(f"  [DEBUG] Robot YAML 검증 (Pose {self.pose_count}/{len(saved['poses'])}개 저장됨)")
        print(f"    Saved at: {last_pose['saved_at']}")


def main(args=None):
    rclpy.init(args=args)
    node = UnifiedCalibrationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()