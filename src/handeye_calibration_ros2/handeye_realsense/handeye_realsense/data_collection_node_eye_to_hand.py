"""
Unified Hand-Eye Calibration Node (ArUco + Robot Transform)
- PySpin (Spinnaker SDK)으로 Grasshopper3 직접 제어
- ROS2 Image topic 구독 제거 → PySpin grab loop
- config: eye_to_hand_config.yaml
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from scipy.spatial.transform import Rotation as R_scipy
from std_msgs.msg import String

import PySpin
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
        self.config_path = '/home/minsoo/bin_picking/src/handeye_calibration_ros2/eye_to_hand_spinnaker/config.yaml'
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)

        aruco_dictionary_name = config["aruco_dictionary_name"]
        self.aruco_marker_name = config["aruco_marker_name"]
        self.aruco_marker_side_length = float(config["aruco_marker_side_length"])
        self.camera_calibration_parameters_filename = config["camera_calibration_parameters_filename"]
        self.calculated_camera_optical_frame_name = config["calculated_camera_optical_frame_name"]
        self.marker_data_file_name = config["marker_data_file_name"]
        self.robot_data_file_name = config["robot_data_file_name"]
        self.image_filename = config["image_filename"]
        self.base_link = config["base_link"]
        self.ee_link = config["ee_link"]

        # PySpin 관련 config (optional, 없으면 기본값)
        self.pyspin_exposure_auto = config.get("pyspin_exposure_auto", True)
        self.pyspin_exposure_us = config.get("pyspin_exposure_us", 20000)
        self.pyspin_gain_auto = config.get("pyspin_gain_auto", True)
        self.pyspin_gain_db = config.get("pyspin_gain_db", 10.0)
        self.pyspin_fps = config.get("pyspin_fps", 15.0)

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

        # ── PySpin 카메라 초기화 ──
        self.cam = None
        self.spin_system = None
        self._init_pyspin()

        # ── TF2 Buffer & Listener ──
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribers / Publishers ──
        self.subscription_keypress = self.create_subscription(String, 'keypress_topic', self.keypress_callback, 10)
        self.keypress_publisher = self.create_publisher(String, 'keypress_topic', 10)

        # ── 상태 변수 ──
        self.pose_count = 0
        self.tfbroadcaster = TransformBroadcaster(self)
        self.latest_frame = None
        self.latest_rvecs = None
        self.latest_tvecs = None

        # ── 실행 시 기존 YAML 초기화 ──
        self._init_yaml(self.marker_data_file_name)
        self._init_yaml(self.robot_data_file_name)

        self.get_logger().info("=== Unified Calibration Node (PySpin) Started ===")
        self.get_logger().info(f"  Config       : {self.config_path}")
        self.get_logger().info(f"  Marker YAML  : {self.marker_data_file_name}  (초기화됨)")
        self.get_logger().info(f"  Robot  YAML  : {self.robot_data_file_name}  (초기화됨)")
        self.get_logger().info(f"  TF Lookup    : {self.base_link} -> {self.ee_link}")
        self.get_logger().info("  'r' = save | 'q' = quit")

        # ── Timer로 프레임 루프 (ROS Image topic 대신) ──
        timer_period = 1.0 / self.pyspin_fps
        self.timer = self.create_timer(timer_period, self.grab_and_process)

    # ─────────────────────── PySpin 초기화 ───────────────────────
    def _init_pyspin(self):
        """Spinnaker SDK로 카메라 초기화"""
        self.spin_system = PySpin.System.GetInstance()
        cam_list = self.spin_system.GetCameras()

        if cam_list.GetSize() == 0:
            self.get_logger().fatal("PySpin: 카메라를 찾을 수 없습니다!")
            self.get_logger().fatal("  1) USB3 포트 확인")
            self.get_logger().fatal("  2) sudo chmod 666 /dev/bus/usb/XXX/YYY")
            self.get_logger().fatal("  3) usbfs_memory_mb >= 1000 확인")
            cam_list.Clear()
            self.spin_system.ReleaseInstance()
            raise RuntimeError("No camera found via PySpin")

        self.cam = cam_list[0]
        self.cam.Init()

        # 카메라 정보 출력
        nodemap_tldev = self.cam.GetTLDeviceNodeMap()
        model = PySpin.CStringPtr(nodemap_tldev.GetNode('DeviceModelName'))
        serial = PySpin.CStringPtr(nodemap_tldev.GetNode('DeviceSerialNumber'))
        self.get_logger().info(f"  Camera       : {model.GetValue()} (S/N: {serial.GetValue()})")

        nodemap = self.cam.GetNodeMap()

        # ── Acquisition Mode: Continuous ──
        node_acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        node_acq_continuous = node_acq_mode.GetEntryByName('Continuous')
        node_acq_mode.SetIntValue(node_acq_continuous.GetValue())

        # ── Exposure ──
        try:
            node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
            if self.pyspin_exposure_auto:
                node_exposure_auto.SetIntValue(
                    node_exposure_auto.GetEntryByName('Continuous').GetValue()
                )
                self.get_logger().info("  Exposure     : Auto")
            else:
                node_exposure_auto.SetIntValue(
                    node_exposure_auto.GetEntryByName('Off').GetValue()
                )
                node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
                exposure_clamped = max(
                    node_exposure_time.GetMin(),
                    min(self.pyspin_exposure_us, node_exposure_time.GetMax())
                )
                node_exposure_time.SetValue(exposure_clamped)
                self.get_logger().info(f"  Exposure     : {exposure_clamped:.0f} us (manual)")
        except PySpin.SpinnakerException as e:
            self.get_logger().warn(f"  Exposure 설정 실패: {e}")

        # ── Gain ──
        try:
            node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
            if self.pyspin_gain_auto:
                node_gain_auto.SetIntValue(
                    node_gain_auto.GetEntryByName('Continuous').GetValue()
                )
                self.get_logger().info("  Gain         : Auto")
            else:
                node_gain_auto.SetIntValue(
                    node_gain_auto.GetEntryByName('Off').GetValue()
                )
                node_gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
                gain_clamped = max(
                    node_gain.GetMin(),
                    min(self.pyspin_gain_db, node_gain.GetMax())
                )
                node_gain.SetValue(gain_clamped)
                self.get_logger().info(f"  Gain         : {gain_clamped:.1f} dB (manual)")
        except PySpin.SpinnakerException as e:
            self.get_logger().warn(f"  Gain 설정 실패: {e}")

        # ── Pixel Format: BGR8 (OpenCV 호환) ──
        try:
            node_pixel_fmt = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
            entry_bgr8 = node_pixel_fmt.GetEntryByName('BGR8')
            if entry_bgr8 is not None and PySpin.IsAvailable(entry_bgr8):
                node_pixel_fmt.SetIntValue(entry_bgr8.GetValue())
                self.get_logger().info("  Pixel Format : BGR8")
            else:
                # BGR8 없으면 BayerRG8 후 소프트웨어 디베이어링
                entry_bayer = node_pixel_fmt.GetEntryByName('BayerRG8')
                if entry_bayer is not None and PySpin.IsAvailable(entry_bayer):
                    node_pixel_fmt.SetIntValue(entry_bayer.GetValue())
                    self.get_logger().info("  Pixel Format : BayerRG8 (SW debayer)")
        except PySpin.SpinnakerException as e:
            self.get_logger().warn(f"  PixelFormat 설정 실패: {e}")

        self.cam.BeginAcquisition()
        self.get_logger().info("  Acquisition  : Started")

        # cam_list 참조 해제 (카메라는 self.cam에 유지)
        cam_list.Clear()

    def _grab_frame(self):
        """PySpin에서 한 프레임 가져오기 → BGR numpy array 반환"""
        try:
            image_result = self.cam.GetNextImage(1000)  # 1초 타임아웃
            if image_result.IsIncomplete():
                self.get_logger().warn(
                    f"Incomplete image: {PySpin.Image_GetImageStatusDescription(image_result.GetImageStatus())}"
                )
                image_result.Release()
                return None

            # BGR 변환
            pixel_fmt = image_result.GetPixelFormat()
            if pixel_fmt == PySpin.PixelFormat_BGR8:
                frame = image_result.GetNDArray().copy()
            else:
                # BayerRG8 등 → BGR 변환
                raw = image_result.GetNDArray().copy()
                frame = cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR)

            image_result.Release()
            return frame

        except PySpin.SpinnakerException as e:
            self.get_logger().error(f"PySpin grab error: {e}")
            return None

    # ─────────────────────── YAML 초기화 ───────────────────────
    @staticmethod
    def _init_yaml(path):
        with open(path, 'w') as f:
            yaml.dump({'poses': []}, f, default_flow_style=False)

    # ─────────────────────── 메인 루프 (Timer) ───────────────────────
    def grab_and_process(self):
        """PySpin에서 프레임 grab → ArUco 검출 → TF broadcast"""
        current_frame = self._grab_frame()
        if current_frame is None:
            return

        corners, marker_ids, _ = cv2.aruco.detectMarkers(
            current_frame, self.this_aruco_dictionary, parameters=self.this_aruco_parameters
        )
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(current_frame, corners, marker_ids)

            # 서브픽셀 보정
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = []
            for c in corners:
                refined = cv2.cornerSubPix(
                    gray, c.reshape(-1, 1, 2), (5, 5), (-1, -1), criteria
                )
                corners_refined.append(refined.reshape(1, 4, 2))
            corners = tuple(corners_refined)

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
            self._cleanup()
            rclpy.shutdown()

    def _save_both(self):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

        if self.latest_rvecs is None or self.latest_frame is None:
            self.get_logger().warn(f"[{now_str}] Save failed: ArUco marker not detected!")
            return

        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_link, self.ee_link, rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as ex:
            self.get_logger().error(f"[{now_str}] Save failed: TF lookup error — {ex}")
            return

        self.pose_count += 1

        marker_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self._append_marker_data(self.latest_rvecs, self.latest_tvecs, marker_time)

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

        image_filename = self.image_filename.format(pose_count=self.pose_count)
        cv2.imwrite(image_filename, self.latest_frame)

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

        with open(self.robot_data_file_name, 'r') as f:
            saved = yaml.safe_load(f)
        last_pose = saved['poses'][-1]
        print(f"  [DEBUG] Robot YAML 검증 (Pose {self.pose_count}/{len(saved['poses'])}개 저장됨)")
        print(f"    Saved at: {last_pose['saved_at']}")

    # ─────────────────────── 정리 ───────────────────────
    def _cleanup(self):
        """PySpin 카메라 및 OpenCV 리소스 정리"""
        cv2.destroyAllWindows()
        if self.cam is not None:
            try:
                self.cam.EndAcquisition()
                self.cam.DeInit()
                self.get_logger().info("PySpin camera released")
            except PySpin.SpinnakerException:
                pass
            self.cam = None
        if self.spin_system is not None:
            self.spin_system.ReleaseInstance()
            self.spin_system = None

    def __del__(self):
        self._cleanup()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = UnifiedCalibrationNode()
    except RuntimeError as e:
        print(f"[FATAL] {e}")
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt — shutting down")
    finally:
        node._cleanup()
        node.destroy_node()


if __name__ == '__main__':
    main()