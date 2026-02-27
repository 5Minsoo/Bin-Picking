"""
Hand-Eye Calibration 실제 적용 테스트 노드 (Eye-to-Hand, PySpin + MoveIt2)

작동 원리:
  1. 카메라로 마커 인식 -> T_m2c (카메라 기준 마커 좌표)
  2. 캘리브레이션 행렬 적용 -> T_m2b = X * T_m2c (로봇 베이스 기준 마커 좌표)
  3. 안전 오프셋 추가 -> T_target = T_m2b + [x_off, y_off, z_off]
  4. MoveItMoveHelper의 move_cartesian()을 통해 실제 로봇 이동
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml
import cv2
import time
import PySpin

import tf2_ros  # ← TF 브로드캐스트용

# 사용자의 MoveIt 헬퍼 클래스 임포트
from bin_picking.moveit_helper_functions import MoveItMoveHelper

# ArUco dictionary 매핑
ARUCO_DICT_MAP = {
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
}


class MoveToMarkerNode(Node):
    def __init__(self):
        super().__init__('move_to_marker_node')
        self.get_logger().info("=== Hand-Eye 적용 및 로봇 이동 테스트 ===")

        # ── config.yaml 로드 ──
        config_path = 'src/handeye_calibration_ros2/eye_to_hand_spinnaker/config.yaml'
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"Config 파일 로드 실패: {e}")
            return

        self.base_frame = config["base_link"]
        self.marker_length = config["aruco_marker_side_length"]
        camera_calib_file = config["camera_calibration_parameters_filename"]
        handeye_result_file = config["handeye_result_file_name"]
        aruco_dict_name = config["aruco_dictionary_name"]
        config_path = 'src/handeye_calibration_ros2/eye_to_hand_spinnaker/config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.handeye_result_file_name = config["handeye_result_file_name"]
        with open(self.handeye_result_file_name, 'r') as file:
            config=yaml.safe_load(file)
        self.camera_trans=np.array(config['translation'])
        self.camera_rotation=np.array(config['rotation']).reshape(3,3)
        quat=R.from_matrix(self.camera_rotation).as_quat()
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
        # [중요] 타겟 위치 오프셋 (단위: 미터)
        self.target_offset = 0.35   

        # ── MoveIt Helper 초기화 ──
        self.get_logger().info("MoveIt 제어기 연동 중...")
        self.move_helper = MoveItMoveHelper()

        # ── ArUco ──
        if aruco_dict_name not in ARUCO_DICT_MAP:
            self.get_logger().error(f"지원하지 않는 ArUco dictionary: {aruco_dict_name}")
            return

        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[aruco_dict_name])
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT_MAP[aruco_dict_name])
            self.aruco_params = cv2.aruco.DetectorParameters_create()

        # ── 캘리브레이션 결과 X = T_cam2base ──
        self.X = self._load_calibration(handeye_result_file)
        if self.X is None:
            return

        # ── 카메라 내부 파라미터 ──
        self.camera_matrix, self.dist_coeffs = self._load_camera_info(camera_calib_file)

        # ── PySpin ──
        self.system = None
        self.cam = None
        if not self._init_spinnaker():
            return

        # ── TF 브로드캐스터 ──
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ── 실시간 마커 감지 타이머 (10Hz) ──
        self.last_T_m2b = None  # 마지막으로 검출된 마커 pose (base 기준)
        self.marker_detected = False
        self.detection_timer = self.create_timer(0.1, self._detection_timer_callback)

        # ── 퍼블리셔 & 서비스 ──
        self.cb_group = ReentrantCallbackGroup()
        self.target_pose_pub = self.create_publisher(PoseStamped, 'target_marker_pose', 10)
        self.move_srv = self.create_service(
            Trigger, 'move_to_marker', self.move_callback, callback_group=self.cb_group)

        self.get_logger().info("준비 완료! RViz에서 TF 실시간 확인 가능")
        self.get_logger().info("이동 명령: ros2 service call /move_to_marker std_srvs/srv/Trigger {}")

    # ═══════════════════════ PySpin ═══════════════════════
    def _init_spinnaker(self):
        try:
            self.system = PySpin.System.GetInstance()
            cam_list = self.system.GetCameras()
            if cam_list.GetSize() == 0:
                self.get_logger().error("Spinnaker 카메라를 찾을 수 없습니다.")
                cam_list.Clear()
                self.system.ReleaseInstance()
                self.system = None
                return False
            self.cam = cam_list[0]
            self.cam.Init()
            self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            self.cam.BeginAcquisition()
            cam_list.Clear()
            return True
        except PySpin.SpinnakerException as e:
            self.get_logger().error(f"Spinnaker 초기화 실패: {e}")
            return False

    def _release_spinnaker(self):
        try:
            if self.cam is not None:
                if self.cam.IsStreaming():
                    self.cam.EndAcquisition()
                self.cam.DeInit()
                del self.cam
                self.cam = None
            if self.system is not None:
                self.system.ReleaseInstance()
                self.system = None
        except PySpin.SpinnakerException:
            pass

    def destroy_node(self):
        self._release_spinnaker()
        super().destroy_node()

    def _grab_frame(self):
        try:
            image_result = self.cam.GetNextImage(1000)
            if image_result.IsIncomplete():
                image_result.Release()
                return None
            pixel_fmt = image_result.GetPixelFormat()
            if pixel_fmt == PySpin.PixelFormat_BGR8:
                frame = image_result.GetNDArray().copy()
            else:
                raw = image_result.GetNDArray().copy()
                frame = cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR)
            image_result.Release()
            return frame
        except PySpin.SpinnakerException:
            return None

    # ═══════════════════════ 실시간 마커 감지 & TF 브로드캐스트 ═══════════════════════
    def _detection_timer_callback(self):
        """10Hz로 프레임 캡처 → 마커 검출 → TF 브로드캐스트"""
        frame = self._grab_frame()
        if frame is None:
            self.marker_detected = False
            return

        T_c2m = self._detect_marker(frame)
        if T_c2m is None:
            self.marker_detected = False
            return

        # Base 기준으로 변환
        T_b2m = self.X @ T_c2m
        self.last_T_m2b = T_b2m
        self.marker_detected = True

        now = self.get_clock().now().to_msg()

        # ── TF 1: 마커 원본 위치 (base → marker) ──
        self._publish_tf(T_b2m, self.base_frame, "aruco_marker", now)

        # ── TF 2: 오프셋 적용된 타겟 위치 (base → target) ──
        T_target = T_b2m.copy()
        T_target[0:3, 3] += T_b2m[:3,2]*self.target_offset
        self._publish_tf(T_target, self.base_frame, "grasp_target", now)

        # PoseStamped도 퍼블리시 (RViz Marker 등에서 활용)
        pose_msg = self._matrix_to_pose_stamped(T_target, self.base_frame)
        self.target_pose_pub.publish(pose_msg)

    def _publish_tf(self, T: np.ndarray, parent_frame: str, child_frame: str, stamp):
        """4x4 변환행렬을 TF로 브로드캐스트"""
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        t.transform.translation.x = float(T[0, 3])
        t.transform.translation.y = float(T[1, 3])
        t.transform.translation.z = float(T[2, 3])

        q = R.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])

        self.tf_broadcaster.sendTransform(t)

        # ═══════════════════════ 서비스 콜백 및 이동 로직 ═══════════════════════
    def move_callback(self, request, response):
        self.get_logger().info(f"\n{'='*50}")

        if not self.marker_detected or self.last_T_m2b is None:
            self.get_logger().info("1. 마커 미검출 상태 → 새로 캡처 시도...")
            frame = self._grab_frame()
            if frame is None:
                response.success, response.message = False, "프레임 캡처 실패"
                return response
            T_m2c = self._detect_marker(frame)
            if T_m2c is None:
                response.success, response.message = False, "마커 검출 실패"
                return response
            self.last_T_m2b = self.X @ T_m2c
        else:
            self.get_logger().info("1. 실시간 검출된 마커 사용")

        self.get_logger().info("2. 좌표 변환 및 오프셋 적용 중...")
        T_m2b = self.last_T_m2b

        # ★ 수정: 타이머와 동일하게 마커 Z축 방향으로 오프셋
        T_target = T_m2b.copy()
        T_target[0:3, 3] += T_m2b[:3, 2] * self.target_offset

        # 디버그 로그 추가
        self.get_logger().info(f"   마커 위치(base): [{T_m2b[0,3]:.3f}, {T_m2b[1,3]:.3f}, {T_m2b[2,3]:.3f}]")
        self.get_logger().info(f"   마커 Z축(base):  [{T_m2b[0,2]:.3f}, {T_m2b[1,2]:.3f}, {T_m2b[2,2]:.3f}]")
        self.get_logger().info(f"   타겟 위치:       [{T_target[0,3]:.3f}, {T_target[1,3]:.3f}, {T_target[2,3]:.3f}]")

        pose_msg = self._matrix_to_pose_stamped(T_target, self.base_frame)
        self.target_pose_pub.publish(pose_msg)

        self.get_logger().info("3. MoveIt을 통한 카테시안 이동 시작...")
        success = self._send_robot_move_command(pose_msg)

        if success:
            self.get_logger().info("✅ 로봇 이동 성공!")
            response.success, response.message = True, "이동 성공"
        else:
            self.get_logger().error("❌ 로봇 이동 실패")
            response.success, response.message = False, "이동 실패"

        return response


    def _send_robot_move_command(self, pose_msg: PoseStamped) -> bool:
        xyz = [
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z
        ]

        o = pose_msg.pose.orientation
        rot = R.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
        approach_vec = rot[:, 2]
        obj_axis_x = rot[:, 0]

        q = self.move_helper.make_grasp_quat_for_approach(approach_vec, obj_axis_x)
        quat = [q.x, q.y, q.z, q.w]

        self.get_logger().info(f"   -> Target XYZ:  [{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}]")
        self.get_logger().info(f"   -> Target Quat: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")

        # ★ 수정: 부호 반전 제거
        success = self.move_helper.move_cartesian(
            item=xyz,
            quat=quat,
            collision=True
        )

        return success

    # ═══════════════════════ 마커 검출 ═══════════════════════
    def _detect_marker(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) == 0:
            return None
        if len(ids) > 1:
            self.get_logger().warn("여러 마커 검출됨. 첫 번째 사용.")

        obj_points = np.array([
            [-self.marker_length / 2,  self.marker_length / 2, 0],
            [ self.marker_length / 2,  self.marker_length / 2, 0],
            [ self.marker_length / 2, -self.marker_length / 2, 0],
            [-self.marker_length / 2, -self.marker_length / 2, 0],
        ], dtype=np.float64)

        img_points = corners[0].reshape(4, 2).astype(np.float64)
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)

        if not success:
            return None

        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.flatten()
        return T

    # ═══════════════════════ 유틸 함수 ═══════════════════════
    def _matrix_to_pose_stamped(self, T, frame_id):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.position.x = T[0, 3]
        msg.pose.position.y = T[1, 3]
        msg.pose.position.z = T[2, 3]
        quat = R.from_matrix(T[:3, :3]).as_quat()
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        return msg

    def _load_calibration(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            T = np.eye(4)
            T[:3, :3] = np.array(data['rotation'], dtype=np.float64).reshape(3, 3)
            T[:3, 3] = np.array(data['translation'], dtype=np.float64).reshape(3, 1).flatten()
            return T
        except Exception as e:
            self.get_logger().error(f"캘리브레이션 로드 실패: {e}")
            return None

    def _load_camera_info(self, file_path):
        try:
            fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                return np.eye(3), np.zeros(5)

            K_node = fs.getNode("K")
            D_node = fs.getNode("D")
            if K_node.empty():
                K_node = fs.getNode("camera_matrix")
            if D_node.empty():
                D_node = fs.getNode("distortion_coefficients")

            K, D = K_node.mat(), D_node.mat()
            fs.release()
            return K, (np.zeros(5) if D is None else D.flatten())
        except Exception:
            return np.eye(3), np.zeros(5)


def main(args=None):
    rclpy.init(args=args)
    node = MoveToMarkerNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()