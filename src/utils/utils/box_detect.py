#!/usr/bin/env python3
"""노란색 바구니 내곽 4점 검출 (HSV 외곽 + HSV 내곽)."""

import json
import yaml
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# ── 캘리브레이션 파일 경로 ──
CONFIG_PATH = "src/handeye_calibration_ros2/handeye_realsense/config.yaml"

# ── HSV 범위 (노란색) ──
H_LOW, H_HIGH = 20, 60
S_LOW, S_HIGH = 80, 255
V_LOW, V_HIGH = 80, 255


# ══════════════════════════════════════════════
#  디버그 시각화 (검출 로직과 완전 분리)
# ══════════════════════════════════════════════
def draw_debug(bgr, outer_pts, inner_pts, center, hsv_mask=None):
    """컬러 + HSV 마스크를 나란히 띄운다."""
    h, w = bgr.shape[:2]
    pw, ph = w // 2, h // 2

    # ── 1) 컬러 디버그 (외곽=초록, 내곽=빨강, 중심=분홍) ──
    vis = bgr.copy()
    if outer_pts is not None:
        cv2.polylines(vis, [outer_pts.astype(int)], True, (0, 255, 0), 2)
        for pt in outer_pts.astype(int):
            cv2.circle(vis, tuple(pt), 5, (0, 255, 0), -1)
    if inner_pts is not None:
        cv2.polylines(vis, [inner_pts.astype(int)], True, (0, 0, 255), 2)
        for pt in inner_pts.astype(int):
            cv2.circle(vis, tuple(pt), 5, (0, 0, 255), -1)
    if center is not None:
        cv2.circle(vis, (int(center[0]), int(center[1])), 7, (255, 0, 255), -1)
        cv2.putText(vis, f"({int(center[0])},{int(center[1])})",
                    (int(center[0]) + 10, int(center[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    p1 = cv2.resize(vis, (pw, ph))

    # ── 2) HSV 마스크 + 검출 사각형 ──
    if hsv_mask is not None:
        p2 = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)
        if outer_pts is not None:
            cv2.polylines(p2, [outer_pts.astype(int)], True, (0, 255, 0), 2)
        if inner_pts is not None:
            cv2.polylines(p2, [inner_pts.astype(int)], True, (0, 0, 255), 2)
        p2 = cv2.resize(p2, (pw, ph))
    else:
        p2 = np.zeros((ph, pw, 3), dtype=np.uint8)

    for img, label in [(p1, "Color"), (p2, "HSV Mask")]:
        cv2.putText(img, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    grid = np.hstack((p1, p2))
    cv2.imshow("Basket Debug", grid)
    cv2.waitKey(1)


def compute_center(inner_pts):
    """내곽 4점의 대각선 교점 = 중심점."""
    if inner_pts is None or len(inner_pts) != 4:
        return None
    # TL-BR 대각선과 TR-BL 대각선의 교점
    p1, p2, p3, p4 = inner_pts  # TL, TR, BR, BL
    def line_intersect(a1, a2, b1, b2):
        d1 = a2 - a1
        d2 = b2 - b1
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-8:
            return (a1 + a2 + b1 + b2) / 4  # 평행하면 평균
        t = ((b1[0] - a1[0]) * d2[1] - (b1[1] - a1[1]) * d2[0]) / cross
        return a1 + t * d1
    return line_intersect(p1, p3, p2, p4)


NUM_CAPTURE = 6  # 캡처할 프레임 수


class BasketDetector(Node):
    def __init__(self):
        super().__init__("basket_detector")
        self.bridge = CvBridge()

        # ── 캘리브레이션 로드 ──
        self._load_calibration()

        # depth (좌표 변환 전용)
        self._depth_image = None
        self.create_subscription(Image, "/camera/camera/depth/image_rect_raw", self._cb_depth, 10)

        # 캡처 상태
        self._captured_frames = []  # list of (bgr, depth)
        self._done = False

        # 최종 결과 (평균)
        self._result_bgr = None
        self._result_outer = None
        self._result_inner = None
        self._result_center = None
        self._result_hsv_mask = None

        self.create_subscription(Image, "/camera/camera/color/image_raw", self._cb_color, 10)
        self._pub_bin_info = self.create_publisher(String, "/perception_bridge/bin_info", 10)

    def _load_calibration(self):
        """handeye 결과 + 카메라 intrinsic 로드."""
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        # handeye 결과 (T_cam2ee)
        he_path = config["handeye_result_file_name"]
        with open(he_path, 'r') as f:
            he = yaml.safe_load(f)
        self._R_cam2ee = np.array(he['rotation']).reshape(3, 3)
        self._t_cam2ee = np.array(he['translation']).reshape(3, 1)

        # 카메라 intrinsic
        cam_path = config["camera_calibration_parameters_filename"]
        fs = cv2.FileStorage(cam_path, cv2.FILE_STORAGE_READ)
        K = fs.getNode("K").mat()
        fs.release()
        self._fx = K[0, 0]
        self._fy = K[1, 1]
        self._cx = K[0, 2]
        self._cy = K[1, 2]

        self.get_logger().info(
            f"캘리브레이션 로드 완료: fx={self._fx:.1f}, fy={self._fy:.1f}, "
            f"cx={self._cx:.1f}, cy={self._cy:.1f}")

    def _cb_depth(self, msg):
        self._depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def _cb_color(self, msg):
        if self._done:
            draw_debug(self._result_bgr,
                       self._result_outer, self._result_inner,
                       self._result_center, self._result_hsv_mask)
            return

        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._captured_frames.append(bgr.copy())
        self.get_logger().info(
            f"프레임 캡처 {len(self._captured_frames)}/{NUM_CAPTURE}")

        if len(self._captured_frames) >= NUM_CAPTURE:
            self._process_batch()

    def _process_batch(self):
        self.get_logger().info(f"{NUM_CAPTURE}장 캡처 완료, 일괄 처리 시작...")

        inner_list = []
        outer_list = []
        last_bgr = None
        last_hsv_mask = None

        for bgr in self._captured_frames:
            last_bgr = bgr

            outer, inner, hsv_mask = self._detect_outer_inner(bgr)
            last_hsv_mask = hsv_mask
            if outer is None or inner is None:
                continue

            outer_list.append(self._contour_to_4pts(outer))
            inner_list.append(self._contour_to_4pts(inner))

        if inner_list:
            avg_inner = np.mean(inner_list, axis=0).astype(np.float32)
            avg_outer = np.mean(outer_list, axis=0).astype(np.float32)
            center = compute_center(avg_inner)
            self.get_logger().info(
                f"평균 내곽 4점 (px): {avg_inner.astype(int).tolist()}")
            self.get_logger().info(
                f"유효 프레임: {len(inner_list)}/{NUM_CAPTURE}")
        else:
            avg_inner = avg_outer = center = None
            self.get_logger().warn("유효한 검출 결과 없음!")

        self._result_bgr = last_bgr
        self._result_outer = avg_outer
        self._result_inner = avg_inner
        self._result_center = center
        self._result_hsv_mask = last_hsv_mask
        self._done = True

        if avg_inner is not None and center is not None:
            self._publish_bin_info(avg_inner, center)

        draw_debug(last_bgr, avg_outer, avg_inner, center, last_hsv_mask)

    def _publish_bin_info(self, inner_pts, center):
        """내곽 4점에서 center, size를 계산하여 퍼블리시."""
        # inner_pts: TL(0), TR(1), BR(2), BL(3)
        tl, tr, br, bl = inner_pts
        w = float((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0)
        h = float((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0)
        cx, cy = float(center[0]), float(center[1])

        data = {"center": [cx, cy], "size": [w, h]}
        msg = String()
        msg.data = json.dumps(data)
        self._pub_bin_info.publish(msg)
        self.get_logger().info(f"bin_info 퍼블리시: {msg.data}")

    @staticmethod
    def _contour_to_4pts(contour):
        eps = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)
        if len(approx) == 4:
            pts = approx.reshape(-1, 2).astype(np.float32)
        else:
            pts = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]
        ordered[1] = pts[np.argmin(d)]
        ordered[2] = pts[np.argmax(s)]
        ordered[3] = pts[np.argmax(d)]
        return ordered

    def _detect_outer_inner(self, bgr):
        """HSV 마스크에서 외곽 컨투어를 찾고, erode로 내곽을 구한다."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([H_LOW, S_LOW, V_LOW]),
                           np.array([H_HIGH, S_HIGH, V_HIGH]))

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)

        # 외곽 컨투어
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, None, mask
        min_area = 0.01 * bgr.shape[0] * bgr.shape[1]
        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) < min_area:
            return None, None, mask
        outer = largest

        # 외곽 영역을 erode → 내곽
        outer_filled = np.zeros_like(mask)
        cv2.drawContours(outer_filled, [outer], 0, 255, -1)
        ek = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
        inner_mask = cv2.erode(outer_filled, ek, iterations=1)

        inner_cnts, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not inner_cnts:
            return None, None, mask
        inner = max(inner_cnts, key=cv2.contourArea)
        if cv2.contourArea(inner) < 100:
            return None, None, mask

        return outer, inner, mask


def main(args=None):
    rclpy.init(args=args)
    node = BasketDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
