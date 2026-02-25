"""
Hand-Eye Calibration Node (Eye-to-Hand)
- cv2.calibrateHandEye()에 로봇 역변환(gripper2base → base2gripper) 적용
- 여러 method 비교 출력
- 결과: camera ← base 변환
"""
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
import os


# calibrateHandEye method 목록
HANDEYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


class HandEyeCalibrationNode(Node):
    def __init__(self):
        super().__init__('hand_eye_calibration_node')
        self.get_logger().info("=== Hand-Eye Calibration Node (Eye-to-Hand) ===")

        # ── Config ──
        config_path = 'src/handeye_calibration_ros2/eye_to_hand_spinnaker/config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.robot_data_file_name = config["robot_data_file_name"]
        self.marker_data_file_name = config["marker_data_file_name"]
        self.handeye_result_file_name = config["handeye_result_file_name"]
        self.handeye_result_profile_file_name = config["handeye_result_profile_file_name"]

        # ── 데이터 로드 ──
        self.R_gripper2base, self.t_gripper2base = self.load_transformations(self.robot_data_file_name)
        self.R_target2cam, self.t_target2cam = self.load_transformations(self.marker_data_file_name)

        if len(self.R_gripper2base) < 3:
            self.get_logger().error(f"포즈 데이터 부족: {len(self.R_gripper2base)}개 (최소 3개 필요)")
            return

        # ── 캘리브레이션 실행 ──
        self.compute_hand_eye()

    # ─────────────────────── 데이터 로드 ───────────────────────
    def load_transformations(self, file_path):
        self.get_logger().info(f"Loading: {file_path}")

        with open(file_path, 'r') as file:
            try:
                data = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                self.get_logger().error(f"YAML 파싱 에러: {exc}")
                return [], []

        poses = data.get('poses', [])
        self.get_logger().info(f"  → {len(poses)}개 포즈 발견")

        rotations = []
        translations = []

        for i, pose in enumerate(poses):
            if not isinstance(pose, dict):
                self.get_logger().error(f"  [{i}] 딕셔너리가 아님: {type(pose)}")
                continue
            if 'rotation' not in pose or 'translation' not in pose:
                self.get_logger().error(f"  [{i}] rotation/translation 키 누락")
                continue
            try:
                rot = np.array(pose['rotation'], dtype=np.float64)
                trans = np.array(pose['translation'], dtype=np.float64)
                rotations.append(rot)
                translations.append(trans)
            except Exception as e:
                self.get_logger().error(f"  [{i}] 변환 에러: {e}")

        self.get_logger().info(f"  → {len(rotations)}/{len(poses)}개 성공")
        return rotations, translations

    # ─────────────────────── Eye-to-Hand 캘리브레이션 ───────────────────────
    def compute_hand_eye(self):
        n = len(self.R_gripper2base)
        self.get_logger().info(f"캘리브레이션 시작: {n}개 포즈")

        # ── gripper2base → base2gripper 역변환 (Eye-to-Hand 핵심) ──
        #
        #   data_collection에서 저장: T_gripper2base (base_link → link_6)
        #   calibrateHandEye Eye-to-Hand 입력: T_base2gripper (역변환)
        #
        #   T_inv = [R^T  | -R^T * t]
        #           [0    |    1    ]
        #
        R_base2gripper = []
        t_base2gripper = []

        for i in range(n):
            R_g2b = self.R_gripper2base[i].reshape(3, 3)
            t_g2b = self.t_gripper2base[i].reshape(3, 1)

            R_inv = R_g2b.T
            t_inv = -R_inv @ t_g2b

            R_base2gripper.append(R_inv)
            t_base2gripper.append(t_inv)

        # ── 마커 데이터 (target2cam) ── 그대로 사용
        R_t2c = [r.reshape(3, 3) for r in self.R_target2cam]
        t_t2c = [t.reshape(3, 1) for t in self.t_target2cam]

        # ── 모든 method로 계산 & 비교 ──
        print("\n" + "=" * 70)
        print("  Hand-Eye Calibration Results (Eye-to-Hand)")
        print("  출력: T_cam2base (camera ← base)")
        print("=" * 70)

        results = {}
        for name, method in HANDEYE_METHODS.items():
            try:
                R_result, t_result = cv2.calibrateHandEye(
                    R_base2gripper, t_base2gripper,
                    R_t2c, t_t2c,
                    method=method
                )

                # 회전행렬 유효성 검사
                det = np.linalg.det(R_result)
                is_valid = abs(det - 1.0) < 0.01

                quat = R.from_matrix(R_result).as_quat()  # [x, y, z, w]
                euler = R.from_matrix(R_result).as_euler('xyz', degrees=True)

                results[name] = {
                    'R': R_result,
                    't': t_result,
                    'det': det,
                    'valid': is_valid,
                    'quat': quat,
                    'euler': euler,
                }

                valid_str = "OK" if is_valid else "BAD"
                print(f"\n  [{name}] (det={det:.6f}) [{valid_str}]")
                print(f"    Rotation:\n{self._fmt_matrix(R_result, 6)}")
                print(f"    Translation: [{t_result[0][0]:.6f}, {t_result[1][0]:.6f}, {t_result[2][0]:.6f}]")
                print(f"    Euler (xyz°): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]")
                print(f"    Quaternion:   [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")

            except Exception as e:
                self.get_logger().warn(f"  [{name}] 실패: {e}")

        print("\n" + "=" * 70)

        # ── 유효한 결과들의 평균 translation으로 이상치 탐지 ──
        valid_results = {k: v for k, v in results.items() if v['valid']}
        if valid_results:
            ts = np.array([v['t'].flatten() for v in valid_results.values()])
            t_mean = ts.mean(axis=0)
            t_std = ts.std(axis=0)
            print(f"\n  유효 결과 {len(valid_results)}개 Translation 통계:")
            print(f"    Mean: [{t_mean[0]:.6f}, {t_mean[1]:.6f}, {t_mean[2]:.6f}]")
            print(f"    Std:  [{t_std[0]:.6f}, {t_std[1]:.6f}, {t_std[2]:.6f}]")

        BEST="PARK"
        if BEST in results and results[BEST]["valid"]:
            best = results[BEST]
        elif valid_results:
            best_name = list(valid_results.keys())[0]
            best = valid_results[best_name]
            self.get_logger().warn(f"TSAI 실패 → {best_name} 결과 저장")
        else:
            self.get_logger().error("유효한 캘리브레이션 결과 없음!")
            return

        self.save_yaml(best['R'], best['t'])
        self.save_yaml_profile(best['quat'], best['t'])

        self.get_logger().info("캘리브레이션 완료!")

    # ─────────────────────── 저장 ───────────────────────
    def save_yaml(self, rotation, translation):
        """회전행렬 + 이동벡터 저장"""
        new_data = {
            'rotation': rotation.flatten().tolist(),
            'translation': translation.flatten().tolist()
        }
        with open(self.handeye_result_file_name, 'w') as file:
            yaml.safe_dump(new_data, file)

        self.get_logger().info(f"Result saved: {self.handeye_result_file_name}")

    def save_yaml_profile(self, quaternion, translation):
        """쿼터니언 + 이동벡터 프로파일 저장 (누적)"""
        new_data = {
            'rotation': quaternion.tolist(),
            'translation': translation.flatten().tolist()
        }

        existing_data = {'transforms': []}
        if os.path.exists(self.handeye_result_profile_file_name) and \
           os.path.getsize(self.handeye_result_profile_file_name) > 0:
            with open(self.handeye_result_profile_file_name, 'r') as file:
                loaded = yaml.safe_load(file)
                if loaded and 'transforms' in loaded:
                    existing_data = loaded

        existing_data['transforms'].append(new_data)

        with open(self.handeye_result_profile_file_name, 'w') as file:
            yaml.safe_dump(existing_data, file)

        self.get_logger().info(f"Profile saved: {self.handeye_result_profile_file_name}")

    # ─────────────────────── 유틸 ───────────────────────
    @staticmethod
    def _fmt_matrix(m, indent=4):
        prefix = " " * indent
        rows = []
        for row in m:
            vals = ", ".join(f"{v:12.8f}" for v in row.flatten())
            rows.append(f"{prefix}[{vals}]")
        return "\n".join(rows)


def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()