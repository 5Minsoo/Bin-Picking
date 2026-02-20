import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

import numpy as np
from scipy.spatial.transform import Rotation as R

# 메시지 타입
from geometry_msgs.msg import PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

class GraspVisualizer(Node):
    def __init__(self):
        super().__init__('grasp_visualizer')
        
        qos_profile = QoSProfile(depth=10)

        # 1. 구독 (물체 위치/자세)
        self.sub_poses = self.create_subscription(
            PoseArray, 
            '/perception_bridge/poses', 
            self.poses_callback, 
            qos_profile
        )
        # 2. 발행 (범위 시각화)
        self.scope_pub = self.create_publisher(MarkerArray, '/debug/neighbor_scope', 10)
        # 3. 발행 (그라스핑 마커)
        self.marker_pub = self.create_publisher(MarkerArray, '/debug/grasp_markers', 10)

        self.get_logger().info("Grasp Visualizer Started. Waiting for poses...")
        self.neighbor_radius = 0.02
        self.neighbor_height_threshold = 0.0

    def poses_callback(self, msg: PoseArray):
        """
        포즈 메시지를 받으면 각 물체에 대해 접근 전략을 계산하고 화살표를 그립니다.
        """
        marker_array = MarkerArray()
        
        # 이전 마커 삭제용 메시지 추가
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # 각 물체(Pose)에 대해 루프 (화살표 그리기용)
        for i, pose in enumerate(msg.poses):
            # 쿼터니언 -> 회전행렬 변환
            q_obj = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            rot = R.from_quat(q_obj)
            rot_mat = rot.as_matrix()
            
            center_pos = np.array([pose.position.x, pose.position.y, pose.position.z])

            # -----------------------------------------------------------
            # [사용자 로직] 1. 자세(Orientation) 점수 및 상태 판단
            # -----------------------------------------------------------
            obj_axis_x = rot_mat[:3, 0] # Local X

            # 상태 판단 로직
            z_dot = np.dot(obj_axis_x, np.array([0, 0, 1]))

            if z_dot > 0.8:   # Standing
                self.process_standing(i, center_pos, marker_array)
            elif z_dot < -0.8: # Flipped
                self.process_flipped(i, center_pos, marker_array)
            else:             # Lying
                self.process_lying(i, center_pos, rot_mat, marker_array)

        # 마커 발행 (화살표)
        self.marker_pub.publish(marker_array)
        
        # [수정됨] 위치 리스트가 아닌 pose 리스트 전체를 넘김 (Rotation 계산을 위해)
        self.visualize_neighbor_scope(msg.poses)

    # --------------------------------------------------------------------------
    # [상태별 로직 1] Flipped / Standing
    # --------------------------------------------------------------------------
    def process_flipped(self, idx, center_pos, marker_array):
        approach_vec = np.array([0.0, 0.0, -1.0])
        offset = np.array([0.0, 0.0, 0.1])
        color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
        self.add_arrow(marker_array, idx * 1000 + 1, center_pos + offset, approach_vec, color, scale=0.1, ns="flipped")

    def process_standing(self, idx, center_pos, marker_array):
        approach_vec = np.array([0.0, 0.0, -1.0])
        offset = np.array([0.0, 0.0, 0.142])
        color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        self.add_arrow(marker_array, idx * 1000 + 2, center_pos + offset, approach_vec, color, scale=0.1, ns="standing")

    # --------------------------------------------------------------------------
    # [상태별 로직 2] Lying
    # --------------------------------------------------------------------------
    def process_lying(self, idx, center_pos, rot_mat, marker_array):
        candidates = []
        obj_axis_x = rot_mat[:3, 0]      
        world_z = np.array([0.0, 0.0, -1.0])
        offset = np.array([0.0, 0.0, 0.01])
        
        plane_normal = np.cross(obj_axis_x, world_z)
        norm_val = np.linalg.norm(plane_normal)
        if norm_val < 1e-6:
            plane_normal = np.array([1.0, 0.0, 0.0])
        else:
            plane_normal /= norm_val

        # --- 후보군 생성 루프 ---
        mid_id = 0
        for deg in range(0, 360, 5):
            rad = np.deg2rad(deg)
            mid_id += 1
            unique_id = (idx * 10000) + mid_id 

            # [Case A] Roll
            v_local = np.array([0.0, np.cos(rad), np.sin(rad)], dtype=float)
            v_roll = rot_mat @ v_local
            
            if v_roll[2] >= np.sin(np.deg2rad(25)): 
                score = abs(v_roll[2])
                candidates.append((score, v_roll, deg))
                c_grey = ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.3)
                self.add_arrow(marker_array, unique_id, center_pos + obj_axis_x * 0.013+ offset, v_roll, c_grey, scale=0.1, ns="lying_candidates")

            # [Case B] Pitch
            rot_obj_pitch = R.from_rotvec(plane_normal * rad)
            v_pitch = rot_obj_pitch.apply(obj_axis_x)
            
            if v_pitch[2] >= np.sin(np.deg2rad(25)):
                score = abs(v_pitch[2])
                candidates.append((score, v_pitch, deg + 1000))
                c_cyan = ColorRGBA(r=0.0, g=0.8, b=0.8, a=0.3)
                self.add_arrow(marker_array, unique_id + 5000, center_pos + obj_axis_x * 0.013 + offset, v_pitch, c_cyan, scale=0.1, ns="lying_candidates")

        # --- Best Candidate ---
        candidates.sort(key=lambda x: x[0], reverse=True)

        if candidates:
            best_score, best_vec, best_deg = candidates[0]
            # [시각화] 최종 선정된 벡터 (진한 초록색)
            c_green = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)

    # --------------------------------------------------------------------------
    # [수정됨] 이웃 탐색 범위 시각화 (오프셋 적용)
    # --------------------------------------------------------------------------
    def visualize_neighbor_scope(self, poses):
        """
        poses: List of geometry_msgs/Pose
        각 Pose에서 Rotation을 구해 Local X * 0.022 만큼 이동한 위치를 중심으로 그립니다.
        """
        marker_array = MarkerArray()

        # 1. 기존 마커 초기화
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        search_radius = self.neighbor_radius * 2
        
        for i, pose in enumerate(poses):
            # 1. Rotation 계산
            q_list = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            rot_mat = R.from_quat(q_list).as_matrix()
            obj_axis_x = rot_mat[:3, 0] # Local X 벡터 추출

            # 2. 중심 좌표 계산 (기존 위치 + Local X * 0.022)
            original_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
            offset_dist = 0.022
            
            # ★ 핵심 수정: 오프셋 적용된 중심 좌표 ★
            final_pos = original_pos + (obj_axis_x * offset_dist)
            
            px, py, pz = final_pos[0], final_pos[1], final_pos[2]

            # -----------------------------------------------------
            # [Marker 1] 거리 탐색 범위 (투명한 구)
            # -----------------------------------------------------
            sphere = Marker()
            sphere.header.frame_id = "base_link" 
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = "search_radius"
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            
            sphere.pose.position.x = px
            sphere.pose.position.y = py
            sphere.pose.position.z = pz
            
            sphere.scale.x = search_radius * 2.0
            sphere.scale.y = search_radius * 2.0
            sphere.scale.z = search_radius * 2.0
            
            sphere.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.15)
            marker_array.markers.append(sphere)

            # -----------------------------------------------------
            # [Marker 2] 높이 제한선 (빨간 원반)
            # -----------------------------------------------------
            disk = Marker()
            disk.header.frame_id = "base_link"
            disk.header.stamp = self.get_clock().now().to_msg()
            disk.ns = "height_limit"
            disk.id = i + 10000 
            disk.type = Marker.CYLINDER
            disk.action = Marker.ADD
            
            # 원반 위치: 오프셋 된 중심에서 threshold만큼 아래
            disk.pose.position.x = px
            disk.pose.position.y = py
            disk.pose.position.z = pz - self.neighbor_height_threshold
            
            disk.scale.x = search_radius * 2.0
            disk.scale.y = search_radius * 2.0
            disk.scale.z = 0.005
            
            disk.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)
            marker_array.markers.append(disk)

        # 발행
        self.scope_pub.publish(marker_array)

    # --------------------------------------------------------------------------
    # [Helper] 마커 생성 함수
    # --------------------------------------------------------------------------
    def add_arrow(self, marker_array, m_id, start_pos, vector, color, scale=0.2, ns="default"):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = m_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        p1 = Point()
        p1.x, p1.y, p1.z = float(start_pos[0]), float(start_pos[1]), float(start_pos[2])
        
        p2 = Point()
        p2.x = start_pos[0] + vector[0] * scale
        p2.y = start_pos[1] + vector[1] * scale
        p2.z = start_pos[2] + vector[2] * scale
        
        marker.points = [p1, p2]
        
        marker.scale.x = 0.005 
        marker.scale.y = 0.01  
        marker.scale.z = 0.0   
        
        marker.color = color
        
        marker_array.markers.append(marker)

def main(args=None):
    rclpy.init(args=args)
    node = GraspVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()