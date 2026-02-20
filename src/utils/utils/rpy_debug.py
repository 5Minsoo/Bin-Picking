#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
import json
import math
import numpy as np

# Ferrari & Canny (1992) Metric 구현을 위한 헬퍼 클래스
class GraspQualityScorer:
    def __init__(self, ideal_axis_z=True):
        self.ideal_axis_z = ideal_axis_z

    def calculate_score(self, pose: Pose):
        # 1. Orientation Stability (안정성)
        q = pose.orientation

        # --- [수정됨] 쿼터니언 -> RPY (Roll, Pitch, Yaw) 변환 (ROS 표준: ZYX 순서) ---
        # 수식 출처: Wikipedia (Conversion between quaternions and Euler angles)
        
        # (1) Roll (x-axis rotation)
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # (2) Pitch (y-axis rotation)
        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            # 짐벌 락(Gimbal Lock) 상황 예외처리: 90도 고정
            pitch = math.copysign(math.pi / 2, sinp) 
        else:
            pitch = math.asin(sinp)

        # (3) Yaw (z-axis rotation)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # 라디안 -> 도(Degree) 변환
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
    
        # --- [수정됨] 상태 판단 로직 ---
        # 1. Stand: Roll과 Pitch가 모두 0도 근처 (똑바로 서 있음)
        if abs(roll_deg) < 10 and abs(pitch_deg) < 10:
            status = "Flipped"
            
        # 2. Flipped: Roll이 180도 근처 (뒤집혀 있음)
        # abs(abs(roll) - 180) < 10 : -180도나 +180도 근처일 때
        elif abs(abs(roll_deg) - 180) < 10 and abs(pitch_deg) < 10:
            status = "Standing"
            
        # 3. Lying: 그 외 (누워 있음)
        else:
            status = "Lying" 
            
        vertical_alignment = 1.0 if status == "Stand" else 0.5 if status == "Flipped" else 0.0
        
        # 중심점 보정 로직
        z_offset = 0.04 * vertical_alignment
        final_z = pose.position.z + z_offset
        
        total_score = final_z 
        
        debug_info = {
            'raw_z': pose.position.z,
            'roll': roll_deg,
            'pitch': pitch_deg,
            'status': status,
            'offset': z_offset,
            'final_z': final_z
        }
        
        return total_score, debug_info

class GraspPlanner(Node):
    def __init__(self):
        super().__init__('grasp_planner')

        # === 설정 ===
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        
        # Subscribers
        self.sub_names = self.create_subscription(
            String, '/perception_bridge/names', self.names_callback, qos_profile)
        self.sub_poses = self.create_subscription(
            PoseArray, '/perception_bridge/poses', self.poses_callback, qos_profile)

        # Publishers
        self.pub_target = self.create_publisher(String, '/grasp_planner/target', 10) 
        self.pub_order = self.create_publisher(String, '/grasp_planner/order', 10)   
        self.pub_markers = self.create_publisher(MarkerArray, '/grasp_planner/markers', 10) 

        self.object_names = []
        self.scorer = GraspQualityScorer()
        
        # 출력 속도 조절을 위한 시간 변수 초기화
        self.last_print_time = self.get_clock().now()
        self.print_interval = 0.5  # 0.5초마다 출력
        
        self.get_logger().info("Grasp Planner Initialized (RPY Debug Mode)")

    def names_callback(self, msg):
        try:
            self.object_names = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def poses_callback(self, msg):
        if not self.object_names:
            return
        
        count = min(len(msg.poses), len(self.object_names))
        if count == 0:
            return

        ranked_objects = []

        # 1. 계산 로직
        for i in range(count):
            name = self.object_names[i]
            pose = msg.poses[i]
            score, debug = self.scorer.calculate_score(pose)
            
            ranked_objects.append({
                'name': name,
                'pose': pose,
                'score': score,
                'debug': debug
            })

        # 2. 정렬 및 토픽 발행
        ranked_objects.sort(key=lambda x: x['score'], reverse=True)
        
        target_obj = ranked_objects[0]['name']
        order_list = [obj['name'] for obj in ranked_objects]

        msg_target = String()
        msg_target.data = target_obj
        self.pub_target.publish(msg_target)

        msg_order = String()
        msg_order.data = json.dumps(order_list)
        self.pub_order.publish(msg_order)

        self.publish_rank_markers(ranked_objects)
        
        # 3. 디버그 출력 (시간 제한 적용)
        current_time = self.get_clock().now()
        time_diff = (current_time - self.last_print_time).nanoseconds / 1e9

        if time_diff >= self.print_interval:
            self.print_debug_table(ranked_objects)
            self.last_print_time = current_time

    def print_debug_table(self, ranked_objects):
        """RPY 및 상태를 포함한 상세 디버그 표 출력"""
        print("\n" + "="*85)
        print(f"{'Object Name':<15} | {'Raw Z':<8} | {'Roll':<6} | {'Pitch':<6} | {'Status':<10} | {'Score':<8}")
        print("-" * 85)

        for obj in ranked_objects:
            d = obj['debug']
            print(f"{obj['name']:<15} | {d['raw_z']:.4f}   | {d['roll']:>6.0f} | {d['pitch']:>6.0f} | {d['status']:<10} | {obj['score']:.4f}")

        print("="*85)
        print(f"I> Best Pick: [{ranked_objects[0]['name']}] (Updated every {self.print_interval}s)", flush=True)

    def publish_rank_markers(self, ranked_objects):
        marr = MarkerArray()
        now = self.get_clock().now().to_msg()

        for rank, obj in enumerate(ranked_objects):
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = now
            m.ns = "grasp_rank"
            m.id = rank
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            
            p = obj['pose'].position
            m.pose.position.x = p.x
            m.pose.position.y = p.y
            m.pose.position.z = p.z + 0.15 
            
            m.scale.z = 0.05 
            
            if rank == 0:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 1.0
                m.text = f"TARGET\n{obj['score']:.3f}"
                m.scale.z = 0.07
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, 0.8
                m.text = f"{rank + 1}\n{obj['score']:.3f}"

            marr.markers.append(m)

        self.pub_markers.publish(marr)

def main(args=None):
    rclpy.init(args=args)
    node = GraspPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()