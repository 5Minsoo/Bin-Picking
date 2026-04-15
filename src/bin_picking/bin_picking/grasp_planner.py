#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


class GraspQualityScorer:
    def __init__(self):
        # 가중치 설정
        self.w_height = 1.5       # 높을수록 좋음
        self.w_vertical = 0.1    # 서있을수록 좋음
        self.w_density = -0.005    # 주변에 방해물이 적을수록 좋음 (음수)
        self.w_clearance = 2.0    # 벽에서 멀수록(중앙일수록) 좋음

    def calculate_score(self, pose: Pose, z_val: float, neighbor_count: int, clearance: float):
        # 1. 자세(Orientation) 점수
        q = pose.orientation
        vx_z = 2 * (q.x * q.z - q.y * q.w)
        
        if vx_z < -0.9: status = "Flipped"
        elif vx_z > 0.9: status = "Standing"
        else: status = "Lying"

        vertical_alignment = 1.0 if status == "Standing" else 1.5 if status == "Flipped" else 0.0
        
        # 2. 항목별 점수 계산
        s_height = z_val * self.w_height
        s_vertical = vertical_alignment * self.w_vertical
        s_density = neighbor_count * self.w_density
        
        # [점수] 벽에서 멀수록(중앙) 점수 높음
        s_clearance = clearance * self.w_clearance 
        
        total_score = s_height + s_vertical + s_density + s_clearance
        
        debug_info = {
            'status': status,
            'raw_z': z_val, 'w_z': s_height,
            'raw_n': neighbor_count, 'w_n': s_density,
            'raw_v': vertical_alignment, 'w_v': s_vertical,
            'raw_c': clearance, 'w_c': s_clearance
        }
        return total_score, debug_info

class GraspPlanner(Node):
    def __init__(self):
        super().__init__('grasp_planner')
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        
        # [구독] 물체 이름, 포즈, 그리고 상자 정보(Bin Info)
        self.sub_names = self.create_subscription(String, '/perception_bridge/names', self.names_callback, qos_profile)
        self.sub_poses = self.create_subscription(PoseArray, '/perception_bridge/poses', self.poses_callback, qos_profile)
        self.sub_bin_info = self.create_subscription(String, '/perception_bridge/bin_info', self.bin_info_callback, qos_profile)

        self.pub_target_pose = self.create_publisher(PoseStamped, '/grasp_planner/target_pose', 10)
        self.pub_target_info = self.create_publisher(String, '/grasp_planner/target_info', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/grasp_planner/visual_markers', 10) 

        self.object_names = []
        self.scorer = GraspQualityScorer()
        self.center=[0,0]
        self.size=[0,0]
        self.yaw=0
        # 상자 경계 정보 저장용 변수
        self.bin_bounds = None 
        
        self.neighbor_radius = 0.02         
        self.neighbor_height_threshold = 0.01 
        # self.local_move_vector = np.array([0.02, 0.0, 0.0]) 
        self.local_move_vector = np.array([0.00, 0.0, 0.0]) 

        # 무시할 영역 (World 좌표 기준, 필요시 사용)
        self.ignore_zone = {
            'x_min': 0.1 ,'x_max': 3.0,
            'y_min': 0.0, 'y_max': 0.4
        }
        
        self.last_print_time = self.get_clock().now()
        
        # [추가] 타겟 유지(Persistence)를 위한 변수
        self.last_valid_target = None       # 마지막으로 계산된 Best Object 정보
        self.last_valid_time = None         # 마지막 유효 타겟 시간
        self.persistence_duration = 0.5     # 데이터가 끊겨도 0.5초간은 이전 타겟 유지

    def names_callback(self, msg):
        try:
            data = json.loads(msg.data)

            if isinstance(data, dict):
                # 카메라 모드: {"obj_1": 0.012, "obj_2": 0.034}
                self.object_names = list(data.keys())
                self.errors       = list(data.values())
            elif isinstance(data, list):
                # Isaac Sim 모드: ["obj_1", "obj_2"]
                self.object_names = data

        except json.JSONDecodeError:
            pass

    def bin_info_callback(self, msg):
        """
        /perception_bridge/bin_info 토픽 콜백
        수신 데이터 예시: {"center": [0.0, 0.8], "size": [0.445, 0.45]}
        """
        try:
            data = json.loads(msg.data)
            self.center = data.get("center", [0, 0])
            self.size = data.get("size", [0.5, 0.5]) # 기본값 처리
            self.yaw=data.get("yaw",0)

            cx, cy = self.center[0], self.center[1]
            w, h = self.size[0], self.size[1] # w: x축 크기, h: y축 크기

            # 경계값 미리 계산 (왼쪽, 오른쪽, 아래, 위)
            self.bin_bounds = {
                'left': cx - (w / 2.0),
                'right': cx + (w / 2.0),
                'bottom': cy - (h / 2.0),
                'top': cy + (h / 2.0)
            }
            
        except json.JSONDecodeError:
            self.get_logger().error("Failed to decode Bin Info JSON")
        except Exception as e:
            self.get_logger().error(f"Error processing bin info: {e}")

    def calculate_clearance(self, x, y):
        """
        [종합 점수] = (가장 가까운 벽까지의 거리) - (중심점과의 거리 * 가중치)
        """
        if self.bin_bounds is None:
            return 0.0

        # 1. [Safety] 가장 가까운 벽까지의 거리 (최소값)
        # 이 값이 0에 가까우면(벽에 붙음) 그리퍼가 들어갈 공간이 없으므로 점수가 낮아야 함
        d_left =abs(x - self.bin_bounds['left'])
        d_right = abs(self.bin_bounds['right'] - x)
        d_bottom = abs(y - self.bin_bounds['bottom'])
        d_top = abs(self.bin_bounds['top'] - y)
        
        min_wall_dist = min(d_left, d_right, d_bottom, d_top)

        # 2. [Efficiency] 박스 중심점과의 거리
        # 이 값이 클수록(가장자리) 감점 요인임
        cx = (self.bin_bounds['left'] + self.bin_bounds['right']) / 2.0
        cy = (self.bin_bounds['bottom'] + self.bin_bounds['top']) / 2.0
        dist_from_center = math.sqrt((x - cx)**2 + (y - cy)**2)

        # 3. [합산] 가중치 조절
        # 예: min_wall_dist(안전)는 그대로 두고, 중심 거리는 0.5배 해서 빼기
        # -> 벽에 붙은 건 절대 안 되지만, 중앙에서 좀 벗어난 건 봐주겠다.
        
        centering_weight = 1.0  # 이 값을 키우면 중앙 집착이 심해짐
        
        final_score = min_wall_dist - (dist_from_center * centering_weight)

        return final_score

    def poses_callback(self, msg):
        frame_id = msg.header.frame_id if msg.header.frame_id else "world"
        current_time = self.get_clock().now()
        
        # 1. 데이터 유효성 검사 및 계산
        ranked_objects = []
        
        # 이름과 포즈 데이터가 모두 있을 때만 계산 시도
        if self.object_names and len(msg.poses) > 0:
            count = min(len(msg.poses), len(self.object_names))
            
            moved_positions = [] 
            rotations = []
            
            for i in range(count):
                p = msg.poses[i]
                r = R.from_quat([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
                rotations.append(r)
                world_offset = r.apply(self.local_move_vector)
                moved_positions.append([
                    p.position.x + world_offset[0],
                    p.position.y + world_offset[1],
                    p.position.z + world_offset[2]
                ])

            neighbor_counts = self.count_neighbors(moved_positions)

            for i in range(count):
                obj_name = self.object_names[i]
                if obj_name.startswith("wall_"): continue

                pose = msg.poses[i]
                target_pos = moved_positions[i]
                clearance = self.calculate_clearance(pose.position.x, pose.position.y)
                score, debug = self.scorer.calculate_score(pose, target_pos[2], neighbor_counts[i], clearance)

                # in_x_range = self.ignore_zone['x_min'] < target_pos[0] < self.ignore_zone['x_max']
                # in_y_range = self.ignore_zone['y_min'] < target_pos[1] < self.ignore_zone['y_max']
                # if in_x_range and in_y_range:
                    # score = -999.0
                    # debug['status'] = "IGNORED (Zone)"
                x_dist=abs(pose.position.x - self.center[0])
                y_dist=abs(pose.position.y - self.center[1])
                if x_dist>self.size[0]/2.0+0.02 or y_dist>self.size[1]/2.0+0.02:
                    score = -999.0
                    debug['status'] = "IGNORED (Out of Bin)"
                ranked_objects.append({
                    'name': obj_name, 'pose': pose, 'moved_pos': target_pos,
                    'score': score, 'debug': debug, 'rotation': rotations[i]
                })

            # 점수순 정렬
            if ranked_objects:
                ranked_objects.sort(key=lambda x: x['score'], reverse=True)
                
                # [중요] 최적 타겟 갱신 (데이터가 있을 때 업데이트)
                self.last_valid_target = ranked_objects[0]
                self.last_valid_time = current_time

        # 2. 타겟 발행 로직 (Persistence 적용)
        # 현재 프레임에 데이터가 없더라도(ranked_objects 빈 리스트), 
        # 최근(0.5초 이내)에 찾은 타겟이 있다면 그걸 계속 발행함 -> 깜빡임 방지
        target_to_publish = None
        
        if self.last_valid_target is not None:
            time_diff = (current_time - self.last_valid_time).nanoseconds / 1e9
            if time_diff < self.persistence_duration:
                target_to_publish = self.last_valid_target
            else:
                # 너무 오래된 데이터면 초기화
                self.last_valid_target = None

        # 발행할 타겟이 없으면 종료
        if target_to_publish is None:
            return

        # 3. 실제 발행 (PoseStamped)
        best_obj = target_to_publish
        
        msg_pose = PoseStamped()
        msg_pose.header.frame_id = frame_id
        msg_pose.header.stamp = current_time.to_msg()
        msg_pose.pose.position.x = best_obj['moved_pos'][0]
        msg_pose.pose.position.y = best_obj['moved_pos'][1]
        msg_pose.pose.position.z = best_obj['moved_pos'][2]
        msg_pose.pose.orientation = best_obj['pose'].orientation
        self.pub_target_pose.publish(msg_pose)

        # 4. JSON 발행
        target_json = {
            "target_name": best_obj['name'],
            "status": best_obj['debug']['status'],
            "pose": {
                "position": {"x": best_obj['moved_pos'][0], "y": best_obj['moved_pos'][1], "z": best_obj['moved_pos'][2]},
                "orientation": {"x": best_obj['pose'].orientation.x, "y": best_obj['pose'].orientation.y, 
                                "z": best_obj['pose'].orientation.z, "w": best_obj['pose'].orientation.w}
            }
        }
        self.pub_target_info.publish(String(data=json.dumps(target_json)))
        
        # 5. 시각화 및 디버그
        # 현재 프레임에 새로 계산된 데이터가 있을 때만 테이블 출력 (로그 도배 방지)
        if ranked_objects and (current_time - self.last_print_time).nanoseconds / 1e9 >= 0.5:
            self.print_debug_table(ranked_objects)
            self.last_print_time = current_time

        # 마커는 계속 유지 (깜빡임 방지 버전 호출)
        self.publish_debug_axes(best_obj, frame_id)

    def count_neighbors(self, positions):
        count = len(positions)
        counts = [0] * count
        for i in range(count):
            p_i = positions[i]
            for j in range(count):
                if i == j: continue
                p_j = positions[j]
                dist = math.sqrt((p_i[0]-p_j[0])**2 + (p_i[1]-p_j[1])**2 + (p_i[2]-p_j[2])**2)
                if dist < self.neighbor_radius * 2:
                    if p_j[2] >= p_i[2] - self.neighbor_height_threshold: 
                        counts[i] += 1
        return counts

    def print_debug_table(self, ranked_objects):
        print("\n" + "="*120)
        header = f"{'Rank':<4} | {'Name':<12} | {'Z(W)':<12} | {'Status(W)':<15} | {'Dens(W)':<10} | {'Clearance(W)':<15} | {'Total':<8}"
        print(header)
        print("-" * 120)
        for i, obj in enumerate(ranked_objects[:10]): 
            d = obj['debug']
            z_str = f"{d['w_z']:.2f}"
            status_str = f"{d['status'][:7]}({d['w_v']:.1f})"
            dens_str = f"{d['w_n']:.2f}"
            clear_str = f"{d['raw_c']:.3f}({d['w_c']:.2f})"
            
            print(f"{i+1:<4} | {obj['name']:<12} | {z_str:<12} | {status_str:<15} | {dens_str:<10} | {clear_str:<15} | {obj['score']:.4f}")
        print("="*120)
        print(f"Best: [{ranked_objects[0]['name']}]", flush=True)

    def publish_debug_axes(self, best_obj, frame_id):
        ma = MarkerArray()
        timestamp = self.get_clock().now().to_msg()
        
        # [수정] DELETEALL 제거함! -> 깜빡임의 주 원인
        # 대신 마커에 lifetime을 주어 시간이 지나면 자연스럽게 사라지게 함
        
        p_orig = [best_obj['pose'].position.x, best_obj['pose'].position.y, best_obj['pose'].position.z]
        r = best_obj['rotation']
        
        # 원본 위치 축
        self.add_rgb_axes(ma, frame_id, timestamp, "origin", p_orig, r, 1.0)
        
        # 이동된 위치(타겟) 축
        p_moved = best_obj['moved_pos']
        self.add_rgb_axes(ma, frame_id, timestamp, "moved", p_moved, r, 0.5)

        # 경로 화살표
        m_arrow = Marker()
        m_arrow.header.frame_id = frame_id
        m_arrow.header.stamp = timestamp
        m_arrow.ns, m_arrow.id = "path", 999
        m_arrow.type = Marker.ARROW
        m_arrow.action = Marker.ADD
        m_arrow.lifetime.sec = 0; m_arrow.lifetime.nanosec = 200000000 # 0.2초 수명
        m_arrow.scale.x, m_arrow.scale.y, m_arrow.scale.z = 0.005, 0.01, 0.0
        m_arrow.color.r, m_arrow.color.g, m_arrow.color.b, m_arrow.color.a = 1.0, 1.0, 0.0, 0.8
        m_arrow.points = [Point(x=p_orig[0], y=p_orig[1], z=p_orig[2]), Point(x=p_moved[0], y=p_moved[1], z=p_moved[2])]
        ma.markers.append(m_arrow)

        self.pub_markers.publish(ma)

    def add_rgb_axes(self, ma, frame_id, timestamp, ns, pos, r, alpha):
        scale = 0.08
        width = 0.005
        vx = r.apply([scale, 0, 0])
        vy = r.apply([0, scale, 0])
        vz = r.apply([0, 0, scale])
        
        axes = [(0, (1.0, 0.0, 0.0), vx), (1, (0.0, 1.0, 0.0), vy), (2, (0.0, 0.0, 1.0), vz)]
        start = Point(x=pos[0], y=pos[1], z=pos[2])
        
        for i, rgb, vec in axes:
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = timestamp
            m.ns = ns
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            
            # [수정] 수명(Lifetime) 설정: 0.2초
            # 0.2초 동안 갱신이 없으면 사라짐. 갱신되면 시간이 리셋됨.
            m.lifetime.sec = 0
            m.lifetime.nanosec = 200000000 
            
            m.scale.x = width; m.scale.y = width * 2; m.scale.z = 0.0
            m.color.r, m.color.g, m.color.b = rgb
            m.color.a = alpha
            end = Point(x=pos[0]+vec[0], y=pos[1]+vec[1], z=pos[2]+vec[2])
            m.points = [start, end]
            ma.markers.append(m)
def main(args=None):
    rclpy.init(args=args)
    node = GraspPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()