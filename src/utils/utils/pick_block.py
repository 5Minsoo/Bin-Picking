#!/usr/bin/env python3
import math
import re
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from simulation_interfaces.srv import GetEntitiesStates

from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene, GetCartesianPath
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    PlanningScene,
    CollisionObject,
)
from geometry_msgs.msg import Pose, Quaternion
from shape_msgs.msg import SolidPrimitive


# ====== 설정 ======
ACTION_NAME = "move_action"          # MoveGroup action name
EXEC_ACTION = "execute_trajectory"   # ExecuteTrajectory action name
CART_SRV    = "compute_cartesian_path"

GROUP      = "manipulator"
EE_LINK    = "link_6"
FRAME      = "base_link"

ENTITIES_SRV = "/get_entities_states"
USER_MIN = 1
USER_MAX = 42  # 1~42 -> obj_0~obj_41

# ✅ Z 오프셋들
Z_OFFSET_ABOVE = 0.23
Z_OFFSET_DOWN  = 0.03
Z_STEP         = 0.02
Z_MIN_SAFETY   = 0.01

# OMPL 접근(above)용 위치 허용 반경
TOL_M = 0.01

# “아래보기” 목표 자세(예: tool이 아래를 보게)
TARGET_RPY = (math.pi, 0.0, 0.0)
ORI_TOL = (0.1, 0.1, 0.15)

NUM_ATTEMPTS = 3
REPLAN       = False
ALLOWED_TIME = 7.0

# Cartesian 설정
CART_MAX_STEP = 0.01     # 작을수록 더 매끈/성공률↑(너무 작으면 느려질 수 있음)
CART_JUMP_TH  = 0.0      # 0이면 점프 검사 비활성(환경에 따라 1~2로 올려볼 수 있음)
CART_MIN_FRAC = 0.98     # fraction이 이보다 낮으면 실패 처리

ADD_FLOOR = True
FLOOR_ID = "floor"
FLOOR_SIZE_XY = 6.0
FLOOR_THICK   = 0.05
FLOOR_Z_TOP   = 0.0
# ================


def quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


def getch():
    """엔터 없이 1글자 키 입력"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class GoToObjectAbove(Node):
    def __init__(self):
        super().__init__("pick_cartesian_updown")

        # OMPL 접근용 MoveGroup action
        self.move_client = ActionClient(self, MoveGroup, ACTION_NAME)

        # Cartesian 경로 계산 서비스 + 실행 액션
        self.cart_client = self.create_client(GetCartesianPath, CART_SRV)
        self.exec_client = ActionClient(self, ExecuteTrajectory, EXEC_ACTION)

        # PlanningScene 서비스
        self.scene_client = self.create_client(GetPlanningScene, "get_planning_scene")
        self.apply_scene_client = self.create_client(ApplyPlanningScene, "apply_planning_scene")

        # 오브젝트 상태 서비스
        self.entities_client = self.create_client(GetEntitiesStates, ENTITIES_SRV)

        # 선택/목표 저장
        self.sel_obj_idx = None
        self.sel_obj_pose = None      # (ox, oy, oz)
        self.cur_target_xyz = None    # (x,y,z) 마지막으로 “도달한” EE 목표

        self._floor_added = False

    # ---------- Planning Scene / State ----------
    def get_current_state(self):
        if not self.scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("get_planning_scene 서비스 대기 실패")
            return None
        req = GetPlanningScene.Request()
        req.components.components = req.components.ROBOT_STATE
        fut = self.scene_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        return fut.result().scene.robot_state if fut.result() else None

    def add_floor_collision_once(self):
        if self._floor_added:
            return True
        if not self.apply_scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("apply_planning_scene 서비스 대기 실패")
            return False

        co = CollisionObject()
        co.id = FLOOR_ID
        co.header.frame_id = FRAME
        co.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(FLOOR_SIZE_XY), float(FLOOR_SIZE_XY), float(FLOOR_THICK)]
        co.primitives.append(box)

        pose = Pose()
        pose.position.x = 0.0
        pose.position.y = 0.0
        pose.position.z = float(FLOOR_Z_TOP - (FLOOR_THICK * 0.5))
        pose.orientation.w = 1.0
        co.primitive_poses.append(pose)

        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)

        req = ApplyPlanningScene.Request()
        req.scene = ps
        fut = self.apply_scene_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)

        ok = bool(fut.result() and fut.result().success)
        if ok:
            self._floor_added = True
            self.get_logger().info(f"✅ 바닥 충돌체 추가됨(1회): id={FLOOR_ID}")
        else:
            self.get_logger().warn("❌ 바닥 충돌체 추가 실패")
        return ok

    # ---------- Object Pose ----------
    def fetch_objects_xyz(self):
        if not self.entities_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f"{ENTITIES_SRV} 서비스 대기 실패")
            return {}

        req = GetEntitiesStates.Request()
        req.filters.filter = r"^/World/obj_[0-9]+$"

        fut = self.entities_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res is None:
            self.get_logger().error("get_entities_states 응답 실패")
            return {}

        out = {}
        for name, st in zip(res.entities, res.states):
            m = re.search(r"obj_(\d+)$", name)
            if not m:
                continue
            idx = int(m.group(1))
            p = st.pose.position
            out[idx] = (float(p.x), float(p.y), float(p.z))
        return out

    def refresh_selected_pose(self):
        if self.sel_obj_idx is None:
            return False
        objs = self.fetch_objects_xyz()
        if self.sel_obj_idx not in objs:
            self.get_logger().warn(f"obj_{self.sel_obj_idx} 가 목록에 없습니다.")
            return False
        self.sel_obj_pose = objs[self.sel_obj_idx]
        ox, oy, oz = self.sel_obj_pose
        self.get_logger().info(f"🔄 obj_{self.sel_obj_idx} 갱신: ({ox:.3f},{oy:.3f},{oz:.3f})")
        return True

    # ---------- Target helpers ----------
    def _compute_targets(self, ox, oy, oz):
        above = (ox, oy, oz + float(Z_OFFSET_ABOVE))
        down  = (ox, oy, oz + float(Z_OFFSET_DOWN))
        return above, down

    def _make_pose(self, x, y, z):
        z = max(float(z), float(Z_MIN_SAFETY))
        r, p, yaw = TARGET_RPY
        qx, qy, qz, qw = quaternion_from_euler(r, p, yaw)
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
        return pose

    # ---------- 1) 접근(above) : OMPL ----------
    def send_ompl_xyz(self, x, y, z) -> bool:
        if ADD_FLOOR:
            self.add_floor_collision_once()

        if not self.move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"Action 서버({ACTION_NAME}) 연결 실패")
            return False

        st = self.get_current_state()

        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP
        goal.request.num_planning_attempts = int(NUM_ATTEMPTS)
        goal.request.allowed_planning_time = float(ALLOWED_TIME)
        if st:
            goal.request.start_state = st

        c = Constraints()
        c.name = "approach_xyz_ori"

        # Position constraint (sphere)
        pc = PositionConstraint()
        pc.header.frame_id = FRAME
        pc.link_name = EE_LINK
        pc.weight = 1.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [float(TOL_M)]
        pc.constraint_region.primitives.append(sphere)

        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = max(float(z), float(Z_MIN_SAFETY))
        pose.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(pose)
        c.position_constraints.append(pc)

        # Orientation constraint
        r, p, yaw = TARGET_RPY
        qx, qy, qz, qw = quaternion_from_euler(r, p, yaw)
        oc = OrientationConstraint()
        oc.header.frame_id = FRAME
        oc.link_name = EE_LINK
        oc.orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
        oc.absolute_x_axis_tolerance = float(ORI_TOL[0])
        oc.absolute_y_axis_tolerance = float(ORI_TOL[1])
        oc.absolute_z_axis_tolerance = float(ORI_TOL[2])
        oc.weight = 1.0
        c.orientation_constraints.append(oc)

        goal.request.goal_constraints.append(c)
        goal.planning_options.plan_only = False
        goal.planning_options.replan = bool(REPLAN)

        self.get_logger().info(f"🚀 [OMPL] 이동 요청: XYZ({x:.3f}, {y:.3f}, {z:.3f})")
        send_future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        gh = send_future.result()
        if not gh or not gh.accepted:
            self.get_logger().error("Goal 거부됨")
            return False

        res_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        wrapped = res_future.result()
        if wrapped is None:
            self.get_logger().error("결과 수신 실패")
            return False

        if wrapped.result.error_code.val != 1:
            self.get_logger().error(f"❌ [OMPL] 실패. MoveIt Error Code: {wrapped.result.error_code.val}")
            return False

        self.cur_target_xyz = (float(x), float(y), float(max(z, Z_MIN_SAFETY)))
        self.get_logger().info("✅ [OMPL] 이동 성공!")
        return True

    # ---------- 2) 내려/올라/스텝 : Cartesian ----------
    def _compute_cartesian(self, target_pose: Pose):
        if not self.cart_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f"{CART_SRV} 서비스 대기 실패")
            return None, 0.0

        st = self.get_current_state()
        if st is None:
            self.get_logger().error("Cartesian: 현재 로봇 상태를 못 가져왔습니다.")
            return None, 0.0

        req = GetCartesianPath.Request()
        req.header.frame_id = FRAME
        req.start_state = st
        req.group_name = GROUP
        req.link_name = EE_LINK

        # 한 개 waypoint면: 현재 EE -> target_pose 를 직선으로 보간 시도
        req.waypoints = [target_pose]

        req.max_step = float(CART_MAX_STEP)
        req.jump_threshold = float(CART_JUMP_TH)
        req.avoid_collisions = True

        fut = self.cart_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res is None:
            self.get_logger().error("Cartesian: 응답 실패")
            return None, 0.0

        return res.solution, float(res.fraction)

    def _execute_trajectory(self, robot_traj) -> bool:
        if not self.exec_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"Action 서버({EXEC_ACTION}) 연결 실패")
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = robot_traj

        send_future = self.exec_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        gh = send_future.result()
        if not gh or not gh.accepted:
            self.get_logger().error("ExecuteTrajectory goal 거부됨")
            return False

        res_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        wrapped = res_future.result()
        if wrapped is None:
            self.get_logger().error("ExecuteTrajectory 결과 수신 실패")
            return False

        # ExecuteTrajectory.Result에는 error_code가 있을 수도/없을 수도(버전에 따라 다름)
        self.get_logger().info("✅ [CART] 실행 완료")
        return True

    def send_cartesian_xyz(self, x, y, z) -> bool:
        if ADD_FLOOR:
            self.add_floor_collision_once()

        target = self._make_pose(x, y, z)

        self.get_logger().info(f"🧭 [CART] 목표: XYZ({target.position.x:.3f}, {target.position.y:.3f}, {target.position.z:.3f})")
        traj, frac = self._compute_cartesian(target)
        if traj is None:
            return False

        self.get_logger().info(f"🧩 [CART] fraction={frac:.3f}")
        if frac < CART_MIN_FRAC:
            self.get_logger().error(f"❌ [CART] fraction 낮음({frac:.3f}) → 실패 처리")
            return False

        if not self._execute_trajectory(traj):
            return False

        self.cur_target_xyz = (float(target.position.x), float(target.position.y), float(target.position.z))
        return True

    # ---------- High-level motions ----------
    def move_above_selected_cartesian(self):
        """현재 위치에서 선택 오브젝트 above까지 '직선'으로 올리고 싶을 때(키 k에서 사용)"""
        if self.sel_obj_pose is None:
            return False
        ox, oy, oz = self.sel_obj_pose
        above, _ = self._compute_targets(ox, oy, oz)
        return self.send_cartesian_xyz(*above)

    def move_down_selected_cartesian(self):
        if self.sel_obj_pose is None:
            return False
        ox, oy, oz = self.sel_obj_pose
        _, down = self._compute_targets(ox, oy, oz)
        return self.send_cartesian_xyz(*down)

    def step_move_cartesian(self, dz):
        if self.cur_target_xyz is None:
            self.get_logger().warn("스텝 이동: 아직 목표가 없습니다. 먼저 위로 이동하세요.")
            return False
        x, y, z = self.cur_target_xyz
        return self.send_cartesian_xyz(x, y, z + float(dz))

    # ---------- Main loop ----------
    def interactive_loop(self):
        self.get_logger().info("✅ 숫자 선택 후, j/k/w/s 로 z방향 카르테시안 이동합니다.")
        self.get_logger().info("  키: j=down(CART), k=up(CART), s=step down(CART), w=step up(CART), r=refresh obj, n=new obj, q=quit")

        while rclpy.ok():
            objs = self.fetch_objects_xyz()
            if objs:
                available = sorted(objs.keys())
                self.get_logger().info(f"현재 감지된 obj 인덱스: {available[:30]}{' ...' if len(available)>30 else ''}")
            else:
                self.get_logger().warn("가져온 obj가 없습니다. (스폰/필터/서비스 확인)")

            s = input(f"\n이동할 번호 입력 ({USER_MIN}~{USER_MAX}) > ").strip().lower()
            if s in ("q", "quit", "exit"):
                break
            if not s or (not s.isdigit()):
                print("숫자만 입력해주세요.")
                continue

            user_n = int(s)
            if user_n < USER_MIN or user_n > USER_MAX:
                print(f"범위 오류: {USER_MIN}~{USER_MAX}")
                continue

            obj_idx = user_n - 1
            if obj_idx not in objs:
                print(f"obj_{obj_idx} 가 목록에 없습니다.")
                continue

            self.sel_obj_idx = obj_idx
            self.sel_obj_pose = objs[obj_idx]
            ox, oy, oz = self.sel_obj_pose
            above, down = self._compute_targets(ox, oy, oz)

            print(f"\n선택: obj_{obj_idx} pos=({ox:.3f},{oy:.3f},{oz:.3f})")
            print(f"  above z={above[2]:.3f} (Z_OFFSET_ABOVE={Z_OFFSET_ABOVE})")
            print(f"  down  z={down[2]:.3f} (Z_OFFSET_DOWN ={Z_OFFSET_DOWN})")

            # 1) 먼저 above까지는 OMPL로 접근(충돌 회피/우회 가능)
            if not self.send_cartesian_xyz(*above):
                print("위로 이동(CART) 실패. (충돌/IK/제약 확인)")
                continue

            print("\n키 제어 모드 진입: j/k/w/s/r/n/q")
            print("  키: j=down(CART), k=up(CART), s=step down(CART), w=step up(CART), r=refresh obj, n=new obj, q=quit")

            while rclpy.ok():
                ch = getch().lower()

                if ch == 'q':
                    return
                if ch == 'n':
                    print("\n다른 오브젝트 선택으로 돌아갑니다.")
                    break
                if ch == 'r':
                    self.refresh_selected_pose()
                    continue

                # ✅ 내려가기/올라가기/스텝은 전부 Cartesian
                if ch == 'j':
                    self.move_down_selected_cartesian()
                    continue
                if ch == 'k':
                    self.move_above_selected_cartesian()
                    continue
                if ch == 's':
                    self.step_move_cartesian(-Z_STEP)
                    continue
                if ch == 'w':
                    self.step_move_cartesian(+Z_STEP)
                    continue


def main():
    rclpy.init()
    node = GoToObjectAbove()
    try:
        node.interactive_loop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
