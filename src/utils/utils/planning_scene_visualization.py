#!/usr/bin/env python3
import math
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Pose, PoseStamped, Vector3Stamped

from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, PlanningScene, AttachedCollisionObject
from moveit_msgs.srv import ApplyPlanningScene

import tf2_ros


# -----------------------------
# Quaternion / RPY helpers
# -----------------------------
def rpy_to_quat(r: float, p: float, y: float) -> Tuple[float, float, float, float]:
    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


def quat_mul(q1, q2) -> Tuple[float, float, float, float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return (x, y, z, w)


def quat_conj(q) -> Tuple[float, float, float, float]:
    x, y, z, w = q
    return (-x, -y, -z, w)


def quat_rotate_vec(q, v) -> Tuple[float, float, float]:
    vx, vy, vz = v
    qv = (vx, vy, vz, 0.0)
    out = quat_mul(quat_mul(q, qv), quat_conj(q))
    return (out[0], out[1], out[2])


def quat_to_rpy(q) -> Tuple[float, float, float]:
    x, y, z, w = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


def compose_pose(
    parent_p: Tuple[float, float, float],
    parent_q: Tuple[float, float, float, float],
    child_p: Tuple[float, float, float],
    child_q: Tuple[float, float, float, float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    rp = quat_rotate_vec(parent_q, child_p)
    p = (parent_p[0] + rp[0], parent_p[1] + rp[1], parent_p[2] + rp[2])
    q = quat_mul(parent_q, child_q)
    return p, q


def to_pose_msg(p: Tuple[float, float, float], q: Tuple[float, float, float, float]) -> Pose:
    m = Pose()
    m.position.x, m.position.y, m.position.z = p
    m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = q
    return m


def transform_pose_with_tf(
    p: Tuple[float, float, float],
    q: Tuple[float, float, float, float],
    tf_t: Tuple[float, float, float],
    tf_q: Tuple[float, float, float, float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    pr = quat_rotate_vec(tf_q, p)
    p2 = (pr[0] + tf_t[0], pr[1] + tf_t[1], pr[2] + tf_t[2])
    q2 = quat_mul(tf_q, q)
    return p2, q2


# -----------------------------
# collision (ring by boxes)
# -----------------------------
def ring_by_boxes_local(
    r_out: float,
    r_in: float,
    height: float,
    z_center: float,
    segments: int,
    q_group: Tuple[float, float, float, float],
) -> Tuple[List[SolidPrimitive], List[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]]:
    segments = max(8, int(segments))
    r_out = max(1e-6, float(r_out))
    r_in = max(0.0, float(r_in))
    r_in = min(r_in, r_out * 0.999)
    height = max(1e-6, float(height))

    thickness = max(1e-6, r_out - r_in)
    r_mid = r_in + thickness * 0.5

    chord = 2.0 * r_mid * math.sin(math.pi / segments)
    seg_len = max(1e-6, chord * 1.08)
    seg_w = thickness
    seg_h = height

    prims: List[SolidPrimitive] = []
    poses: List[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]] = []

    for i in range(segments):
        ang = (2.0 * math.pi * i) / segments
        x = r_mid * math.cos(ang)
        y = r_mid * math.sin(ang)

        q_local = rpy_to_quat(0.0, 0.0, ang + math.pi * 0.5)
        p_local = (x, y, z_center)

        p_rot = quat_rotate_vec(q_group, p_local)
        q_rot = quat_mul(q_group, q_local)

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [seg_len, seg_w, seg_h]

        prims.append(box)
        poses.append((p_rot, q_rot))

    return prims, poses


def tshape_collision_local(
    base_r_out: float,
    base_h: float,
    body_r_out: float,
    body_h: float,
    wall_t: float,
    segments: int,
    q_group: Tuple[float, float, float, float],
    base_hole_clearance: float = 0.0,
) -> Tuple[List[SolidPrimitive], List[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]]:
    base_r_out = max(1e-6, float(base_r_out))
    base_h = max(1e-6, float(base_h))
    body_r_out = max(1e-6, float(body_r_out))
    body_h = max(1e-6, float(body_h))
    wall_t = max(1e-6, float(wall_t))
    wall_t = min(wall_t, body_r_out * 0.95)

    body_r_in = max(0.0, body_r_out - wall_t)
    base_r_in = body_r_in + max(0.0, float(base_hole_clearance))

    min_radial = max(0.002, wall_t)
    if base_r_out < base_r_in + min_radial:
        base_r_out = base_r_in + min_radial

    z_base = 0.0
    z_body = (base_h * 0.5) + (body_h * 0.5)

    prims1, poses1 = ring_by_boxes_local(base_r_out, base_r_in, base_h, z_base, segments, q_group)
    prims2, poses2 = ring_by_boxes_local(body_r_out, body_r_in, body_h, z_body, segments, q_group)
    return prims1 + prims2, poses1 + poses2


# -----------------------------
# Main node
# -----------------------------
class GzEntitiesToMoveItScene(Node):
    def __init__(self):
        super().__init__("gz_entities_to_moveit_scene")

        # Inputs
        self.declare_parameter("pose_tf_topic", "/gz_pose_tf")
        self.declare_parameter("planning_frame", "world")
        self.declare_parameter("apply_service", "/apply_planning_scene")

        # Rate / apply
        self.declare_parameter("update_hz", 1.0)
        self.declare_parameter("apply_timeout_sec", 2.0)
        self._pending_apply_future = None
        self._pending_apply_start_time = None

        # Change detection to avoid spamming ApplyPlanningScene
        self.declare_parameter("pose_pos_epsilon", 1e-4)    # meters
        self.declare_parameter("pose_ang_epsilon", 1e-3)    # radians (approx)

        # Names / cleanup
        self.declare_parameter("bin_name", "bin_001")
        self.declare_parameter("obj_prefix", "obj_")
        self.declare_parameter("num_objects", 10)
        self.declare_parameter("cleanup_on_start", True)
        self.declare_parameter("cleanup_max_index", 199)

        # Bin collision
        self.declare_parameter("bin_inner_xy", [0.40, 0.30])
        self.declare_parameter("bin_wall_h", 0.20)
        self.declare_parameter("bin_wall_thick", 0.01)
        self.declare_parameter("bin_floor_thick", 0.01)

        # ㅗ collision params
        self.declare_parameter("col_base_outer_radius", 0.025)
        self.declare_parameter("col_base_h", 0.006)
        self.declare_parameter("col_cyl_outer_radius", 0.015)
        self.declare_parameter("col_cyl_outer_radius_scale", 1.30)
        self.declare_parameter("col_cyl_h", 0.030)
        self.declare_parameter("col_cyl_wall_t", 0.002)
        self.declare_parameter("col_cyl_segments", 12)
        self.declare_parameter("col_base_hole_clearance", 0.0)
        self.declare_parameter("col_group_rpy", [0.0, math.pi / 2.0, 0.0])

        # Pose publish
        self.declare_parameter("publish_pose_hz", 5.0)
        self.declare_parameter("publish_all_object_poses", True)
        self.declare_parameter("selected_object", "")
        self.declare_parameter("selected_pose_frame", "")

        # Attach
        self.declare_parameter("enable_attach", True)
        self.declare_parameter("use_tf_for_attach", True)
        self.declare_parameter("eef_link", "tool0")
        self.declare_parameter("touch_links", [])

        # State
        self.entity_pose: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]] = {}
        self.attached_id: Optional[str] = None

        # Cache for "last applied" to avoid spamming MoveIt
        self._last_applied_pose: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]] = {}

        # MoveIt apply client
        apply_srv = str(self.get_parameter("apply_service").value)
        self.apply_cli = self.create_client(ApplyPlanningScene, apply_srv)

        # TF listener (attach)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscriptions
        pose_topic = str(self.get_parameter("pose_tf_topic").value)
        self.create_subscription(TFMessage, pose_topic, self._on_pose_tf, 50)
        self.create_subscription(String, "~/attach", self._on_attach, 10)
        self.create_subscription(String, "~/detach", self._on_detach, 10)

        # Publishers
        self.pub_all_poses = self.create_publisher(String, "/gz_scene/object_poses", 10)
        self.pub_sel_name = self.create_publisher(String, "/gz_scene/selected_object/name", 10)
        self.pub_sel_pose = self.create_publisher(PoseStamped, "/gz_scene/selected_object/pose", 10)
        self.pub_sel_rpy = self.create_publisher(Vector3Stamped, "/gz_scene/selected_object/rpy", 10)
        self.pub_sel_ypr_deg = self.create_publisher(Vector3Stamped, "/gz_scene/selected_object/ypr_deg", 10)

        # Precompute collision templates
        self._bin_prims_local, self._bin_local_poses = self._precompute_bin_local()
        self._obj_prims_local, self._obj_local_poses = self._precompute_obj_local()

        # Cleanup
        if bool(self.get_parameter("cleanup_on_start").value):
            self._cleanup_scene_all()

        # Timers
        hz = float(self.get_parameter("update_hz").value)
        self.timer = self.create_timer(1.0 / max(0.1, hz), self._tick_apply_scene)

        phz = float(self.get_parameter("publish_pose_hz").value)
        self.timer_pub = self.create_timer(1.0 / max(0.1, phz), self._tick_publish_poses)

        self.get_logger().info(
            f"Started.\n"
            f"- pose_tf_topic: {pose_topic}\n"
            f"- apply_service: {apply_srv}\n"
            f"- planning_frame: {self.get_parameter('planning_frame').value}\n"
            f"Attach: ros2 topic pub --once /{self.get_name()}/attach std_msgs/msg/String \"{{data: 'obj_002'}}\"\n"
            f"Detach: ros2 topic pub --once /{self.get_name()}/detach std_msgs/msg/String \"{{data: 'obj_002'}}\""
        )

    # -----------------------------
    # small helpers
    # -----------------------------
    def _pose_changed(
        self,
        a: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
        b: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
        pos_eps: float,
        ang_eps: float,
    ) -> bool:
        (pa, qa) = a
        (pb, qb) = b
        dx = pa[0] - pb[0]
        dy = pa[1] - pb[1]
        dz = pa[2] - pb[2]
        if (dx * dx + dy * dy + dz * dz) > (pos_eps * pos_eps):
            return True

        # quaternion "distance" (very lightweight check)
        # Use 1 - |dot| as angular difference proxy
        dot = abs(qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3])
        dot = max(0.0, min(1.0, dot))
        ang = 2.0 * math.acos(dot)  # [0, pi]
        return ang > ang_eps

    # ---- Precompute collisions ----
    def _precompute_bin_local(self):
        inner_xy = self.get_parameter("bin_inner_xy").value
        inner_x, inner_y = float(inner_xy[0]), float(inner_xy[1])
        wall_h = float(self.get_parameter("bin_wall_h").value)
        wall_t = float(self.get_parameter("bin_wall_thick").value)
        floor_t = float(self.get_parameter("bin_floor_thick").value)

        outer_x = inner_x + 2.0 * wall_t
        outer_y = inner_y + 2.0 * wall_t
        floor_z = -floor_t * 0.5
        wall_z = wall_h * 0.5

        prims: List[SolidPrimitive] = []
        poses: List[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]] = []

        def add_box(size_xyz, p_xyz):
            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])]
            prims.append(box)
            poses.append((p_xyz, (0.0, 0.0, 0.0, 1.0)))

        add_box((outer_x, outer_y, floor_t), (0.0, 0.0, floor_z))
        add_box((wall_t, outer_y, wall_h), (+(inner_x * 0.5 + wall_t * 0.5), 0.0, wall_z))
        add_box((wall_t, outer_y, wall_h), (-(inner_x * 0.5 + wall_t * 0.5), 0.0, wall_z))
        add_box((outer_x, wall_t, wall_h), (0.0, +(inner_y * 0.5 + wall_t * 0.5), wall_z))
        add_box((outer_x, wall_t, wall_h), (0.0, -(inner_y * 0.5 + wall_t * 0.5), wall_z))

        return prims, poses

    def _precompute_obj_local(self):
        base_r_out = float(self.get_parameter("col_base_outer_radius").value)
        base_h = float(self.get_parameter("col_base_h").value)

        body_r_out0 = float(self.get_parameter("col_cyl_outer_radius").value)
        body_scale = float(self.get_parameter("col_cyl_outer_radius_scale").value)
        body_r_out = body_r_out0 * body_scale

        body_h = float(self.get_parameter("col_cyl_h").value)
        wall_t = float(self.get_parameter("col_cyl_wall_t").value)
        seg = int(self.get_parameter("col_cyl_segments").value)
        clearance = float(self.get_parameter("col_base_hole_clearance").value)

        group_rpy = self.get_parameter("col_group_rpy").value
        q_group = rpy_to_quat(float(group_rpy[0]), float(group_rpy[1]), float(group_rpy[2]))

        prims, poses = tshape_collision_local(
            base_r_out=base_r_out,
            base_h=base_h,
            body_r_out=body_r_out,
            body_h=body_h,
            wall_t=wall_t,
            segments=seg,
            q_group=q_group,
            base_hole_clearance=clearance,
        )
        return prims, poses

    # ---- Pose callback (TFMessage) ----
    def _on_pose_tf(self, msg: TFMessage):
        bin_name = str(self.get_parameter("bin_name").value)
        obj_prefix = str(self.get_parameter("obj_prefix").value)
        planning_frame = str(self.get_parameter("planning_frame").value)

        for ts in msg.transforms:
            parent = ts.header.frame_id.strip()
            child = ts.child_frame_id.strip()

            # (3) Frame guard: accept only if parent == planning_frame (for stability)
            if parent and parent != planning_frame:
                continue

            if not (child == bin_name or child.startswith(obj_prefix)):
                continue

            p = (ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z)
            q = (ts.transform.rotation.x, ts.transform.rotation.y, ts.transform.rotation.z, ts.transform.rotation.w)
            self.entity_pose[child] = (p, q)

    # ---- ApplyPlanningScene (async only) ----
    def _apply_scene_async(self, scene: PlanningScene) -> bool:
        if not self.apply_cli.service_is_ready():
            self.get_logger().warn("/apply_planning_scene not ready.")
            return False

        if self._pending_apply_future is not None and not self._pending_apply_future.done():
            return False

        req = ApplyPlanningScene.Request()
        req.scene = scene
        self._pending_apply_future = self.apply_cli.call_async(req)
        self._pending_apply_start_time = self.get_clock().now()
        return True

    def _poll_apply_future(self) -> bool:
        """Return True if 'pending' is cleared (done or timeout)."""
        if self._pending_apply_future is None:
            return True

        if self._pending_apply_future.done():
            res = self._pending_apply_future.result()
            ok = (res is not None and bool(res.success))
            if not ok:
                self.get_logger().warn("ApplyPlanningScene failure.")
            self._pending_apply_future = None
            self._pending_apply_start_time = None
            return True

        timeout = float(self.get_parameter("apply_timeout_sec").value)
        if self._pending_apply_start_time is not None:
            dt = (self.get_clock().now() - self._pending_apply_start_time).nanoseconds * 1e-9
            if dt > timeout:
                self.get_logger().warn("ApplyPlanningScene timeout.")
                self._pending_apply_future = None
                self._pending_apply_start_time = None
                return True

        return False

    # ---- Scene cleanup ----
    def _cleanup_scene_all(self):
        frame = str(self.get_parameter("planning_frame").value)
        bin_name = str(self.get_parameter("bin_name").value)
        obj_prefix = str(self.get_parameter("obj_prefix").value)
        max_idx = int(self.get_parameter("cleanup_max_index").value)

        scene = PlanningScene()
        scene.is_diff = True

        # remove bin
        co = CollisionObject()
        co.header.frame_id = frame
        co.id = bin_name
        co.operation = CollisionObject.REMOVE
        scene.world.collision_objects.append(co)

        # remove objs wide range
        for i in range(max_idx + 1):
            oid = f"{obj_prefix}{i:03d}"
            c = CollisionObject()
            c.header.frame_id = frame
            c.id = oid
            c.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(c)

        scene.robot_state.is_diff = True
        ok = self._apply_scene_async(scene)
        self.get_logger().info(f"Initial cleanup requested: {ok}")

        # cleanup also resets cache, so first real apply won't be skipped
        self._last_applied_pose.clear()

    def _make_collision_object_world(
        self,
        obj_id: str,
        prims_local: List[SolidPrimitive],
        local_poses: List[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]],
        world_pose: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
    ) -> CollisionObject:
        frame = str(self.get_parameter("planning_frame").value)
        (wp, wq) = world_pose

        co = CollisionObject()
        co.header.frame_id = frame
        co.id = obj_id
        co.operation = CollisionObject.ADD
        co.primitives = prims_local
        co.primitive_poses = []

        for (lp, lq) in local_poses:
            p2, q2 = compose_pose(wp, wq, lp, lq)
            co.primitive_poses.append(to_pose_msg(p2, q2))

        return co

    # ---- Attach / Detach (optional) ----
    def _on_attach(self, msg: String):
        if not bool(self.get_parameter("enable_attach").value):
            return
        oid = msg.data.strip()
        if not oid:
            return
        self.attached_id = oid
        self.get_logger().info(f"Attach requested: {oid}")
        self._apply_attach_detach()

    def _on_detach(self, msg: String):
        if not bool(self.get_parameter("enable_attach").value):
            return
        oid = msg.data.strip() or self.attached_id
        if not oid:
            return
        self.get_logger().info(f"Detach requested: {oid}")
        self.attached_id = None
        self._apply_attach_detach(detach_id=oid)

    def _apply_attach_detach(self, detach_id: Optional[str] = None):
        frame = str(self.get_parameter("planning_frame").value)
        eef_link = str(self.get_parameter("eef_link").value)
        touch_links = list(self.get_parameter("touch_links").value)
        use_tf = bool(self.get_parameter("use_tf_for_attach").value)

        # If apply is pending, skip; we'll try next message/tick
        if self._pending_apply_future is not None and not self._pending_apply_future.done():
            return

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True

        if detach_id:
            aco = AttachedCollisionObject()
            aco.link_name = eef_link
            aco.object.id = detach_id
            aco.object.operation = CollisionObject.REMOVE
            scene.robot_state.attached_collision_objects.append(aco)
            self._apply_scene_async(scene)
            return

        if not self.attached_id:
            return

        oid = self.attached_id
        if oid not in self.entity_pose:
            self.get_logger().warn(f"Attach requested but pose unknown: {oid}")
            return

        # remove from world
        rem = CollisionObject()
        rem.header.frame_id = frame
        rem.id = oid
        rem.operation = CollisionObject.REMOVE
        scene.world.collision_objects.append(rem)

        # attached
        aco = AttachedCollisionObject()
        aco.link_name = eef_link
        aco.touch_links = touch_links

        world_pose = self.entity_pose[oid]

        if use_tf:
            try:
                tf = self.tf_buffer.lookup_transform(
                    target_frame=eef_link,
                    source_frame=frame,
                    time=Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2),
                )
                tf_t = (tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z)
                tf_q = (tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w)

                co = CollisionObject()
                co.header.frame_id = eef_link
                co.id = oid
                co.operation = CollisionObject.ADD
                co.primitives = self._obj_prims_local
                co.primitive_poses = []

                (wp, wq) = world_pose
                for (lp, lq) in self._obj_local_poses:
                    pw, qw = compose_pose(wp, wq, lp, lq)
                    pe, qe = transform_pose_with_tf(pw, qw, tf_t, tf_q)
                    co.primitive_poses.append(to_pose_msg(pe, qe))

                aco.object = co
            except Exception as e:
                self.get_logger().warn(f"TF attach fallback (no TF). reason: {e}")
                aco.object = self._make_collision_object_world(oid, self._obj_prims_local, self._obj_local_poses, world_pose)
        else:
            aco.object = self._make_collision_object_world(oid, self._obj_prims_local, self._obj_local_poses, world_pose)

        scene.robot_state.attached_collision_objects.append(aco)
        self._apply_scene_async(scene)

    # ---- Periodic: apply scene ----
    def _tick_apply_scene(self):
        # 0) finish/timeout pending apply first
        if not self._poll_apply_future():
            return

        bin_name = str(self.get_parameter("bin_name").value)
        obj_prefix = str(self.get_parameter("obj_prefix").value)
        num_objects = int(self.get_parameter("num_objects").value)

        pos_eps = float(self.get_parameter("pose_pos_epsilon").value)
        ang_eps = float(self.get_parameter("pose_ang_epsilon").value)

        scene = PlanningScene()
        scene.is_diff = True

        any_change = False

        # bin
        if bin_name in self.entity_pose:
            pose = self.entity_pose[bin_name]
            last = self._last_applied_pose.get(bin_name)
            if last is None or self._pose_changed(pose, last, pos_eps, ang_eps):
                scene.world.collision_objects.append(
                    self._make_collision_object_world(bin_name, self._bin_prims_local, self._bin_local_poses, pose)
                )
                self._last_applied_pose[bin_name] = pose
                any_change = True

        # objects
        for i in range(num_objects):
            oid = f"{obj_prefix}{i:03d}"
            if oid == self.attached_id:
                continue
            pose = self.entity_pose.get(oid)
            if pose is None:
                continue
            last = self._last_applied_pose.get(oid)
            if last is None or self._pose_changed(pose, last, pos_eps, ang_eps):
                scene.world.collision_objects.append(
                    self._make_collision_object_world(oid, self._obj_prims_local, self._obj_local_poses, pose)
                )
                self._last_applied_pose[oid] = pose
                any_change = True

        if any_change and scene.world.collision_objects:
            self._apply_scene_async(scene)

    # ---- Periodic: publish poses + RPY ----
    def _tick_publish_poses(self):
        publish_all = bool(self.get_parameter("publish_all_object_poses").value)
        obj_prefix = str(self.get_parameter("obj_prefix").value)
        num_objects = int(self.get_parameter("num_objects").value)

        planning_frame = str(self.get_parameter("planning_frame").value)
        sel_frame = str(self.get_parameter("selected_pose_frame").value).strip() or planning_frame
        selected = str(self.get_parameter("selected_object").value).strip()

        # 1) publish all objects YAML (quat + rpy)
        if publish_all:
            lines = []
            lines.append(f"frame_id: {planning_frame}")
            lines.append("objects:")
            for i in range(num_objects):
                oid = f"{obj_prefix}{i:03d}"
                if oid not in self.entity_pose:
                    continue
                (p, q) = self.entity_pose[oid]
                rr, rp, ry = quat_to_rpy(q)

                lines.append(f"  - name: {oid}")
                lines.append("    position:")
                lines.append(f"      x: {p[0]}")
                lines.append(f"      y: {p[1]}")
                lines.append(f"      z: {p[2]}")
                lines.append("    orientation:")
                lines.append(f"      x: {q[0]}")
                lines.append(f"      y: {q[1]}")
                lines.append(f"      z: {q[2]}")
                lines.append(f"      w: {q[3]}")
                lines.append("    rpy:")
                lines.append(f"      roll:  {rr}")
                lines.append(f"      pitch: {rp}")
                lines.append(f"      yaw:   {ry}")
            self.pub_all_poses.publish(String(data="\n".join(lines)))

        # 2) selected object: name + pose + rpy topics
        if selected:
            self.pub_sel_name.publish(String(data=selected))

            if selected in self.entity_pose:
                (p, q) = self.entity_pose[selected]
                rr, rp, ry = quat_to_rpy(q)

                ps = PoseStamped()
                ps.header.stamp = self.get_clock().now().to_msg()
                ps.header.frame_id = sel_frame
                ps.pose = to_pose_msg(p, q)
                self.pub_sel_pose.publish(ps)

                rpy_msg = Vector3Stamped()
                rpy_msg.header = ps.header
                rpy_msg.vector.x = rr
                rpy_msg.vector.y = rp
                rpy_msg.vector.z = ry
                self.pub_sel_rpy.publish(rpy_msg)

                ypr_deg = Vector3Stamped()
                ypr_deg.header = ps.header
                ypr_deg.vector.x = ry * 180.0 / math.pi  # yaw deg
                ypr_deg.vector.y = rp * 180.0 / math.pi  # pitch deg
                ypr_deg.vector.z = rr * 180.0 / math.pi  # roll deg
                self.pub_sel_ypr_deg.publish(ypr_deg)


def main():
    rclpy.init()
    node = GzEntitiesToMoveItScene()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
