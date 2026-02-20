#!/usr/bin/env python3
import math
import os
import random
import subprocess
import tempfile
from typing import Tuple, Optional

import rclpy
from rclpy.node import Node

from ros_gz_interfaces.srv import DeleteEntity
from ros_gz_interfaces.msg import Entity


# -----------------------------
# Math helpers
# -----------------------------
def rpy_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


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
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


def apply_group_rot_to_pose(
    q_group: Tuple[float, float, float, float],
    pos_xyz: Tuple[float, float, float],
    rpy: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    링크 프레임 기준:
      - 위치: p' = R_group * p
      - 자세: q' = q_group * q_local
    """
    px, py, pz = pos_xyz
    lr, lp, ly = rpy
    q_local = rpy_to_quat(lr, lp, ly)
    p_rot = quat_rotate_vec(q_group, (px, py, pz))
    q_out = quat_mul(q_group, q_local)
    r_out, p_out, y_out = quat_to_rpy(q_out)
    return (p_rot, (r_out, p_out, y_out))


# -----------------------------
# SDF helpers
# -----------------------------
def sdf_surface_block(mu: float, mu2: float, restitution: float, bounce_threshold: float = 1.0) -> str:
    return f"""
        <surface>
          <friction>
            <ode>
              <mu>{mu}</mu>
              <mu2>{mu2}</mu2>
            </ode>
          </friction>
          <bounce>
            <restitution_coefficient>{restitution}</restitution_coefficient>
            <threshold>{bounce_threshold}</threshold>
          </bounce>
        </surface>
    """


def sdf_debug_material(transparency: float) -> str:
    t = min(1.0, max(0.0, float(transparency)))
    return f"""
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <emissive>0 0 0 1</emissive>
          <transparency>{t}</transparency>
        </material>
    """


def box_inertia(mass: float, size_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = size_xyz
    ixx = (1.0 / 12.0) * mass * (y * y + z * z)
    iyy = (1.0 / 12.0) * mass * (x * x + z * z)
    izz = (1.0 / 12.0) * mass * (x * x + y * y)
    return ixx, iyy, izz


def ring_volume(r_out: float, r_in: float, h: float) -> float:
    r_out = max(0.0, float(r_out))
    r_in = max(0.0, float(r_in))
    h = max(0.0, float(h))
    if r_out <= r_in or h <= 0.0:
        return 0.0
    return math.pi * (r_out * r_out - r_in * r_in) * h


def sdf_ring_by_boxes(
    *,
    name_prefix: str,
    surface: str,
    r_out: float,
    r_in: float,
    height: float,
    z_center: float,
    segments: int,
    q_group: Tuple[float, float, float, float],
    debug_visual: bool,
    debug_transparency: float,
) -> Tuple[str, str]:
    """
    속이 빈 링(r_in~r_out)을 segments개의 박스로 원형 배치해서 근사.
    """
    r_out = max(1e-6, float(r_out))
    r_in = max(0.0, float(r_in))
    height = max(1e-6, float(height))
    segments = max(8, int(segments))

    r_in = min(r_in, r_out * 0.999)
    thickness = max(1e-6, r_out - r_in)
    r_mid = r_in + thickness * 0.5

    chord = 2.0 * r_mid * math.sin(math.pi / segments)
    seg_len = max(1e-6, chord * 1.08)  # 약간 겹치게
    seg_w = thickness
    seg_h = height

    col_xml = ""
    vis_xml = ""
    mat = sdf_debug_material(debug_transparency) if debug_visual else ""

    for i in range(segments):
        ang = (2.0 * math.pi * i) / segments
        x = r_mid * math.cos(ang)
        y = r_mid * math.sin(ang)
        yaw = ang + (math.pi * 0.5)

        p_local = (x, y, z_center)
        rpy_local = (0.0, 0.0, yaw)
        (p, rpy) = apply_group_rot_to_pose(q_group, p_local, rpy_local)

        col_xml += f"""
      <collision name="{name_prefix}_{i:02d}">
        <pose>{p[0]} {p[1]} {p[2]} {rpy[0]} {rpy[1]} {rpy[2]}</pose>
        <geometry>
          <box><size>{seg_len} {seg_w} {seg_h}</size></box>
        </geometry>
        {surface}
      </collision>
        """

        if debug_visual:
            vis_xml += f"""
      <visual name="debug_{name_prefix}_{i:02d}">
        <pose>{p[0]} {p[1]} {p[2]} {rpy[0]} {rpy[1]} {rpy[2]}</pose>
        <geometry>
          <box><size>{seg_len} {seg_w} {seg_h}</size></box>
        </geometry>
        {mat}
      </visual>
            """

    return col_xml, vis_xml


def sdf_tshape_ringbase_collisions_and_debug_visuals(
    *,
    surface: str,
    base_r_out: float,
    base_h: float,
    body_r_out: float,
    body_h: float,
    wall_t: float,
    segments: int,
    base_hole_clearance: float,
    q_group: Tuple[float, float, float, float],
    debug_visual: bool,
    debug_transparency: float,
) -> Tuple[str, str, Tuple[float, float, float], Tuple[float, float, float], float]:
    """
    ㅗ 모양 충돌:
      - 바닥: 링(속 빈 원형 디스크) => 바닥에도 몸통만큼 구멍
      - 몸통: 속 빈 튜브
    """
    base_r_out = max(1e-6, float(base_r_out))
    base_h = max(1e-6, float(base_h))
    body_r_out = max(1e-6, float(body_r_out))
    body_h = max(1e-6, float(body_h))
    wall_t = max(1e-6, float(wall_t))
    segments = max(8, int(segments))
    base_hole_clearance = max(0.0, float(base_hole_clearance))

    wall_t = min(wall_t, body_r_out * 0.95)
    body_r_in = max(0.0, body_r_out - wall_t)

    base_r_in = body_r_in + base_hole_clearance

    min_radial = max(0.002, wall_t)
    if base_r_out < base_r_in + min_radial:
        base_r_out = base_r_in + min_radial

    z_base = 0.0
    z_body = (base_h * 0.5) + (body_h * 0.5)

    v_base = ring_volume(base_r_out, base_r_in, base_h)
    v_body = ring_volume(body_r_out, body_r_in, body_h)
    v_sum = max(1e-12, v_base + v_body)
    z_com_local = (v_base * z_base + v_body * z_body) / v_sum
    com_rot = quat_rotate_vec(q_group, (0.0, 0.0, z_com_local))

    span = max(2.0 * base_r_out, 2.0 * body_r_out)
    inertia_bbox = (span, span, base_h + body_h)

    col_base, vis_base = sdf_ring_by_boxes(
        name_prefix="col_base_ring",
        surface=surface,
        r_out=base_r_out,
        r_in=base_r_in,
        height=base_h,
        z_center=z_base,
        segments=segments,
        q_group=q_group,
        debug_visual=debug_visual,
        debug_transparency=debug_transparency,
    )

    col_body, vis_body = sdf_ring_by_boxes(
        name_prefix="col_body_ring",
        surface=surface,
        r_out=body_r_out,
        r_in=body_r_in,
        height=body_h,
        z_center=z_body,
        segments=segments,
        q_group=q_group,
        debug_visual=debug_visual,
        debug_transparency=debug_transparency,
    )

    return (col_base + col_body), (vis_base + vis_body), inertia_bbox, com_rot, v_sum


def sdf_open_top_bin(
    name: str,
    inner_x: float,
    inner_y: float,
    wall_h: float,
    wall_thick: float,
    floor_thick: float,
    mu: float,
    mu2: float,
    restitution: float,
) -> str:
    outer_x = inner_x + 2.0 * wall_thick
    outer_y = inner_y + 2.0 * wall_thick

    floor_z = -floor_thick * 0.5
    wall_z = wall_h * 0.5

    surface = sdf_surface_block(mu=mu, mu2=mu2, restitution=restitution)

    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <pose>0 0 0 0 0 0</pose>
    <link name="bin_link">

      <collision name="floor_col">
        <pose>0 0 {floor_z} 0 0 0</pose>
        <geometry>
          <box><size>{outer_x} {outer_y} {floor_thick}</size></box>
        </geometry>
        {surface}
      </collision>
      <visual name="floor_vis">
        <pose>0 0 {floor_z} 0 0 0</pose>
        <geometry>
          <box><size>{outer_x} {outer_y} {floor_thick}</size></box>
        </geometry>
      </visual>

      <collision name="wall_px_col">
        <pose>{(inner_x*0.5 + wall_thick*0.5)} 0 {wall_z} 0 0 0</pose>
        <geometry>
          <box><size>{wall_thick} {outer_y} {wall_h}</size></box>
        </geometry>
        {surface}
      </collision>
      <visual name="wall_px_vis">
        <pose>{(inner_x*0.5 + wall_thick*0.5)} 0 {wall_z} 0 0 0</pose>
        <geometry>
          <box><size>{wall_thick} {outer_y} {wall_h}</size></box>
        </geometry>
      </visual>

      <collision name="wall_nx_col">
        <pose>{-(inner_x*0.5 + wall_thick*0.5)} 0 {wall_z} 0 0 0</pose>
        <geometry>
          <box><size>{wall_thick} {outer_y} {wall_h}</size></box>
        </geometry>
        {surface}
      </collision>
      <visual name="wall_nx_vis">
        <pose>{-(inner_x*0.5 + wall_thick*0.5)} 0 {wall_z} 0 0 0</pose>
        <geometry>
          <box><size>{wall_thick} {outer_y} {wall_h}</size></box>
        </geometry>
      </visual>

      <collision name="wall_py_col">
        <pose>0 {(inner_y*0.5 + wall_thick*0.5)} {wall_z} 0 0 0</pose>
        <geometry>
          <box><size>{outer_x} {wall_thick} {wall_h}</size></box>
        </geometry>
        {surface}
      </collision>
      <visual name="wall_py_vis">
        <pose>0 {(inner_y*0.5 + wall_thick*0.5)} {wall_z} 0 0 0</pose>
        <geometry>
          <box><size>{outer_x} {wall_thick} {wall_h}</size></box>
        </geometry>
      </visual>

      <collision name="wall_ny_col">
        <pose>0 {-(inner_y*0.5 + wall_thick*0.5)} {wall_z} 0 0 0</pose>
        <geometry>
          <box><size>{outer_x} {wall_thick} {wall_h}</size></box>
        </geometry>
        {surface}
      </collision>
      <visual name="wall_ny_vis">
        <pose>0 {-(inner_y*0.5 + wall_thick*0.5)} {wall_z} 0 0 0</pose>
        <geometry>
          <box><size>{outer_x} {wall_thick} {wall_h}</size></box>
        </geometry>
      </visual>

    </link>
  </model>
</sdf>
"""


def sdf_object_with_visual_mesh_and_tshape_collision(
    name: str,
    mesh_abs_path: str,
    scale_xyz: Tuple[float, float, float],
    mass: float,
    mu: float,
    mu2: float,
    restitution: float,
    col_base_outer_radius: float,
    col_base_h: float,
    col_cyl_outer_radius: float,
    col_cyl_outer_radius_scale: float,
    col_cyl_h: float,
    col_cyl_wall_t: float,
    col_cyl_segments: int,
    col_base_hole_clearance: float,
    col_group_rpy: Tuple[float, float, float],
    debug_collision_visual: bool,
    debug_collision_transparency: float,
) -> Tuple[str, float]:
    sx, sy, sz = scale_xyz
    scale_str = f"{sx} {sy} {sz}"
    mesh_uri = f"file://{mesh_abs_path}"

    surface = sdf_surface_block(mu=mu, mu2=mu2, restitution=restitution)
    q_group = rpy_to_quat(col_group_rpy[0], col_group_rpy[1], col_group_rpy[2])

    body_r_out = float(col_cyl_outer_radius) * float(col_cyl_outer_radius_scale)

    col_xml, vis_debug_xml, inertia_bbox, com_xyz, volume = sdf_tshape_ringbase_collisions_and_debug_visuals(
        surface=surface,
        base_r_out=col_base_outer_radius,
        base_h=col_base_h,
        body_r_out=body_r_out,
        body_h=col_cyl_h,
        wall_t=col_cyl_wall_t,
        segments=col_cyl_segments,
        base_hole_clearance=col_base_hole_clearance,
        q_group=q_group,
        debug_visual=debug_collision_visual,
        debug_transparency=debug_collision_transparency,
    )

    ixx, iyy, izz = box_inertia(mass, inertia_bbox)

    sdf_xml = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <inertial>
        <pose>{com_xyz[0]} {com_xyz[1]} {com_xyz[2]} 0 0 0</pose>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx}</ixx><iyy>{iyy}</iyy><izz>{izz}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>

      <visual name="vis_mesh">
        <geometry>
          <mesh>
            <uri>{mesh_uri}</uri>
            <scale>{scale_str}</scale>
          </mesh>
        </geometry>
      </visual>

      {vis_debug_xml}
      {col_xml}

    </link>
  </model>
</sdf>
"""
    return sdf_xml, volume


# -----------------------------
# Spawner (Gazebo Sim)
# -----------------------------
class GzSimBinAndObjectsSpawner(Node):
    def __init__(self):
        super().__init__("gzsim_bin_and_objects_spawner")

        # ======= Parameters =======
        self.declare_parameter("world_name", "empty")

        self.declare_parameter("mesh_path", "/home/minsoo/bin_picking/src/utils/object.stl")
        self.declare_parameter("mesh_scale", [0.001, 0.001, 0.001])

        self.declare_parameter("num_objects", 42)
        self.declare_parameter("random_seed", 42)

        # Bin pose in world
        self.declare_parameter("bin_xyz", [0.8, 0.0, 0.0])
        self.declare_parameter("bin_rpy", [0.0, 0.0, 0.0])

        # Bin geometry
        self.declare_parameter("bin_inner_xy", [0.40, 0.30])
        self.declare_parameter("bin_wall_h", 0.20)
        self.declare_parameter("bin_wall_thick", 0.01)
        self.declare_parameter("bin_floor_thick", 0.01)

        # Spawn region
        self.declare_parameter("spawn_margin", 0.03)
        self.declare_parameter("spawn_z_range", [0.15, 0.30])

        # ---- 마찰/밀도(요청 반영) ----
        self.declare_parameter("friction_mu", 0.12)
        self.declare_parameter("friction_mu2", 0.10)
        self.declare_parameter("restitution", 0.02)

        # 8 g/cm^3 = 8000 kg/m^3
        self.declare_parameter("object_mass", -1.0)     # <=0이면 density로 계산
        self.declare_parameter("object_density", 8000.0)

        # ---- spawn/delete policy ----
        self.declare_parameter("spawn_bin", True)
        self.declare_parameter("delete_existing_bin", True)
        self.declare_parameter("delete_existing_objects", True)

        self.declare_parameter("bin_name", "bin_001")
        self.declare_parameter("object_name_prefix", "obj_")

        # ✅ 삭제 범위 확장(디폴트 obj_000~obj_199)
        self.declare_parameter("delete_object_index_min", 0)
        self.declare_parameter("delete_object_index_max", 199)

        # ---- Collision(ㅗ) params ----
        self.declare_parameter("col_base_outer_radius", 0.025)
        self.declare_parameter("col_base_h", 0.006)

        self.declare_parameter("col_cyl_outer_radius", 0.015)
        self.declare_parameter("col_cyl_outer_radius_scale", 1.45)  # 지름 +30%
        self.declare_parameter("col_cyl_h", 0.030)
        self.declare_parameter("col_cyl_wall_t", 0.002)             # 2mm
        self.declare_parameter("col_cyl_segments", 16)

        self.declare_parameter("col_base_hole_clearance", 0.0)

        # collision-only rotation: Y +90deg
        self.declare_parameter("col_group_rpy", [0.0, math.pi / 2.0, 0.0])

        # collision visualization
        self.declare_parameter("debug_collision_visual", True)
        self.declare_parameter("debug_collision_transparency", 0.55)

        self.spawn_all()

    # ---------- ROS delete service ----------
    def _delete_client(self) -> Optional[rclpy.client.Client]:
        world = str(self.get_parameter("world_name").value)
        srv_name = f"/world/{world}/remove"
        client = self.create_client(DeleteEntity, srv_name)
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                f"[delete] ROS service '{srv_name}'가 보이지 않습니다. "
                f"지금처럼 수동 호출이 된다면, 이 노드를 실행한 쉘 환경에서 "
                f"RMW/DOMAIN/네임스페이스가 동일한지 확인해주세요."
            )
            return None
        return client

    def _delete_model(self, client: rclpy.client.Client, name: str, timeout_sec: float = 2.0) -> bool:
        req = DeleteEntity.Request()
        req.entity = Entity()
        req.entity.id = 0
        req.entity.name = name
        req.entity.type = 2  # MODEL

        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout_sec)
        if fut.done() and fut.result() is not None:
            ok = bool(fut.result().success)
            if ok:
                self.get_logger().info(f"[delete] OK: {name}")
            return ok
        return False

    # ---------- shell helpers ----------
    def _run_cmd(self, cmd: list, warn_only: bool = False) -> bool:
        try:
            out = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if out.stdout.strip():
                self.get_logger().info(out.stdout.strip())
            if out.stderr.strip():
                self.get_logger().warn(out.stderr.strip())
            return True
        except subprocess.CalledProcessError as e:
            msg = (e.stdout or "") + "\n" + (e.stderr or "")
            if warn_only:
                self.get_logger().warn(f"Command failed (ignored): {' '.join(cmd)}\n{msg}")
                return False
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{msg}") from e

    def _run_ros_gz_create(
        self,
        name: str,
        sdf_xml: str,
        pose_xyzrpy: Tuple[float, float, float, float, float, float],
        warn_only: bool = False,
    ):
        world = str(self.get_parameter("world_name").value)
        x, y, z, r, p, yw = pose_xyzrpy

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sdf") as f:
            f.write(sdf_xml)
            sdf_path = f.name

        cmd = [
            "ros2", "run", "ros_gz_sim", "create",
            "-world", world,
            "-name", name,
            "-file", sdf_path,
            "-x", str(x), "-y", str(y), "-z", str(z),
            "-R", str(r), "-P", str(p), "-Y", str(yw),
        ]

        try:
            self.get_logger().info(f"[ros_gz_sim/create] Spawning {name} in world={world}")
            self._run_cmd(cmd, warn_only=warn_only)
        finally:
            try:
                os.remove(sdf_path)
            except OSError:
                pass

    def _compute_object_mass(self, object_mass: float, density: float, volume: float) -> float:
        if object_mass > 0.0:
            return object_mass
        return max(1e-6, density * max(1e-12, volume))

    def spawn_all(self):
        random_seed = int(self.get_parameter("random_seed").value)
        random.seed(random_seed)

        world = str(self.get_parameter("world_name").value)

        mesh_path = str(self.get_parameter("mesh_path").value)
        scale = self.get_parameter("mesh_scale").value
        scale_xyz = (float(scale[0]), float(scale[1]), float(scale[2]))

        num_objects = int(self.get_parameter("num_objects").value)

        # bin pose
        bin_xyz = self.get_parameter("bin_xyz").value
        bin_rpy = self.get_parameter("bin_rpy").value
        bx, by, bz = float(bin_xyz[0]), float(bin_xyz[1]), float(bin_xyz[2])
        br, bp, byaw = float(bin_rpy[0]), float(bin_rpy[1]), float(bin_rpy[2])

        # bin geom
        inner_xy = self.get_parameter("bin_inner_xy").value
        inner_x, inner_y = float(inner_xy[0]), float(inner_xy[1])
        wall_h = float(self.get_parameter("bin_wall_h").value)
        wall_t = float(self.get_parameter("bin_wall_thick").value)
        floor_t = float(self.get_parameter("bin_floor_thick").value)

        margin = float(self.get_parameter("spawn_margin").value)
        z_range = self.get_parameter("spawn_z_range").value
        zmin, zmax = float(z_range[0]), float(z_range[1])

        mu = float(self.get_parameter("friction_mu").value)
        mu2 = float(self.get_parameter("friction_mu2").value)
        restitution = float(self.get_parameter("restitution").value)

        object_mass_param = float(self.get_parameter("object_mass").value)
        object_density = float(self.get_parameter("object_density").value)

        spawn_bin = bool(self.get_parameter("spawn_bin").value)
        delete_existing_bin = bool(self.get_parameter("delete_existing_bin").value)
        delete_existing_objects = bool(self.get_parameter("delete_existing_objects").value)

        bin_name = str(self.get_parameter("bin_name").value)
        obj_prefix = str(self.get_parameter("object_name_prefix").value)

        del_min = int(self.get_parameter("delete_object_index_min").value)
        del_max = int(self.get_parameter("delete_object_index_max").value)
        if del_min < 0:
            del_min = 0
        if del_max < del_min:
            del_max = del_min

        # collision params
        col_base_outer_radius = float(self.get_parameter("col_base_outer_radius").value)
        col_base_h = float(self.get_parameter("col_base_h").value)
        col_cyl_outer_radius = float(self.get_parameter("col_cyl_outer_radius").value)
        col_cyl_outer_radius_scale = float(self.get_parameter("col_cyl_outer_radius_scale").value)
        col_cyl_h = float(self.get_parameter("col_cyl_h").value)
        col_cyl_wall_t = float(self.get_parameter("col_cyl_wall_t").value)
        col_cyl_segments = int(self.get_parameter("col_cyl_segments").value)
        col_base_hole_clearance = float(self.get_parameter("col_base_hole_clearance").value)

        col_group_rpy_list = self.get_parameter("col_group_rpy").value
        col_group_rpy = (float(col_group_rpy_list[0]), float(col_group_rpy_list[1]), float(col_group_rpy_list[2]))

        debug_collision_visual = bool(self.get_parameter("debug_collision_visual").value)
        debug_collision_transparency = float(self.get_parameter("debug_collision_transparency").value)

        self.get_logger().info(
            f"[Config] world={world} spawn_bin={spawn_bin} del_bin={delete_existing_bin} del_obj={delete_existing_objects} "
            f"del_range={obj_prefix}{del_min:03d}~{obj_prefix}{del_max:03d} "
            f"friction(mu,mu2)=({mu},{mu2}) density={object_density} object_mass={object_mass_param}"
        )

        del_client = self._delete_client() if (delete_existing_objects or delete_existing_bin) else None

        # ---- Delete objects with wide range (default 000~199) ----
        if delete_existing_objects and del_client is not None:
            deleted_cnt = 0
            for idx in range(del_min, del_max + 1):
                if self._delete_model(del_client, f"{obj_prefix}{idx:03d}", timeout_sec=2.0):
                    deleted_cnt += 1
            self.get_logger().info(f"[delete] objects deleted: {deleted_cnt} (range {del_min}~{del_max})")

        # ---- Delete bin (요청 반영) ----
        if spawn_bin and delete_existing_bin and del_client is not None:
            self._delete_model(del_client, bin_name, timeout_sec=2.0)

        # ---- Spawn bin (박스 생성) ----
        if spawn_bin:
            bin_sdf = sdf_open_top_bin(
                name=bin_name,
                inner_x=inner_x,
                inner_y=inner_y,
                wall_h=wall_h,
                wall_thick=wall_t,
                floor_thick=floor_t,
                mu=mu,
                mu2=mu2,
                restitution=restitution,
            )
            self._run_ros_gz_create(bin_name, bin_sdf, (bx, by, bz, br, bp, byaw), warn_only=False)

        # ---- Spawn objects ----
        xmin = -(inner_x * 0.5) + margin
        xmax = +(inner_x * 0.5) - margin
        ymin = -(inner_y * 0.5) + margin
        ymax = +(inner_y * 0.5) - margin

        q_bin = rpy_to_quat(br, bp, byaw)

        volume_used = None
        mass_used = None

        for i in range(num_objects):
            obj_name = f"{obj_prefix}{i:03d}"

            mass_now = object_mass_param if object_mass_param > 0.0 else 0.1
            obj_sdf, vol = sdf_object_with_visual_mesh_and_tshape_collision(
                name=obj_name,
                mesh_abs_path=mesh_path,
                scale_xyz=scale_xyz,
                mass=mass_now,
                mu=mu,
                mu2=mu2,
                restitution=restitution,
                col_base_outer_radius=col_base_outer_radius,
                col_base_h=col_base_h,
                col_cyl_outer_radius=col_cyl_outer_radius,
                col_cyl_outer_radius_scale=col_cyl_outer_radius_scale,
                col_cyl_h=col_cyl_h,
                col_cyl_wall_t=col_cyl_wall_t,
                col_cyl_segments=col_cyl_segments,
                col_base_hole_clearance=col_base_hole_clearance,
                col_group_rpy=col_group_rpy,
                debug_collision_visual=debug_collision_visual,
                debug_collision_transparency=debug_collision_transparency,
            )

            if volume_used is None:
                volume_used = vol
                mass_used = self._compute_object_mass(object_mass_param, object_density, volume_used)
                self.get_logger().info(f"[Mass] volume={volume_used:.6e} m^3 -> mass={mass_used:.4f} kg")

            if object_mass_param <= 0.0:
                obj_sdf, _ = sdf_object_with_visual_mesh_and_tshape_collision(
                    name=obj_name,
                    mesh_abs_path=mesh_path,
                    scale_xyz=scale_xyz,
                    mass=mass_used,
                    mu=mu,
                    mu2=mu2,
                    restitution=restitution,
                    col_base_outer_radius=col_base_outer_radius,
                    col_base_h=col_base_h,
                    col_cyl_outer_radius=col_cyl_outer_radius,
                    col_cyl_outer_radius_scale=col_cyl_outer_radius_scale,
                    col_cyl_h=col_cyl_h,
                    col_cyl_wall_t=col_cyl_wall_t,
                    col_cyl_segments=col_cyl_segments,
                    col_base_hole_clearance=col_base_hole_clearance,
                    col_group_rpy=col_group_rpy,
                    debug_collision_visual=debug_collision_visual,
                    debug_collision_transparency=debug_collision_transparency,
                )

            lx = random.uniform(xmin, xmax)
            ly = random.uniform(ymin, ymax)
            lz = random.uniform(zmin, zmax)

            dx, dy, dz = quat_rotate_vec(q_bin, (lx, ly, lz))
            wx = bx + dx
            wy = by + dy
            wz = bz + dz

            yaw_local = random.uniform(-math.pi, math.pi)
            q_obj = quat_mul(q_bin, rpy_to_quat(0.0, 0.0, yaw_local))
            oroll, opitch, oyaw = quat_to_rpy(q_obj)

            self._run_ros_gz_create(obj_name, obj_sdf, (wx, wy, wz, oroll, opitch, oyaw), warn_only=False)

        self.get_logger().info("Done. (bin + objects deleted & respawned; objects deleted by wide range)")


def main():
    rclpy.init()
    node = None
    try:
        node = GzSimBinAndObjectsSpawner()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
