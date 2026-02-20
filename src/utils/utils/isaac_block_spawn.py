import asyncio
import random
import math
import omni.usd
import omni.timeline
from pxr import Usd, UsdGeom, Gf, UsdPhysics, UsdShade, Sdf

# PhysX 스키마 확인
try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None

# =======================
# 사용자 설정
# =======================
OBJECT_USD = "/home/minsoo/isaac_bin_picking/assets/objects/USD_converted/object.usd"

NUM_OBJECTS  = 20
MODEL_SCALE  = 0.001
BIN_CENTER = (0.0, 0.8, 0.0)

BIN_INNER_X = 0.45
BIN_INNER_Y = 0.272
BIN_WALL_H  = 0.12
BIN_THICK   = 0.02

MASS = 0.12
GRAVITY = 9.81

SPACING_X = 0.06
SPACING_Y = 0.06
SPACING_Z = 0.08
POS_JITTER = 0.015
START_Z = BIN_CENTER[2] + BIN_THICK + 0.15

MU_S, MU_D, REST, DENSITY = 0.08, 0.06, 0.02, 7800.0
PHYS_MAT_PATH = "/World/physicsMaterials/PTFE_contact"

# =======================
# 유틸리티 함수
# =======================
def get_stage():
    return omni.usd.get_context().get_stage()

def set_xform_ops(prim, t=(0,0,0), q=(1,0,0,0), s=(1,1,1)):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    
    op_t = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    op_t.Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))

    op_r = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    op_r.Set(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])))

    op_s = xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    op_s.Set(Gf.Vec3d(float(s[0]), float(s[1]), float(s[2])))

def rand_quat_wxyz():
    x, y, z, w = [random.random() for _ in range(4)]
    n = math.sqrt(x*x + y*y + z*z + w*w)
    return (w/n, x/n, y/n, z/n)

def remove_if_exists(path: str):
    stage = get_stage()
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        stage.RemovePrim(path)

def make_static_collider_cube(path, size_xyz, pos_xyz):
    stage = get_stage()
    prim = stage.DefinePrim(path, "Cube")
    set_xform_ops(prim, t=pos_xyz, s=(size_xyz[0]/2, size_xyz[1]/2, size_xyz[2]/2))
    UsdPhysics.CollisionAPI.Apply(prim)
    return prim

def get_or_create_physics_material():
    stage = get_stage()
    if not stage.GetPrimAtPath("/World/physicsMaterials").IsValid():
        stage.DefinePrim("/World/physicsMaterials", "Scope")

    mat = UsdShade.Material.Define(stage, PHYS_MAT_PATH)
    mat_prim = mat.GetPrim()
    mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
    mat_api.CreateStaticFrictionAttr().Set(float(MU_S))
    mat_api.CreateDynamicFrictionAttr().Set(float(MU_D))
    mat_api.CreateRestitutionAttr().Set(float(REST))
    mat_api.CreateDensityAttr().Set(float(DENSITY))
    return mat

def bind_physics_material(target_prim, material: UsdShade.Material):
    binding_api = UsdShade.MaterialBindingAPI.Apply(target_prim)
    binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

def apply_convex_hull_colliders_recursively(root_prim) -> int:
    cnt = 0
    for p in Usd.PrimRange(root_prim):
        if p.IsA(UsdGeom.Mesh):
            if not p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(p)
            mca = UsdPhysics.MeshCollisionAPI.Apply(p)
            mca.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
            cnt += 1
    return cnt

def disable_nested_rigid_bodies(subtree_root_prim) -> int:
    removed_count = 0
    for p in Usd.PrimRange(subtree_root_prim):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            p.RemoveAPI(UsdPhysics.RigidBodyAPI)
            removed_count += 1
        if p.HasAPI(UsdPhysics.MassAPI):
            p.RemoveAPI(UsdPhysics.MassAPI)
        if PhysxSchema and p.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            p.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
            removed_count += 1
    return removed_count


# =======================
# 메인 실행 로직 (수정됨)
# =======================
async def run_scenario():
    print("🚀 시나리오 초기화 중...")
    
    timeline = omni.timeline.get_timeline_interface()
    stage = get_stage()
    
    # ✅ 1. 시뮬레이션 완전 정지 (중요: 경고 메시지 해결의 핵심)
    if timeline.is_playing():
        timeline.stop()
        # 정지 후 물리 엔진이 리셋될 시간을 확보하기 위해 대기
        await asyncio.sleep(0.1)
    
    # 2. 기존 오브젝트 정리
    remove_if_exists("/World/bin")
    to_remove = [p.GetPath().pathString for p in stage.Traverse() if "/obj_" in p.GetPath().pathString]
    
    for p in sorted(set(to_remove), key=len, reverse=True):
        remove_if_exists(p)
        
    # 삭제 처리가 USD 스테이지에 반영되도록 한 틱 대기
    await asyncio.sleep(0.05) 

    # 3. PhysicsScene 생성
    if not stage.GetPrimAtPath("/World/physicsScene").IsValid():
        scene_prim = stage.DefinePrim("/World/physicsScene", "PhysicsScene")
        scene = UsdPhysics.Scene(scene_prim)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(float(GRAVITY))

    # 4. Material 준비
    ptfe_mat = get_or_create_physics_material()

    # 5. 바구니 생성
    bin_root = stage.DefinePrim("/World/bin", "Xform")
    cx, cy, cz = BIN_CENTER

    make_static_collider_cube("/World/bin/floor", (BIN_INNER_X + 2*BIN_THICK, BIN_INNER_Y + 2*BIN_THICK, BIN_THICK), (cx, cy, cz + BIN_THICK/2))
    
    wz = cz + BIN_THICK + BIN_WALL_H/2
    make_static_collider_cube("/World/bin/wall_px", (BIN_THICK, BIN_INNER_Y + 2*BIN_THICK, BIN_WALL_H), (cx + BIN_INNER_X/2 + BIN_THICK/2, cy, wz))
    make_static_collider_cube("/World/bin/wall_nx", (BIN_THICK, BIN_INNER_Y + 2*BIN_THICK, BIN_WALL_H), (cx - BIN_INNER_X/2 - BIN_THICK/2, cy, wz))
    make_static_collider_cube("/World/bin/wall_py", (BIN_INNER_X + 2*BIN_THICK, BIN_THICK, BIN_WALL_H), (cx, cy + BIN_INNER_Y/2 + BIN_THICK/2, wz))
    make_static_collider_cube("/World/bin/wall_ny", (BIN_INNER_X + 2*BIN_THICK, BIN_THICK, BIN_WALL_H), (cx, cy - BIN_INNER_Y/2 - BIN_THICK/2, wz))

    bind_physics_material(bin_root, ptfe_mat)

    # 6. 물체 생성 루프
    cols = max(1, int(BIN_INNER_X / SPACING_X))
    rows = max(1, int(BIN_INNER_Y / SPACING_Y))
    per_layer = cols * rows
    if per_layer == 0: per_layer = 1

    ox, oy = (cols - 1) * SPACING_X / 2.0, (rows - 1) * SPACING_Y / 2.0
    
    spawned = 0
    total_mesh_colliders = 0

    for i in range(NUM_OBJECTS):
        root_path = f"/World/obj_{i}"
        geom_path = root_path + "/geom"

        root_prim = stage.DefinePrim(root_path, "Xform")
        geom_prim = stage.DefinePrim(geom_path, "Xform")

        refs = geom_prim.GetReferences()
        refs.ClearReferences()
        refs.AddReference(OBJECT_USD)
        
        disable_nested_rigid_bodies(geom_prim)

        layer = i // per_layer
        rem   = i % per_layer
        c, r  = rem % cols, rem // cols

        tx = cx + (c * SPACING_X - ox) + random.uniform(-POS_JITTER, POS_JITTER)
        ty = cy + (r * SPACING_Y - oy) + random.uniform(-POS_JITTER, POS_JITTER)
        tz = START_Z + (layer * SPACING_Z) + random.uniform(0.0, 0.02)

        set_xform_ops(root_prim, t=(tx, ty, tz), q=rand_quat_wxyz())
        set_xform_ops(geom_prim, s=(MODEL_SCALE, MODEL_SCALE, MODEL_SCALE))

        rb = UsdPhysics.RigidBodyAPI.Apply(root_prim)
        rb.CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(float(MASS))

        total_mesh_colliders += apply_convex_hull_colliders_recursively(geom_prim)
        bind_physics_material(root_prim, ptfe_mat)
        spawned += 1

    print(f"✅ 생성 완료: {spawned}개")
    
    # ✅ 7. 모든 준비가 끝나면 시뮬레이션 시작
    timeline.play()

# 비동기 실행
asyncio.ensure_future(run_scenario())