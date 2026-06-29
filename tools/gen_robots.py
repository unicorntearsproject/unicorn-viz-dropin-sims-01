"""Generate procedural, license-free robot USD scenes for sims-01.

Authors two articulated robots as USD (Y-up, looping rigid joint animation via
time-sampled Xforms) and writes them into ``../scenes/``. The geometry is
authored from scratch here, so the output carries no third-party licence and is
safe to commit. Re-run to regenerate:

    python drop-ins/sims-01/tools/gen_robots.py

Requires OpenUSD (``pip install -r drop-ins/sims-01/requirements.txt``).
"""
from __future__ import annotations

import math
from pathlib import Path

from pxr import Usd, UsdGeom, Gf

_SCENES = Path(__file__).resolve().parents[1] / 'scenes'
_FPS = 30.0
_FRAMES = 120          # 4-second loop
_TAU = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Geometry primitives (returned as points + per-face vertex counts/indices)
# ---------------------------------------------------------------------------

def _box(cx, cy, cz, sx, sy, sz):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    pts = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
           (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    pts = [(x + cx, y + cy, z + cz) for (x, y, z) in pts]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    counts = [4] * 6
    idx = [i for f in faces for i in f]
    return pts, counts, idx


def _cylinder(cx, cy, cz, radius, height, segments=16):
    hy = height / 2
    pts = []
    for i in range(segments):
        a = _TAU * i / segments
        x, z = radius * math.cos(a), radius * math.sin(a)
        pts.append((cx + x, cy - hy, cz + z))   # bottom ring
        pts.append((cx + x, cy + hy, cz + z))   # top ring
    bottom_c = len(pts); pts.append((cx, cy - hy, cz))
    top_c = len(pts); pts.append((cx, cy + hy, cz))
    counts, idx = [], []
    for i in range(segments):
        b0, t0 = 2 * i, 2 * i + 1
        b1, t1 = 2 * ((i + 1) % segments), 2 * ((i + 1) % segments) + 1
        counts.append(4); idx += [b0, b1, t1, t0]          # side quad
        counts.append(3); idx += [bottom_c, b1, b0]        # bottom fan
        counts.append(3); idx += [top_c, t0, t1]           # top fan
    return pts, counts, idx


def _add_mesh(stage, path, geo):
    pts, counts, idx = geo
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return mesh


def _xform(stage, path):
    return UsdGeom.Xform.Define(stage, path)


def _anim_rotate(xform, axis_values):
    """Author a time-sampled rotateXYZ op. ``axis_values`` is fn(frame)->(x,y,z)."""
    op = xform.AddRotateXYZOp()
    for f in range(_FRAMES + 1):
        op.Set(Gf.Vec3f(*axis_values(f)), Usd.TimeCode(float(f)))


def _anim_translate(xform, fn):
    op = xform.AddTranslateOp()
    for f in range(_FRAMES + 1):
        op.Set(Gf.Vec3d(*fn(f)), Usd.TimeCode(float(f)))


def _new_stage(default_prim_name):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(float(_FRAMES))
    stage.SetTimeCodesPerSecond(_FPS)
    root = _xform(stage, f'/{default_prim_name}')
    stage.SetDefaultPrim(root.GetPrim())
    return stage, root


def _phase(f):
    return _TAU * f / _FRAMES   # 0..2π over the loop (seamless)


# ---------------------------------------------------------------------------
# Robot 1 — 5-axis industrial arm doing a pick/place sweep
# ---------------------------------------------------------------------------

def build_robot_arm(out: Path):
    stage, root = _new_stage('RobotArm')
    _add_mesh(stage, '/RobotArm/Base', _cylinder(0, 0.25, 0, 1.1, 0.5, 24))

    shoulder = _xform(stage, '/RobotArm/Shoulder')
    shoulder.AddTranslateOp().Set(Gf.Vec3d(0, 0.5, 0))
    _anim_rotate(shoulder, lambda f: (0, 70 * math.sin(_phase(f)), 0))
    _add_mesh(stage, '/RobotArm/Shoulder/Hub', _cylinder(0, 0.2, 0, 0.7, 0.6, 20))

    upper = _xform(stage, '/RobotArm/Shoulder/Upper')
    upper.AddTranslateOp().Set(Gf.Vec3d(0, 0.4, 0))
    _anim_rotate(upper, lambda f: (35 + 35 * math.sin(_phase(f) + 0.6), 0, 0))
    _add_mesh(stage, '/RobotArm/Shoulder/Upper/Link', _box(0, 1.1, 0, 0.5, 2.4, 0.5))

    elbow = _xform(stage, '/RobotArm/Shoulder/Upper/Elbow')
    elbow.AddTranslateOp().Set(Gf.Vec3d(0, 2.2, 0))
    _anim_rotate(elbow, lambda f: (-60 - 45 * math.sin(_phase(f) + 1.2), 0, 0))
    _add_mesh(stage, '/RobotArm/Shoulder/Upper/Elbow/Fore',
              _box(0, 0.9, 0, 0.4, 1.9, 0.4))

    wrist = _xform(stage, '/RobotArm/Shoulder/Upper/Elbow/Wrist')
    wrist.AddTranslateOp().Set(Gf.Vec3d(0, 1.8, 0))
    _anim_rotate(wrist, lambda f: (0, 0, 90 * math.sin(_phase(f) * 2.0)))
    _add_mesh(stage, '/RobotArm/Shoulder/Upper/Elbow/Wrist/Jaw1',
              _box(0.18, 0.35, 0, 0.12, 0.5, 0.3))
    _add_mesh(stage, '/RobotArm/Shoulder/Upper/Elbow/Wrist/Jaw2',
              _box(-0.18, 0.35, 0, 0.12, 0.5, 0.3))

    stage.GetRootLayer().Export(str(out))
    print('wrote', out)


# ---------------------------------------------------------------------------
# Robot 2 — quadcopter drone: spinning rotors, bob / yaw / tilt
# ---------------------------------------------------------------------------

def build_drone(out: Path):
    stage, root = _new_stage('Drone')
    body = _xform(stage, '/Drone/Body')
    _anim_translate(body, lambda f: (0, 0.4 * math.sin(_phase(f)), 0))
    _anim_rotate(body, lambda f: (8 * math.sin(_phase(f) + 1.0),
                                  360 * f / _FRAMES,            # full yaw spin
                                  6 * math.cos(_phase(f))))
    _add_mesh(stage, '/Drone/Body/Hull', _box(0, 0, 0, 1.4, 0.35, 1.4))
    _add_mesh(stage, '/Drone/Body/Dome', _cylinder(0, 0.25, 0, 0.4, 0.25, 16))

    arms = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for i, (sx, sz) in enumerate(arms):
        ax, az = sx * 1.1, sz * 1.1
        _add_mesh(stage, f'/Drone/Body/Arm{i}',
                  _box(ax / 2, 0, az / 2, abs(ax) + 0.2, 0.16, 0.22 if sx else 0.22))
        rotor = _xform(stage, f'/Drone/Body/Rotor{i}')
        rotor.AddTranslateOp().Set(Gf.Vec3d(ax, 0.18, az))
        # Fast spin; alternate direction per rotor like a real quad.
        spin = 1 if (i % 2 == 0) else -1
        _anim_rotate(rotor, lambda f, s=spin: (0, s * 8 * 360 * f / _FRAMES, 0))
        _add_mesh(stage, f'/Drone/Body/Rotor{i}/Blade',
                  _box(0, 0, 0, 1.5, 0.04, 0.16))
        _add_mesh(stage, f'/Drone/Body/Rotor{i}/Hub',
                  _cylinder(0, 0, 0, 0.12, 0.12, 10))

    stage.GetRootLayer().Export(str(out))
    print('wrote', out)


# ---------------------------------------------------------------------------
# Legged walkers (quadruped + hexapod) — phased stepping gaits
# ---------------------------------------------------------------------------

def _leg(stage, parent, name, hip_pos, phase,
         swing=28.0, lift=36.0, femur=1.2, tibia=1.2):
    """A 2-segment leg that swings fore/aft at the hip and lifts via the knee.

    ``phase`` offsets this leg in the gait cycle; the knee bends only during the
    forward half-swing (``max(0, sin)``) so the foot clears the ground.
    """
    hip = _xform(stage, f'{parent}/{name}')
    hip.AddTranslateOp().Set(Gf.Vec3d(*hip_pos))
    _anim_rotate(hip, lambda f: (swing * math.sin(_phase(f) + phase), 0, 0))
    _add_mesh(stage, f'{parent}/{name}/Femur', _box(0, -femur / 2, 0, 0.24, femur, 0.24))
    knee = _xform(stage, f'{parent}/{name}/Knee')
    knee.AddTranslateOp().Set(Gf.Vec3d(0, -femur, 0))
    _anim_rotate(knee, lambda f: (lift * max(0.0, math.sin(_phase(f) + phase)), 0, 0))
    _add_mesh(stage, f'{parent}/{name}/Knee/Tibia', _box(0, -tibia / 2, 0, 0.18, tibia, 0.18))
    _add_mesh(stage, f'{parent}/{name}/Knee/Foot', _cylinder(0, -tibia, 0, 0.17, 0.12, 10))


def build_quadruped(out: Path):
    stage, root = _new_stage('Quadruped')
    body = _xform(stage, '/Quadruped/Body')
    # Bob at twice the step frequency; legs hang from the torso underside.
    _anim_translate(body, lambda f: (0, 2.4 + 0.12 * math.sin(2 * _phase(f)), 0))
    _add_mesh(stage, '/Quadruped/Body/Torso', _box(0, 0, 0, 1.6, 0.7, 2.8))
    _add_mesh(stage, '/Quadruped/Body/Head', _box(0, 0.25, 1.7, 0.7, 0.6, 0.6))
    _add_mesh(stage, '/Quadruped/Body/Tail', _box(0, 0.1, -1.7, 0.2, 0.2, 0.6))
    # Trot gait: diagonal pairs move together (FL+BR, then FR+BL).
    pi = math.pi
    legs = [('FL', -0.8, 1.2, 0.0), ('FR', 0.8, 1.2, pi),
            ('BL', -0.8, -1.2, pi), ('BR', 0.8, -1.2, 0.0)]
    for nm, x, z, ph in legs:
        _leg(stage, '/Quadruped/Body', nm, (x, -0.25, z), ph)
    stage.GetRootLayer().Export(str(out))
    print('wrote', out)


def build_hexapod(out: Path):
    stage, root = _new_stage('Hexapod')
    body = _xform(stage, '/Hexapod/Body')
    _anim_translate(body, lambda f: (0, 1.9 + 0.1 * math.sin(2 * _phase(f)), 0))
    _add_mesh(stage, '/Hexapod/Body/Carapace', _cylinder(0, 0, 0, 1.5, 0.5, 6))
    _add_mesh(stage, '/Hexapod/Body/Head', _cylinder(0, 0, 1.4, 0.45, 0.4, 8))
    # Tripod gait: two alternating tripods 180° out of phase.
    pi = math.pi
    legs = [('L0', -1.35, 1.0, 0.0), ('L1', -1.35, 0.0, pi), ('L2', -1.35, -1.0, 0.0),
            ('R0', 1.35, 1.0, pi), ('R1', 1.35, 0.0, 0.0), ('R2', 1.35, -1.0, pi)]
    for nm, x, z, ph in legs:
        _leg(stage, '/Hexapod/Body', nm, (x, 0.0, z), ph,
             swing=24.0, lift=32.0, femur=1.0, tibia=1.1)
    stage.GetRootLayer().Export(str(out))
    print('wrote', out)


if __name__ == '__main__':
    _SCENES.mkdir(parents=True, exist_ok=True)
    build_robot_arm(_SCENES / 'robot_arm.usda')
    build_drone(_SCENES / 'robot_drone.usda')
    build_quadruped(_SCENES / 'robot_quadruped.usda')
    build_hexapod(_SCENES / 'robot_hexapod.usda')
