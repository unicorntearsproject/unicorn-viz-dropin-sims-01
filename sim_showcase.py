"""Sim Showcase drop-in effect.

Renders a gallery of OpenUSD scenes and IsaacLab-compatible models as
full-screen, audio-reactive 3D visuals using a GL-native rasterizer built
on top of the moderngl context already owned by unicorn-viz.

Scene sources:
- USDA/USDC/USDZ files anywhere inside `drop-ins/sims-01/scenes/`
- `[effects.SimShowcase].scene_dir` overrides the default directory

For each scene the loader:
1. Opens the USD stage with `pxr.Usd`.
2. Traverses the stage for meshes (`UsdGeom.Mesh`) and point clouds
   (`UsdGeom.Points`).
3. Flattens geometry into a combined VBO on the GL side.
4. Evaluates skeletal / time-sampled animation (if present) on the CPU at
   runtime and streams updated vertex data each frame.

Rendering pipeline (all GL 3.3 core, no PyOpenGL):
- Per-scene geometry is batched into a single VAO.
- A custom vertex + fragment shader handles MVP transform, per-vertex normals,
  audio-reactive rim lighting, palette cycling, beat-flash bloom pass, and
  film-grain.
- When no USD scenes are found, a built-in procedural particle galaxy is used
  as a placeholder so the effect is never blank.

Audio reactivity:
- bass  → model scale pulse and emission intensity
- mid   → camera orbit speed and palette shift rate
- treble → rim-light brightness and particle jitter
- beat  → hard camera cut between scenes and bloom spike

Scene cycling:
- Each scene stays up for a random dwell in [`dwell_min`, `dwell_max`] seconds.
- A burst of beats may cut early, but never before `dwell_min`.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import threading
from pathlib import Path
from random import shuffle
from typing import NamedTuple

import moderngl
import numpy as np

try:
    import pxr  # type: ignore  # noqa: F401 - presence gate; USD work in usd_scene
    _USD = True
except Exception:  # pragma: no cover
    _USD = False

from unicornviz.effects.base import BaseEffect, AudioData
from unicornviz.paths import resolve_path

# Reusable, GL-free USD loader (sibling module) — handles static meshes plus
# skinned/animated character rigs and blendshapes across NVIDIA packs.
_US_PATH = Path(__file__).resolve().parent / 'usd_scene.py'
_us_spec = importlib.util.spec_from_file_location('sims_usd_scene', _US_PATH)
if _us_spec is None or _us_spec.loader is None:  # pragma: no cover - import guard
    raise ImportError(f'sims-01: failed to load {_US_PATH}')
_usd_scene = importlib.util.module_from_spec(_us_spec)
sys.modules[_us_spec.name] = _usd_scene
_us_spec.loader.exec_module(_usd_scene)
usd_load_scene = _usd_scene.load_scene
UsdLoadOptions = _usd_scene.LoadOptions

log = logging.getLogger(__name__)

_SCENE_CACHE: dict[str, list['_SceneMesh']] = {}
# Cached scene-file selection (random motion-variant pick) shared between the
# background prewarm and the on-activate load so they bake/use the same clips.
_SELECTION_CACHE: dict[str, list[Path]] = {}
_SCENE_LOAD_LOCK = threading.Lock()

# Directory names that hold referenced sub-assets / animation variants rather
# than standalone scenes (skipped during recursive scene discovery so a
# character loads once as its main rig). Lower-cased for case-insensitive match.
_SUBASSET_DIRS = frozenset({'materials', 'textures', 'motions', 'props', '.thumbs'})

# Motion-variant filename fragments to skip when choosing a character's clip —
# we want lively performances, not the near-static idle/rest poses.
_MOTION_EXCLUDE = ('idle', 'pose')


# ---------------------------------------------------------------------------
# Style presets — same rhythm as other asset showcase effects
# ---------------------------------------------------------------------------

class _StylePreset(NamedTuple):
    name: str
    cam_radius: float        # base orbit radius
    cam_elevation: float     # base elevation angle (radians)
    orbit_speed: float       # base radians/s
    light_hue: float         # dominant hue for rim light palette
    palette_phase: float     # offset into the shared palette
    bloom_strength: float    # 0..1
    grain: float             # 0..1


_STYLE_PRESETS: list[_StylePreset] = [
    _StylePreset('neon',      3.2, 0.45, 0.22, 0.62, 0.00, 0.35, 0.02),
    _StylePreset('void',      4.0, 0.20, 0.14, 0.80, 0.12, 0.50, 0.04),
    _StylePreset('solar',     2.8, 0.60, 0.31, 0.12, 0.25, 0.40, 0.01),
    _StylePreset('acid',      3.5, 0.35, 0.45, 0.38, 0.50, 0.28, 0.06),
    _StylePreset('frost',     3.0, 0.55, 0.18, 0.55, 0.75, 0.22, 0.03),
    _StylePreset('inferno',   3.8, 0.30, 0.38, 0.05, 0.88, 0.60, 0.05),
    _StylePreset('spectral',  2.6, 0.70, 0.26, 0.72, 0.15, 0.45, 0.02),
    _StylePreset('deep',      5.0, 0.12, 0.10, 0.90, 0.40, 0.55, 0.03),
    _StylePreset('pulse',     3.2, 0.42, 0.60, 0.30, 0.62, 0.65, 0.04),
    _StylePreset('dream',     4.2, 0.50, 0.20, 0.68, 0.30, 0.35, 0.05),
]


# ---------------------------------------------------------------------------
# Scene mesh data container
# ---------------------------------------------------------------------------

class _SceneMesh:
    """Geometry data extracted from a single USD stage."""

    __slots__ = (
        'path', 'name', 'vertices', 'normals', 'indices',
        'point_positions', 'vbo', 'nbo', 'ibo', 'pbo', 'vao', 'pvao',
        'vert_count', 'point_count', 'bbox_min', 'bbox_max', 'loaded',
        'time_samples', 'current_time',
        # Skeletal-animation playback (see usd_scene.LoadedScene):
        'frames', 'frame_normals', 'fps', 'blend_deltas', 'blend_names',
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = path.stem
        self.vertices: np.ndarray | None = None
        self.normals: np.ndarray | None = None
        self.indices: np.ndarray | None = None
        self.point_positions: np.ndarray | None = None
        self.vbo: moderngl.Buffer | None = None
        self.nbo: moderngl.Buffer | None = None
        self.ibo: moderngl.Buffer | None = None
        self.pbo: moderngl.Buffer | None = None
        self.vao: moderngl.VertexArray | None = None
        self.pvao: moderngl.VertexArray | None = None
        self.vert_count: int = 0
        self.point_count: int = 0
        self.bbox_min = np.zeros(3, dtype=np.float32)
        self.bbox_max = np.zeros(3, dtype=np.float32)
        self.loaded: bool = False
        self.time_samples: list[float] = []
        self.current_time: float = 0.0
        # Baked skeletal-animation playback (None for static scenes).
        self.frames: np.ndarray | None = None         # (F, V, 3)
        self.frame_normals: np.ndarray | None = None  # (F, V, 3)
        self.fps: float = 30.0
        self.blend_deltas: np.ndarray | None = None   # (B, V, 3)
        self.blend_names: list[str] = []

    def release(self) -> None:
        for attr in ('vbo', 'nbo', 'ibo', 'pbo', 'vao', 'pvao'):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
            setattr(self, attr, None)
        self.loaded = False


# ---------------------------------------------------------------------------
# Vertex / fragment shaders
# ---------------------------------------------------------------------------

_VERT_MESH = """
#version 330
// Vertex shader for USD mesh geometry.
// Uniforms: iMVP (model-view-projection), iModel (model matrix for normal
// transform), iNormalMat (inverse-transpose of model-view for lighting).
// Outputs v_normal and v_worldPos to the fragment stage.
in vec3 in_vert;
in vec3 in_normal;

uniform mat4 iMVP;
uniform mat4 iModel;
uniform mat3 iNormalMat;

out vec3 v_normal;
out vec3 v_worldPos;

void main() {
    vec4 wp = iModel * vec4(in_vert, 1.0);
    v_worldPos = wp.xyz;
    v_normal   = normalize(iNormalMat * in_normal);
    gl_Position = iMVP * vec4(in_vert, 1.0);
}
"""

_VERT_POINTS = """
#version 330
// Vertex shader for USD point-cloud / particle geometry.
// Audio-reactive jitter and scale applied in VS for zero Python overhead.
in vec3 in_pos;

uniform mat4 iMVP;
uniform float iTime;
uniform float iBass;
uniform float iTreble;

out vec3 v_color;

vec3 palette(float t) {
    vec3 a = vec3(0.5);
    vec3 b = vec3(0.5);
    vec3 c = vec3(1.0, 0.9, 0.8);
    vec3 d = vec3(0.18, 0.28, 0.55);
    return clamp(a + b * cos(6.28318 * (c * t + d)), 0.0, 1.0);
}

void main() {
    float jitter = iTreble * 0.08;
    float hash   = fract(sin(dot(in_pos.xy, vec2(127.1, 311.7))) * 43758.5453);
    vec3  pos    = in_pos + vec3(
        sin(iTime * 0.7 + hash * 6.28) * jitter,
        cos(iTime * 0.5 + hash * 6.28) * jitter,
        sin(iTime * 0.9 + hash * 3.14) * jitter * 0.5
    );
    float scale = 1.0 + iBass * 0.25;
    pos *= scale;
    gl_Position  = iMVP * vec4(pos, 1.0);
    gl_PointSize = clamp(3.0 + iBass * 8.0 + iTreble * 4.0, 1.0, 18.0);
    v_color      = palette(hash + iTime * 0.04);
}
"""

_FRAG_MESH = """
#version 330
// Fragment shader for USD mesh — Phong + audio-reactive rim lighting,
// palette cycling, beat bloom, and film grain.
in vec3 v_normal;
in vec3 v_worldPos;

uniform float iTime;
uniform float iBass;
uniform float iMid;
uniform float iTreble;
uniform float iBeat;
uniform float iStylePhase;   // 0..1 per style preset
uniform float iPalShift;     // running palette phase
uniform float iBloom;        // 0..1 bloom from beat
uniform float iGrain;        // grain amount
uniform vec2  iResolution;
uniform vec3  iCamPos;
uniform float iFade;         // scene-transition dip-to-black [0,1]

out vec4 fragColor;

vec3 palette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 0.9, 0.7);
    vec3 d = vec3(0.10 + iStylePhase * 0.30,
                  0.20 + iStylePhase * 0.25,
                  0.40 + iStylePhase * 0.40);
    return clamp(a + b * cos(6.28318 * (c * t + d)), 0.0, 1.0);
}

float sdGrid(vec3 p, float cellSize) {
    vec3 q = mod(p, cellSize) - cellSize * 0.5;
    return min(min(abs(q.x), abs(q.y)), abs(q.z));
}

void main() {
    vec3  N       = normalize(v_normal);
    vec3  L       = normalize(vec3(0.8, 1.2, 0.6));    // key light
    vec3  V       = normalize(iCamPos - v_worldPos);
    vec3  H       = normalize(L + V);

    // Diffuse + specular
    float diff    = max(dot(N, L), 0.0) * 0.7 + 0.3;
    float spec    = pow(max(dot(N, H), 0.0), 48.0) * 0.6;

    // Audio-reactive rim light
    float rim     = 1.0 - clamp(dot(N, V), 0.0, 1.0);
    rim           = pow(rim, 2.5) * (0.4 + iBass * 1.2 + iTreble * 0.6);
    vec3  rimCol  = palette(iPalShift + iStylePhase * 0.3 + iTime * 0.05);

    // Base surface colour from palette + mid-frequency shift
    vec3  baseCol = palette(iPalShift + dot(v_worldPos, vec3(0.1)) * 0.08
                            + iMid * 0.25);

    vec3  col     = baseCol * diff + vec3(spec) + rimCol * rim;

    // Beat bloom spike — additive white flash decaying quickly
    col += vec3(iBloom * 0.6);

    // Subtle grid wireframe overlay
    float grid    = smoothstep(0.02, 0.00, sdGrid(v_worldPos, 0.5));
    col           = mix(col, rimCol * 1.4, grid * 0.18);

    // Film grain
    if (iGrain > 0.0) {
        float n = fract(sin(dot(gl_FragCoord.xy / iResolution * 200.0,
                               vec2(12.9898, 78.233))) * 43758.5453 + iTime);
        col += (n - 0.5) * iGrain;
    }

    fragColor = vec4(clamp(col, 0.0, 1.0) * iFade, 1.0);
}
"""

_FRAG_POINTS = """
#version 330
// Fragment shader for point-cloud particles — circular soft sprites.
in vec3 v_color;
uniform float iFade;         // scene-transition dip-to-black [0,1]
out vec4 fragColor;

void main() {
    vec2  d = gl_PointCoord - 0.5;
    float r = dot(d, d) * 4.0;
    if (r > 1.0) discard;
    float alpha = 1.0 - r;
    fragColor = vec4(v_color * iFade, alpha * iFade);
}
"""


# ---------------------------------------------------------------------------
# Procedural fallback galaxy (used when no USD scenes exist)
# ---------------------------------------------------------------------------

def _make_galaxy_points(n: int = 8000) -> np.ndarray:
    """Return a (n, 3) float32 array of spiral-galaxy point positions."""
    rng   = np.random.default_rng(42)
    arms  = 3
    idx   = np.arange(n, dtype=np.float32)
    arm   = idx % arms
    t     = (idx / n) * 4.0 * np.pi
    r     = 0.2 + t * 0.4 + rng.standard_normal(n).astype(np.float32) * 0.15
    theta = t + arm * (2.0 * np.pi / arms)
    theta += rng.standard_normal(n).astype(np.float32) * 0.12
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rng.standard_normal(n).astype(np.float32) * 0.12 * (1.0 - r / (r.max() + 1e-6))
    return np.stack([x, y, z], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Math helpers — pure numpy, no scipy dependency
# ---------------------------------------------------------------------------

def _perspective(fovy: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(fovy * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _lookat(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = center - eye
    f /= np.linalg.norm(f) + 1e-9
    r = np.cross(f, up)
    r /= np.linalg.norm(r) + 1e-9
    u = np.cross(r, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3]  = -np.dot(r, eye)
    m[1, 3]  = -np.dot(u, eye)
    m[2, 3]  =  np.dot(f, eye)
    return m


def _mat4_mul(*mats: np.ndarray) -> np.ndarray:
    result = mats[0]
    for m in mats[1:]:
        result = result @ m
    return result


# ---------------------------------------------------------------------------
# Effect class
# ---------------------------------------------------------------------------

class SimShowcase(BaseEffect):
    """Audio-reactive OpenUSD scene showcase.

    Loads .usda / .usdc / .usdz files from scenes/ and renders them with
    beat-synced camera cuts, palette cycling, rim lighting, and particles.
    Falls back to a procedural spiral galaxy when no scene files are found.
    """

    NAME   = 'Sim Showcase'
    AUTHOR = 'unicorn-viz drop-in'
    TAGS = ['media', '3d', 'usd', 'simulation', 'groovy', 'intense']
    HELP_ENTRIES = [
        ('Sim Showcase', 'Ctrl+N', 'Next USD scene'),
        ('Sim Showcase', 'Ctrl+P', 'Prev USD scene'),
        ('Sim Showcase', 'Ctrl+R', 'Random USD scene'),
    ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'mix_time': float(self.config.get('mix_time', 14.0)),
            'camera_energy': float(self.config.get('camera_energy', 1.0)),
            'zoom': float(self.config.get('zoom', 1.0)),
        }

        # Per-scene dwell window (seconds). Each scene stays up for a random
        # time in [dwell_min, dwell_max]; a beat burst may cut early but never
        # before dwell_min, so scenes never flicker past too fast.
        self._dwell_min = max(0.5, float(self.config.get('dwell_min', 7.0)))
        self._dwell_max = max(self._dwell_min,
                              float(self.config.get('dwell_max', 20.0)))
        # Camera distance multiplier (<1 pulls the camera in / makes the subject
        # larger). 0.8 = a noticeable ~20% closer baseline without crowding.
        self._cam_distance = max(0.3, float(self.config.get('camera_distance', 0.8)))
        # Scene-to-scene transition: dip-to-black fade. transition="cut" or
        # transition_s<=0 disables it (hard cuts).
        self._transition_s = max(0.0, float(self.config.get('transition_s', 0.6)))
        self._transitions = (self._transition_s > 0.0
                             and str(self.config.get('transition', 'fade')).lower() != 'cut')
        self._fade = 1.0           # 1 = fully visible, 0 = black (mid-transition)
        self._pending_swap = None  # deferred scene swap, fired at the black point
        # Dance: audio-reactive secondary motion (beat bounce + hop + sway/twist)
        # layered on top of each clip. 0 = off, 1 = default, >1 = wilder.
        self._dance = max(0.0, float(self.config.get('dance', 1.0)))

        self._mesh_prog  = self._make_program(_VERT_MESH,   _FRAG_MESH)
        self._point_prog = self._make_program(_VERT_POINTS, _FRAG_POINTS)

        self._scenes: list[_SceneMesh] = []
        self._current  = 0
        self._timer    = 0.0
        self._dwell    = self.rng.uniform(self._dwell_min, self._dwell_max)
        self._beat_acc = 0.0
        self._bass     = 0.0
        self._mid      = 0.0
        self._treble   = 0.0
        self._beat     = 0.0
        self._bloom    = 0.0
        self._orbit    = 0.0
        self._cam_time = 0.0
        self._pal_shift = 0.0
        self._style_index = 0
        self._style_order: list[int] = list(range(len(_STYLE_PRESETS)))
        shuffle(self._style_order)
        self._beat_kick = 0.0

        # Skeletal-animation playback options (Tier 1/2) + audio-reactive
        # blendshapes (Tier 3). All optional; static packs ignore them.
        self._skel_animate = bool(self.config.get('skel_animate', True))
        self._skel_frames = int(self.config.get('skel_frames', 48))
        self._skel_speed = float(self.config.get('skel_speed', 1.0))
        self._face_audio_react = bool(self.config.get('face_audio_react', False))
        # How many dynamic motion clips to load per character (each is a scene).
        self._motions_per_char = max(1, int(self.config.get('motions_per_character', 1)))
        pose_t = self.config.get('skel_pose_time', None)
        self._anim_clock = 0.0          # seconds, advances animation playback
        self._dyn_frame = -1            # last frame index uploaded (dirty check)

        # Pre-allocate camera state
        self._cam_eye    = np.array([0.0, 0.0, 3.0], dtype=np.float32)
        self._proj       = np.eye(4, dtype=np.float32)
        self._view       = np.eye(4, dtype=np.float32)

        # Load USD scenes
        load_opts = UsdLoadOptions(
            animate=self._skel_animate,
            frames=max(2, self._skel_frames),
            pose_time=float(pose_t) if pose_t is not None else None,
            blendshapes=self._face_audio_react,
        )
        # Copy the cached list so we can shuffle per-instance without mutating
        # the shared cache; a random order (and thus a random first scene) makes
        # each run distinct per the effect-randomization guideline.
        self._scenes = list(self.__class__._load_scene_pack_cached(
            self.ctx,
            self.__class__._scene_dirs(self.config),
            load_opts,
            self._motions_per_char,
        ))

        # Fallback galaxy
        if not self._scenes:
            self._scenes = [self._make_galaxy_scene()]

        shuffle(self._scenes)
        log.info('SimShowcase: %d scene(s) ready', len(self._scenes))
        self._announce_scene('initial')

    @property
    def current_position_label(self) -> str:
        return 'SCENE IDX'

    @property
    def current_position_text(self) -> str:
        total = len(self._scenes)
        if total <= 0:
            return '-/-'
        return f'{self._current + 1}/{total}'

    # ------------------------------------------------------------------
    # Class-level scene dir / cache
    # ------------------------------------------------------------------

    @classmethod
    def _scene_dirs(cls, cfg: dict) -> list[Path]:
        """Resolve one or more scene directories from config.

        Supported keys:
        - scene_dir: single directory override
        - scene_dirs: comma-separated list of additional directories
        """
        dirs: list[Path] = []

        if cfg:
            single = str(cfg.get('scene_dir', '')).strip()
            if single:
                dirs.append(Path(single).expanduser())

            multi_raw = str(cfg.get('scene_dirs', '')).strip()
            if multi_raw:
                for token in multi_raw.split(','):
                    token = token.strip()
                    if token:
                        dirs.append(Path(token).expanduser())

        # Built-in + common external drop locations for public USD datasets.
        here = Path(__file__).parent
        dirs.append(here / 'scenes')
        dirs.append(resolve_path('assets/sims'))
        dirs.append(resolve_path('assets/robotics-usd'))

        out: list[Path] = []
        seen: set[str] = set()
        for d in dirs:
            key = str(d.resolve()) if d.exists() else str(d)
            if key in seen:
                continue
            seen.add(key)
            if d.is_dir():
                out.append(d)
        return out

    @classmethod
    def _discover_scene_files(
        cls, scene_dirs: list[Path], motions_per_character: int,
    ) -> list[Path]:
        """Resolve the scene file list (random motion-variant pick included).

        Cached per (dirs, motions) so the background prewarm and the on-activate
        load agree on the *same* shuffled selection — otherwise the prewarm would
        bake different motion clips than the effect ends up using.
        """
        sel_key = '|'.join(str(d.resolve() if d.exists() else d)
                           for d in scene_dirs) + f'#m{motions_per_character}'
        with _SCENE_LOAD_LOCK:
            sel = _SELECTION_CACHE.get(sel_key)
        if sel is not None:
            return sel

        paths_set: set[Path] = set()
        for scene_dir in scene_dirs:
            paths_set.update(
                p for p in scene_dir.rglob('*')
                # ``.usd`` (auto crate/ascii) is just as valid as the explicit
                # forms — NVIDIA character packs use it for the main rig file.
                if p.suffix.lower() in ('.usd', '.usda', '.usdc', '.usdz')
                # Skip referenced sub-assets / animation variants so each
                # character loads once as its main rig rather than as dozens of
                # material/texture/motion/prop sub-layers.
                and not (_SUBASSET_DIRS & {part.lower() for part in p.parent.parts})
            )
        paths = sorted(paths_set)

        # Expand character rigs into their dynamic motion variants: Reallusion
        # packs ship Props/<stem>.<Motion>.usd files that bind lively clips
        # (Taunt_Roar, Music_Swinging, …). The base file usually binds only a
        # near-static idle, so prefer a few random non-idle variants per
        # character for actual performances. Files with no such variants (the
        # demo scenes) pass through unchanged.
        expanded: list[Path] = []
        for p in paths:
            variants = sorted((p.parent / 'Props').glob(f'{p.stem}.*.usd'))
            dynamic = [v for v in variants
                       if not any(k in v.stem.lower() for k in _MOTION_EXCLUDE)]
            if dynamic:
                shuffle(dynamic)
                expanded.extend(dynamic[:max(1, motions_per_character)])
            else:
                expanded.append(p)
        with _SCENE_LOAD_LOCK:
            _SELECTION_CACHE[sel_key] = expanded
        return expanded

    @classmethod
    def prewarm_async(cls, cfg: dict) -> None:
        """Pre-bake scenes (GL-free) on a daemon thread during app splash.

        Populates usd_scene's CPU cache so the first activation only needs fast
        GL uploads instead of stalling on USD parsing + skinning/rigid baking.
        No-op without OpenUSD. Safe to call once at startup.
        """
        if not _usd_scene._USD:  # noqa: SLF001 - module-level availability flag
            return
        ecfg = cfg or {}
        scene_dirs = cls._scene_dirs(ecfg)
        opts = UsdLoadOptions(
            animate=bool(ecfg.get('skel_animate', True)),
            frames=max(2, int(ecfg.get('skel_frames', 48))),
            pose_time=(float(ecfg['skel_pose_time'])
                       if ecfg.get('skel_pose_time') is not None else None),
            blendshapes=bool(ecfg.get('face_audio_react', False)),
        )
        motions = max(1, int(ecfg.get('motions_per_character', 1)))

        def _worker() -> None:
            try:
                files = cls._discover_scene_files(scene_dirs, motions)
                for f in files:
                    usd_load_scene(f, opts)   # result stored in usd_scene CPU cache
                log.info('SimShowcase: prewarmed %d scene(s) in background',
                         len(files))
            except Exception as exc:
                log.warning('SimShowcase prewarm failed: %s', exc)

        threading.Thread(target=_worker, name='sims-prewarm', daemon=True).start()

    @classmethod
    def _load_scene_pack_cached(
        cls,
        ctx: moderngl.Context,
        scene_dirs: list[Path],
        opts: 'UsdLoadOptions',
        motions_per_character: int = 1,
    ) -> list[_SceneMesh]:
        cache_key = '|'.join(
            str(d.resolve() if d.exists() else d)
            for d in scene_dirs
        ) + (f'#anim={opts.animate},frames={opts.frames},'
             f'blend={opts.blendshapes},motions={motions_per_character}')
        with _SCENE_LOAD_LOCK:
            cached = _SCENE_CACHE.get(cache_key)
            if cached is not None:
                for scene in cached:
                    cls._ensure_scene_gpu_resources(ctx, scene)
                return cached

        if not scene_dirs:
            log.info('SimShowcase: no valid scene directories resolved')
            _SCENE_CACHE[cache_key] = []
            return []

        paths = cls._discover_scene_files(scene_dirs, motions_per_character)

        if not paths:
            log.info('SimShowcase: no .usd* files in %s', ', '.join(str(d) for d in scene_dirs))
            _SCENE_CACHE[cache_key] = []
            return []

        scenes: list[_SceneMesh] = []
        for path in paths:
            try:
                scene = _SceneMesh(path)
                cls._load_scene_gl(ctx, scene, opts)
                if scene.vert_count == 0 and scene.point_count == 0:
                    # No geometry extracted — almost always because pxr/OpenUSD
                    # is not installed (then every stage yields empty arrays), or
                    # an empty stage. Skip it so the caller falls back to the
                    # built-in galaxy scene instead of rendering a black frame.
                    scene.release()
                    log.warning('SimShowcase: %s produced no geometry; skipping',
                                path.name)
                    continue
                scenes.append(scene)
                log.info('SimShowcase: loaded %s (verts=%d pts=%d)',
                         path.name, scene.vert_count, scene.point_count)
            except Exception as exc:
                log.warning('SimShowcase: failed to load %s: %s', path, exc)

        if not scenes and not _USD:
            log.warning(
                'SimShowcase: pxr/OpenUSD is not installed, so no .usd* scene '
                'can be loaded — falling back to the built-in galaxy scene. '
                'Install OpenUSD (pip install usd-core) to enable the USD scenes.'
            )

        with _SCENE_LOAD_LOCK:
            _SCENE_CACHE[cache_key] = scenes
        return scenes

    @classmethod
    def _load_scene_gl(cls, ctx: moderngl.Context, scene: _SceneMesh,
                       opts: 'UsdLoadOptions') -> None:
        loaded = usd_load_scene(scene.path, opts)
        if loaded is None:
            scene.vertices = np.zeros((0, 3), dtype=np.float32)
            scene.normals = np.zeros((0, 3), dtype=np.float32)
            scene.indices = np.zeros((0, 3), dtype=np.uint32)
            scene.point_positions = np.zeros((0, 3), dtype=np.float32)
            cls._upload_scene_buffers(ctx, scene)
            return
        # Prefer the filename stem (e.g. "Orc.Taunt_Roar") — more descriptive
        # than the generic default-prim name shared by a character's variants.
        scene.name = scene.path.stem or loaded.name
        scene.vertices = loaded.vertices
        scene.normals = loaded.normals
        scene.indices = loaded.indices
        scene.point_positions = loaded.points
        # Animation + blendshape payload (None for static scenes).
        scene.frames = loaded.frames
        scene.frame_normals = loaded.frame_normals
        scene.fps = float(loaded.fps)
        scene.blend_deltas = loaded.blend_deltas
        scene.blend_names = list(loaded.blend_names)
        cls._upload_scene_buffers(ctx, scene)

    @classmethod
    def _upload_scene_buffers(cls, ctx: moderngl.Context, scene: _SceneMesh) -> None:
        """Upload mesh/point arrays into GL buffers for the given scene."""
        # Ensure stale GPU handles are gone before creating fresh buffers.
        for attr in ('vbo', 'nbo', 'ibo', 'pbo'):
            obj = getattr(scene, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
                setattr(scene, attr, None)

        # Upload mesh geometry
        verts = scene.vertices if scene.vertices is not None else np.zeros((0, 3), dtype=np.float32)
        norms = scene.normals if scene.normals is not None else np.zeros((0, 3), dtype=np.float32)
        idxs = scene.indices if scene.indices is not None else np.zeros((0, 3), dtype=np.uint32)
        pts = scene.point_positions if scene.point_positions is not None else np.zeros((0, 3), dtype=np.float32)

        if len(verts) > 0 and len(idxs) > 0:
            scene.vbo = ctx.buffer(verts.astype(np.float32).tobytes())
            scene.nbo = ctx.buffer(norms.astype(np.float32).tobytes())
            scene.ibo = ctx.buffer(idxs.astype(np.uint32).tobytes())
            scene.vert_count = int(len(idxs) * 3)
            # Build a dummy mesh prog VAO; actual prog bound in render
            # We use a dummy context just for the buffer layout here.
        else:
            scene.vert_count = 0

        # Upload point geometry
        if len(pts) > 0:
            scene.pbo         = ctx.buffer(pts.tobytes())
            scene.point_count = int(len(pts))
        else:
            scene.point_count = 0

        scene.loaded = True

    @classmethod
    def _ensure_scene_gpu_resources(cls, ctx: moderngl.Context, scene: _SceneMesh) -> None:
        """Recreate GPU buffers for cached scenes that were previously destroyed."""
        need_mesh = scene.vert_count > 0 and (scene.vbo is None or scene.nbo is None or scene.ibo is None)
        need_points = scene.point_count > 0 and scene.pbo is None
        if not scene.loaded or need_mesh or need_points:
            cls._upload_scene_buffers(ctx, scene)

    # ------------------------------------------------------------------
    # Fallback galaxy scene
    # ------------------------------------------------------------------

    def _make_galaxy_scene(self) -> _SceneMesh:
        from pathlib import Path as _P
        fake_path = _P(__file__).parent / 'scenes' / '_galaxy_fallback.usda'
        scene = _SceneMesh(fake_path)
        scene.name = 'Galaxy'
        pts = _make_galaxy_points(10000)
        scene.pbo   = self.ctx.buffer(pts.tobytes())
        scene.point_count = len(pts)
        scene.loaded = True
        return scene

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, dt: float, audio: AudioData) -> None:
        super().update(dt, audio)
        self._bass   = float(audio.bass_n)
        self._mid    = float(audio.mid_n)
        self._treble = float(audio.treble_n)

        # Beat accumulator — a burst of beats triggers scene cut
        beat = float(audio.beat)
        self._beat_acc = max(0.0, self._beat_acc - dt * 2.5)
        if beat:
            self._beat_acc += 0.35
        self._beat = beat

        # Bloom decay
        if beat:
            self._bloom = min(1.0, self._bloom + 0.8)
            self._beat_kick = min(1.0, self._beat_kick + 0.9)
        self._bloom = max(0.0, self._bloom - dt * 4.0)
        self._beat_kick = max(0.0, self._beat_kick - dt * 3.0)

        self._timer += dt * max(0.1, self.parameters['speed'])

        # Advance at the rolled dwell, or let a beat burst cut early — but never
        # before dwell_min, so each scene stays up a sensible 7-20s. When idle
        # (not mid-transition), a due advance kicks off a dip-to-black fade.
        due = self._timer >= self._dwell or (
            self._timer >= self._dwell_min and self._beat_acc >= 1.0
        )
        if (due and len(self._scenes) > 1
                and self._pending_swap is None and self._fade >= 1.0):
            self._begin_transition(self._advance_scene)
        self._tick_transition(dt)

        # Camera orbit speed modulated by mid
        self._orbit += dt * (0.18 + self._mid * 0.35) * max(0.1, self.parameters['speed'])
        self._cam_time += dt * (0.55 + self._mid * 0.95) * max(0.1, self.parameters['speed'])
        self._pal_shift += dt * (0.04 + self._mid * 0.08)
        # Advance skeletal-animation playback (slight beat-driven push).
        # Baked clip playback surges with the beat/bass so the motion itself
        # (steps, sweeps) pumps with the music, on top of the dance transform.
        surge = 1.0 + (self._beat_kick * 1.2 + self._bass * 0.4) * (1.0 if self._dance else 0.0)
        self._anim_clock += dt * max(0.0, self._skel_speed) * surge

    def _reset_scene_clocks(self) -> None:
        """Reset per-scene timers + animation playback (call on scene change)."""
        self._timer = 0.0
        self._beat_acc = 0.0
        self._anim_clock = 0.0
        self._dyn_frame = -1
        # Re-roll this scene's dwell so each gets its own 7-20s (configurable).
        self._dwell = self.rng.uniform(self._dwell_min, self._dwell_max)

    def _announce_scene(self, reason: str) -> None:
        """Info-log the scene that just became active."""
        if not self._scenes:
            return
        sc = self._scenes[self._current]
        kind = (f'animated:{len(sc.frames)}f@{sc.fps:.0f}fps'
                if sc.frames is not None else
                ('static-mesh' if sc.vert_count else 'points'))
        log.info('SimShowcase: now showing [%d/%d] %s (%s, %s) via %s',
                 self._current + 1, len(self._scenes), sc.name,
                 kind,
                 f'verts={sc.vert_count}' if sc.vert_count else f'pts={sc.point_count}',
                 reason)

    def _begin_transition(self, swap_fn) -> None:
        """Start a dip-to-black transition; ``swap_fn`` runs at the black point.

        Falls back to an instant swap when transitions are disabled or one is
        already in progress.
        """
        if not self._transitions or self._pending_swap is not None or self._fade < 1.0:
            swap_fn()
            return
        self._pending_swap = swap_fn

    def _tick_transition(self, dt: float) -> None:
        """Advance the fade envelope: out to black, swap, then back in."""
        if self._transition_s <= 0.0:
            return
        rate = 1.0 / (self._transition_s * 0.5)
        if self._pending_swap is not None:
            self._fade = max(0.0, self._fade - dt * rate)
            if self._fade <= 0.0:
                self._pending_swap()        # perform the swap at full black
                self._pending_swap = None
        elif self._fade < 1.0:
            self._fade = min(1.0, self._fade + dt * rate)

    def _advance_scene(self) -> None:
        if len(self._scenes) > 1:
            self._current = (self._current + 1) % len(self._scenes)
            self._style_index = (self._style_index + 1) % len(self._style_order)
            if self._style_index == 0:
                shuffle(self._style_order)
            self._reset_scene_clocks()
            self._announce_scene('auto-advance')
        else:
            self._reset_scene_clocks()

    def next_scene(self) -> str | None:
        """Advance to next scene and return its display name."""
        self._advance_scene()
        if not self._scenes:
            return None
        return self._scenes[self._current].name

    def prev_scene(self) -> str | None:
        """Step to previous scene and return its display name."""
        if not self._scenes:
            return None
        self._current = (self._current - 1) % len(self._scenes)
        self._reset_scene_clocks()
        self._announce_scene('prev')
        return self._scenes[self._current].name

    def random_scene(self) -> str | None:
        """Jump to a random scene and return its display name."""
        if not self._scenes:
            return None
        if len(self._scenes) > 1:
            cur = self._current
            while self._current == cur:
                self._current = int(self.rng.integers(0, len(self._scenes)))
        self._reset_scene_clocks()
        self._announce_scene('random')
        return self._scenes[self._current].name

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _stream_animation_frame(self, scene: _SceneMesh) -> None:
        """Stream the current baked animation frame (+ optional audio-reactive
        blendshapes) into the scene's vertex/normal buffers.

        No-op for static scenes (``scene.frames is None``).  Frames are baked,
        normalised CPU arrays; we write the active frame into the existing GPU
        buffers each render (single active effect instance at a time).
        """
        frames = scene.frames
        if frames is None or scene.vbo is None or scene.nbo is None:
            return
        n = len(frames)
        idx = int(self._anim_clock * scene.fps) % n if n > 1 else 0

        verts = frames[idx]
        norms = scene.frame_normals[idx] if scene.frame_normals is not None \
            else scene.normals

        # Tier 3: audio-reactive blendshape weighting (e.g. jaw on bass).
        # Only engages when the pack ships name-matched blendshapes; otherwise
        # blend_deltas is None and playback is animation-only.
        deltas = scene.blend_deltas
        if self._face_audio_react and deltas is not None and len(deltas):
            bands = (self._bass, self._treble, self._mid, self._beat)
            verts = verts.copy()
            for i in range(len(deltas)):
                w = float(np.clip(bands[i % len(bands)] * 1.4, 0.0, 1.2))
                if w > 1e-3:
                    verts += deltas[i] * w

        # Skip redundant uploads when the frame is unchanged and no face react.
        if idx == self._dyn_frame and not self._face_audio_react:
            return
        self._dyn_frame = idx
        scene.vbo.write(np.ascontiguousarray(verts, dtype=np.float32).tobytes())
        scene.nbo.write(np.ascontiguousarray(norms, dtype=np.float32).tobytes())

    def render(self) -> None:
        if not self._scenes:
            return

        scene  = self._scenes[self._current]
        style  = _STYLE_PRESETS[self._style_order[self._style_index]]
        aspect = self.width / max(1, self.height)
        cam_energy = max(0.2, float(self.parameters.get('camera_energy', 1.0)))

        # Cinematic fly camera: layered dolly/orbit/strafe with beat accents.
        drift = self._cam_time
        radius = style.cam_radius * (
            1.0
            + 0.20 * np.sin(drift * 0.23)
            + 0.10 * np.cos(drift * 0.51)
            + self._bass * 0.22
            + self._beat_kick * 0.12
        )
        fly_in = max(0.0, np.sin(drift * 0.17 + 0.8)) ** 6
        fly_scale = 1.0 - (0.58 * fly_in + 0.22 * self._beat_kick)
        radius *= max(0.22, fly_scale)
        radius *= self._cam_distance      # pull the baseline camera in a touch
        radius = max(0.5, radius)
        elev = style.cam_elevation + (
            0.18 * np.sin(drift * 0.37 + style.palette_phase * 6.28318)
            + 0.08 * np.cos(drift * 0.79)
            + self._treble * 0.05
        )
        azim = (
            drift * (0.72 + style.orbit_speed * 0.85)
            + 0.45 * np.sin(drift * 0.19)
            + 0.20 * np.sin(self._orbit * 0.7)
        )

        strafe = np.array([
            np.sin(drift * 0.29),
            np.sin(drift * 0.41 + 1.7),
            np.cos(drift * 0.33 + 0.8),
        ], dtype=np.float32) * (0.32 * cam_energy)

        target = np.array([
            0.22 * np.sin(drift * 0.27 + 0.8),
            0.16 * np.sin(drift * 0.43),
            0.22 * np.cos(drift * 0.31 + 0.4),
        ], dtype=np.float32) * cam_energy
        target += np.array([0.0, self._beat_kick * 0.06, 0.0], dtype=np.float32)

        eye = np.array([
            radius * np.cos(elev) * np.cos(azim),
            radius * np.sin(elev),
            radius * np.cos(elev) * np.sin(azim),
        ], dtype=np.float32)
        eye += strafe
        self._cam_eye[:] = eye

        # Gentle banking via dynamic up vector for a fly-by feel.
        roll = 0.10 * np.sin(drift * 0.52) + self._treble * 0.08
        up = np.array([
            np.sin(roll),
            np.cos(roll),
            0.0,
        ], dtype=np.float32)

        proj = _perspective(np.radians(55.0 / max(float(self.parameters['zoom']), 0.1)), aspect, 0.01, 60.0)
        view = _lookat(eye, target, up)

        # Model: audio-reactive "dance" — beat squash/stretch + hop + sway/twist
        # layered on the baked clip so every sim grooves to the music.
        d = self._dance
        kick = self._beat_kick
        bass = self._bass
        sway  = (0.05 + 0.12 * self._mid) * np.sin(self.time * 2.3) * d
        twist = (0.04 + 0.10 * self._treble) * np.sin(self.time * 1.7) * d
        hop   = (0.16 * kick + 0.05 * bass) * d
        scl_y = 1.0 + (0.18 * kick + 0.06 * bass) * d
        scl_xz = 1.0 - 0.10 * kick * d + 0.02 * bass * d
        cz, sz = np.cos(sway), np.sin(sway)
        ct, st = np.cos(twist), np.sin(twist)
        rot_z = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0],
                          [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        rot_y = np.array([[ct, 0, st, 0], [0, 1, 0, 0],
                          [-st, 0, ct, 0], [0, 0, 0, 1]], dtype=np.float32)
        scale_m = np.diag([scl_xz, scl_y, scl_xz, 1.0]).astype(np.float32)
        trans_m = np.eye(4, dtype=np.float32)
        trans_m[1, 3] = hop
        model = trans_m @ rot_z @ rot_y @ scale_m

        mvp       = _mat4_mul(proj, view, model)
        normal_mat = np.linalg.inv(model[:3, :3]).T.astype(np.float32)

        def su(prog: moderngl.Program, name: str, value) -> None:
            """Set uniform silently if not present."""
            try:
                prog[name].value = value
            except KeyError:
                pass

        style_phase = self._style_order[self._style_index] / max(1, len(_STYLE_PRESETS))

        # --- Skeletal-animation playback: stream the current frame ---
        self._stream_animation_frame(scene)

        # --- Render mesh geometry ---
        if scene.vert_count > 0 and scene.vbo is not None:
            vao = self.ctx.vertex_array(
                self._mesh_prog,
                [
                    (scene.vbo, '3f', 'in_vert'),
                    (scene.nbo, '3f', 'in_normal'),
                ],
                scene.ibo,
            )
            su(self._mesh_prog, 'iMVP',        tuple(mvp.T.flatten()))
            su(self._mesh_prog, 'iModel',      tuple(model.T.flatten()))
            su(self._mesh_prog, 'iNormalMat',  tuple(normal_mat.flatten()))
            su(self._mesh_prog, 'iTime',       self.time)
            su(self._mesh_prog, 'iBass',       self._bass)
            su(self._mesh_prog, 'iMid',        self._mid)
            su(self._mesh_prog, 'iTreble',     self._treble)
            su(self._mesh_prog, 'iBeat',       self._beat)
            su(self._mesh_prog, 'iStylePhase', style_phase)
            su(self._mesh_prog, 'iPalShift',   self._pal_shift % 1.0)
            su(self._mesh_prog, 'iBloom',      self._bloom)
            su(self._mesh_prog, 'iGrain',      style.grain)
            su(self._mesh_prog, 'iResolution', (float(self.width), float(self.height)))
            su(self._mesh_prog, 'iCamPos',     tuple(float(v) for v in eye))
            su(self._mesh_prog, 'iFade',       self._fade)
            vao.render(moderngl.TRIANGLES)
            vao.release()

        # --- Render point cloud / particles ---
        if scene.point_count > 0 and scene.pbo is not None:
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            pvao = self.ctx.vertex_array(
                self._point_prog,
                [(scene.pbo, '3f', 'in_pos')],
            )
            su(self._point_prog, 'iMVP',    tuple(mvp.T.flatten()))
            su(self._point_prog, 'iTime',   self.time)
            su(self._point_prog, 'iBass',   self._bass)
            su(self._point_prog, 'iTreble', self._treble)
            su(self._point_prog, 'iFade',   self._fade)
            pvao.render(moderngl.POINTS)
            pvao.release()

    # ------------------------------------------------------------------
    # Destroy
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        for scene in self._scenes:
            scene.release()
        if hasattr(self, '_mesh_prog') and self._mesh_prog:
            self._mesh_prog.release()
        if hasattr(self, '_point_prog') and self._point_prog:
            self._point_prog.release()
