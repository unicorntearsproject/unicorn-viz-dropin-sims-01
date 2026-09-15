# sims-01 — 3D Scene Showcase

**Version 0.7.2**


A drop-in effect for [unicorn-viz](https://github.com/iDoMeteor/unicorn-viz) that renders a gallery of 3D scenes and models using OpenUSD (USD/USDC/USDZ) and IsaacLab-compatible geometries with full audio-reactivity.

## Description

Sim Showcase loads 3D scenes from a configurable directory and renders them as full-screen, audio-reactive visuals using a custom GL-native rasterizer built on top of moderngl. Each scene is presented with one of ten rotating presentation styles that vary camera movement, lighting, palette, and post-processing.

### Scene Format Support

- **OpenUSD files** — `.usd`, `.usdc`, `.usdz`
- **Skeletal animation** — time-sampled deformations evaluated on the CPU and streamed to GPU
- **Point clouds** — `UsdGeom.Points` primitives rendered as particle systems
- **Batched geometry** — multiple meshes per scene combined into a single VAO for efficiency

### Rendering Pipeline

1. **USD stage traversal** — scans for `UsdGeom.Mesh` and `UsdGeom.Points` primitives
2. **Geometry flattening** — combines all scene meshes into a single VBO
3. **Animation streaming** — evaluates time-sampled transforms and vertex data each frame
4. **Audio-reactive shading** — rim lighting, palette shift, bloom, and grain modulated by audio signals
5. **Presentation styles** — ten preset combinations of camera movement, lighting theme, and post-processing

## Features

- **10 presentation styles** — neon, void, solar, acid, frost, inferno, spectral, deep, pulse, dream
- **Audio reactivity** — bass (scale/emission), mid (orbit/palette), treble (rim light/jitter), beat (camera cut/bloom spike)
- **Beat-aware scene transitions** — cuts between scenes on beat events; configurable dwell time
- **Fallback particle galaxy** — if no USD scenes are found, a procedural particle system keeps the effect visible
- **Lazy loading** — scenes can be preloaded at startup or opened on first display
- **Skeletal animation support** — animated armatures and deformations play back in real-time
- **Texture binding** — shaders can reference diffuse, normal, and metallic maps from USD material definitions

## Audio Reactivity

| Signal  | Effect |
|---------|--------|
| `bass`  | Model scale pulse + emission intensity |
| `mid`   | Camera orbit speed + palette shift rate |
| `treble` | Rim-light brightness + particle jitter |
| `beat`  | Hard camera cut between scenes + bloom spike |

The audio-reactive parameters are weighted by the scene's presentation style and can be modulated by global reactivity settings (F7/Alt+A).

## Configuration

```toml
[effects.SimShowcase]
speed = 1.0
dwell_min = 7.0              # min seconds a scene stays up
dwell_max = 20.0             # max seconds; each scene picks a random dwell in range
camera_distance = 0.8        # <1 pulls the camera in (larger subject)
transition = "fade"          # "fade" (dip-to-black between sims) or "cut"
transition_s = 0.6           # fade duration in seconds (out + in)
dance = 1.0                  # audio-reactive bounce/hop/sway (0=off, >1=wilder)
preload = true               # pre-bake scenes in the background during app splash
scene_dir = ""               # override path; empty = bundled scenes folder
# scene_dirs = "drop-ins/sims-01/scenes,assets/sims"   # extra dirs (comma list)
# --- Skinned/animated USD characters (UsdSkel rigs) ---
skel_animate = true          # bake + play the rig's skeletal animation
skel_frames = 48             # frames baked across the clip (24-60 typical)
skel_speed = 1.0             # animation playback rate
motions_per_character = 1    # dynamic motion clips per character (each a scene)
# skel_pose_time = 0.0       # static pose timecode when skel_animate = false
face_audio_react = false     # name-matched facial blendshapes from audio
# per-effect randomization (same as other effects)
# random_speed_min = 0.85
# random_speed_max = 1.15
```

Skinned characters (e.g. the NVIDIA Reallusion pack) are baked to a frame
sequence and played back — see
[docs/planning-usd-characters.md](docs/planning-usd-characters.md) for the
design and [assets/sims/README.md](../../assets/sims/README.md) for licensed
asset placement. The reusable loader lives in `usd_scene.py`. Note: a pack must
ship **named** facial blendshapes for `face_audio_react` to engage; packs that
export numbered-only blendshapes (like the Reallusion set) play animation but
skip facial reactivity.

### Adding Your Own Scenes

1. Create a folder `drop-ins/sims-01/scenes/`
2. Add your USD/USDC/USDZ files there
3. (Optional) Point `[effects.SimShowcase].scene_dir` to a custom folder path
4. Restart Unicorn Viz; scenes will be auto-discovered and shuffled

**Performance tip:** For best results, keep scene complexity reasonable (under 100k triangles per scene). Large meshes with many bone influences may impact real-time performance on integrated graphics.

## Hotkeys

None; Sim Showcase uses standard effect shortcuts to activate it.

Once active, scenes cycle automatically based on `mix_time`. Use standard hotkeys to:

- `+` / `-` — adjust playback speed
- `[` / `]` — increase/decrease audio reactivity
- `Z` / `Shift+Z` — zoom in/out
- `N` / `P` — jump to next/previous effect (if you want to leave Sim Showcase)

## Presentation Styles

Each presentation style preset combines camera behavior, rim-light hue, palette phase, bloom strength, and film-grain for a distinct mood:

1. **Neon** — fast orbit, high saturation palette, 35% bloom
2. **Void** — slow orbit, dark cold colors, 50% bloom, high grain
3. **Solar** — medium orbit, warm yellows/oranges, 40% bloom
4. **Acid** — very fast orbit, acid greens/magentas, 28% bloom
5. **Frost** — slow elevated camera, icy blues, 22% bloom
6. **Inferno** — medium fast orbit, red/orange/black, 60% bloom
7. **Spectral** — moderate orbit, spectral rainbow shift, 45% bloom
8. **Deep** — very slow orbit, deep purples/blues, 55% bloom
9. **Pulse** — fastest orbit, varied palette with beat pulses, 65% bloom
10. **Dream** — moderate orbit, soft pastels, 35% bloom

Styles cycle automatically with each scene transition.

## Dependencies

### Runtime Requirements

- **Python 3.11+**
- **moderngl** (already required by unicorn-viz)
- **numpy**
- **OpenUSD SDK** (optional; if missing, falls back to procedural galaxy)

### Installing OpenUSD

#### pip (recommended — cross-platform)
```bash
pip install -r drop-ins/sims-01/requirements.txt   # installs usd-core
```
The PyPI `usd-core` wheel provides the `pxr` module on Linux/macOS/Windows.

#### Fedora / Arch Linux
```bash
sudo dnf install python3-pxr      # Fedora
sudo pacman -S opensubdiv-python   # Arch (includes Pixar USD bindings)
```

#### macOS
```bash
brew install pixar-usd
```

#### Windows
Download the pre-built Pixar USD SDK from [pixar.io/downloads](https://pixar.io/downloads) and follow the installation guide.

#### From Source
Clone and build [pixar-usd](https://github.com/PixarAnimationStudios/USD) if your distro doesn't package it. This is complex; only recommended if pre-built packages are unavailable.

If the USD runtime is not available, Sim Showcase will:
- Log a warning at startup
- Display a message in the HUD
- Render a fallback procedural particle galaxy instead

This is a graceful degradation; the effect remains visually interesting and responsive to audio.

## Bundled robots (license-free)

Five procedural, articulated robots ship in `scenes/` and animate out of the box
via time-varying joint transforms (no skeleton): **`robot_arm`** (5-axis
pick/place sweep), **`robot_drone`** (quadcopter with spinning rotors),
**`robot_quadruped`** (trot gait), **`robot_hexapod`** (tripod gait) and
**`robot_hand`** (grasp/release finger curl). They are authored from scratch
(see `tools/gen_robots.py`) so they carry no third-party licence. Regenerate
with:

```bash
python drop-ins/sims-01/tools/gen_robots.py
```

The loader handles both **skinned** (UsdSkel character) animation and **rigid /
articulated** (joint-Xform) animation, so real robot USD assets — e.g. NVIDIA
Isaac or open-source URDF→USD rigs — drop into a scene dir and animate the same
way (subject to their licence; curate restricted packs locally per
`assets/sims/README.md`).

## Interaction with Other Drop-ins

- **Compatible with** all multi-head, webcam, and streaming drop-ins
- **Does NOT conflict** with postfx-01 or auto-vj-01
- Scene rendering is entirely self-contained and does not interfere with other effects or overlays

## Architecture Notes

- **Single VAO per scene** — all geometry from a stage is combined into one GPU buffer for efficiency
- **CPU animation** — skeletal transforms are evaluated on the CPU and streamed each frame; this keeps VRAM usage low but requires CPU work proportional to animation complexity
- **Lazy USD loading** — stages are not loaded until first display (unless `preload_videos = true`)
- **Presentation as stateless decoration** — styles are applied post-render and can change without reloading geometry
- **No persistent state** — scene selections and style rotations are ephemeral per-session

## Troubleshooting

**Scene doesn't appear:**
- Check that the `.usd` file is valid with `pxr.Usd.Stage.Open(path)` in a Python REPL
- Ensure the scene doesn't contain plugins or custom schemas not available in your USD build
- If using relative paths inside the USD file, make sure they resolve correctly

**Slow performance with certain scenes:**
- Reduce animation bone count in your source model (Blender/Maya/Houdini)
- Lower geometry complexity (combine meshes, reduce vertex count)
- Disable `preload_videos` to open scenes on demand instead of at startup

**"USD not available" warning:**
- Install the OpenUSD SDK for your platform (see Dependencies section)
- Verify the Python environment can import `pxr.Usd`
- Restart Unicorn Viz after installing USD

## Technical Details

For developers interested in extending or modifying Sim Showcase:

- Scene mesh data is loaded asynchronously on a background thread to avoid stalling the render loop
- Animation frame data is cached in a dict keyed by (stage_path, frame_index) to amortize USD evaluation cost
- The presentation style system allows swapping aesthetics without reloading geometry (useful for beat-driven theme changes)
- The renderer uses deferred shading with a single forward pass; no G-buffer or shadow maps are employed to keep overhead minimal

## Roadmap

- [Plan: NVIDIA Reallusion USD character sets](docs/planning-usd-characters.md)
  — supporting skinned/animated/blendshape character rigs (Tier 0–3).

## 2026-05-22 Update

- Sim Showcase audio channels now consume normalized band dynamics (`bass_n/mid_n/treble_n`) for more consistent behavior across track loudness swings.

## Changelog

- **0.7.2** — Articulated robots are assembled again. The rigid-animation bake
  applied each mesh's world transform as `points @ M.T`, but USD's `GfMatrix4d`
  is row-vector: a point transforms as `p' = p * M`, so the translation lives
  in the last *row* and the last *column* is `(0, 0, 0, 1)`. Transposing it
  therefore dropped every joint offset outright — the `w` term multiplied a
  column of zeros — and applied the rotation transposed, which for a rotation
  is its inverse. So every link collapsed toward the origin and counter-rotated
  against its own parent. All five rigid robots shipped that way: the arm drove
  its links through its own base, the quadruped's and hexapod's legs shrank to
  stubs under the body, the hand's fingers folded into the palm, and the
  drone's rotors sat inside the chassis. Measured bounding volume was
  0.40-0.49x of correct, with joint offsets out by up to 5.0 world units.
  Skinned characters were never affected; they go through UsdSkel's own
  `ComputeSkinnedPoints`, which is why the earlier character fixes did not
  touch this. Covered by a regression test that checks every transformed mesh
  in all five scenes against USD's own point transform.

- **0.7.1** — fix grid-overlay shader bug that painted blotchy dark patches
  across every mesh (the "subtle wireframe" distance field was measuring
  distance to each grid cell's *centre* instead of its *line*, inverting the
  effect).
- **0.7.0** — initial versioned release.
