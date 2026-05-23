# sims-01 — 3D Scene Showcase

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
mix_time = 30.0              # seconds before auto-advancing to next scene
scene_dir = ""               # override path; empty = bundled scenes folder
preload_videos = false       # true = open all scenes at startup (uses more memory)
# per-effect randomization (same as other effects)
# random_speed_min = 0.85
# random_speed_max = 1.15
```

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

## 2026-05-22 Update

- Sim Showcase audio channels now consume normalized band dynamics (`bass_n/mid_n/treble_n`) for more consistent behavior across track loudness swings.
