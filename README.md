# InferenceVisualizer

A desktop application for inspecting computer-vision inference results. It loads
images and videos and shows the **original** alongside the **processed** output —
the original on the left, the model's segmentation overlay on the right — using
ONNX models run through ONNX Runtime.

The project was built incrementally as a hands-on way to learn **Qt Widgets** and
**modern C++**, with each step kept small enough to build, run, and understand
before moving on. Its first real workload is an industrial-inspection use case:
detecting **critical corrosion** on steel pieces so those regions can be flagged
for manual shot-blasting before the automated process.

![InferenceVisualizer running on a corroded steel piece — original on the left,
critical-corrosion segmentation overlay with a confidence chip on the right](docs/images/screenshot-corrosion.png)

> *Left: the original frame. Right: the model's overlay — the red mask marks the
> region classified as critical corrosion, with a confidence chip showing how
> sure the model is.*

---

## Features

- **Side-by-side viewer** — a draggable split view: original frame vs. processed result.
- **Images and video** — open still images, or play video files (play / pause / scrub).
  The video controls hide automatically in image mode.
- **ONNX inference** via ONNX Runtime, with two model families behind a common interface:
  - **YOLOv8 / YOLO11 instance segmentation** — confidence filter, non-max suppression,
    per-instance masks. Shows the single most-confident detection with its mask and a
    `NN%` confidence chip.
  - **UNet semantic segmentation** — per-pixel argmax with a probability floor, coloured
    mask, and a chip showing the mean confidence of the highlighted region.
- **Settings dialog** (`Inference → Settings…`) — choose the model family, adjust mask
  opacity, set the per-class overlay colours, and tune the confidence threshold. Settings
  persist across launches via `QSettings`.
- **Live preview** — settings changes re-render the current image instantly.
- **Smooth video under slow inference** — frames are queued in order (never dropped) and
  the video source is paused/resumed via watermark backpressure, so every frame is
  processed even when inference is slower than real time.
- **Cross-platform** — builds and runs on Windows (MinGW) and Linux (GCC).

---

## Tech stack

| Area | Choice |
|---|---|
| Language | C++17 |
| GUI | Qt 6 Widgets (hand-coded, no Qt Designer `.ui` files) |
| Media | Qt Multimedia (`QMediaPlayer` + `QVideoSink`) |
| Inference | ONNX Runtime (CPU) |
| Build | CMake + Ninja |
| Platforms | Windows (MinGW) · Linux (GCC) |

---

## Getting started

### Prerequisites

- **Qt 6.5+** (built with 6.11) — Widgets + Multimedia.
- **ONNX Runtime** (Microsoft's prebuilt CPU package).
- **CMake 3.21+**, **Ninja**, and a C++17 compiler (MinGW on Windows, GCC on Linux).

**Windows:** extract ONNX Runtime to `C:\Tools\onnxruntime`.
**Linux:** extract it to `~/onnxruntime` (`sudo apt install qt6-base-dev qt6-multimedia-dev`
for Qt). Override the location with `-DONNXRUNTIME_DIR=...` if you put it elsewhere.

Full, step-by-step setup for both platforms — including the ONNX Runtime download links
and the VS Code launch configuration — is in [`docs/index.html`](docs/index.html) §6.

### Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

On Windows the build copies `onnxruntime.dll` next to the executable automatically.
On Linux, set `LD_LIBRARY_PATH` to include `~/onnxruntime/lib` when running.

### Run with a model

The app auto-loads a model named **`model_tcc.onnx`** placed next to the executable
(`build/bin/`) at startup. Otherwise use **`Inference → Load Model…`** (Ctrl+M) to pick any
`.onnx` file. Then open an image (Ctrl+O) or a video (Ctrl+Shift+O) and toggle
**`Inference → Run Inference`** (Ctrl+R).

> The model file itself is not in the repository — trained `.onnx` files live under
> `models/` (git-ignored). Export your segmentation model to ONNX (e.g.
> `yolo export format=onnx` for an Ultralytics seg model, or `torch.onnx.export` for a
> UNet) and drop it in.

---

## Project structure

```
InferenceVisualizer/
├── CMakeLists.txt            # finds Qt + ONNX Runtime
├── src/
│   ├── main.cpp              # entry point + file logger
│   ├── app/                  # MainWindow, InferenceController (orchestration)
│   ├── media/                # VideoController (QMediaPlayer + QVideoSink)
│   ├── inference/            # IInferenceBackend, OnnxRunner, YOLO + UNet backends
│   └── ui/                   # ImageView, VideoControls, SettingsDialog
├── docs/                     # learning documentation (see below)
└── models/                   # trained .onnx files (git-ignored)
```

---

## Documentation

This is a learning project, so it carries two pieces of HTML documentation that grew
alongside the code:

- **[`docs/index.html`](docs/index.html)** — the *project doc*: what was built and why,
  the architectural decisions (with code snippets), build instructions for both platforms,
  and a changelog.
- **[`docs/concepts.html`](docs/concepts.html)** — a *Qt & modern-C++ tutorial* organised by
  topic, using this codebase as the worked example: the Qt object system, signals/slots,
  the multimedia pipeline, the inference pipeline, `QPainter` overlays, `QSettings`,
  deployment on Windows, and a catalogue of pitfalls.

Open either file in a browser; they share `docs/style.css`.

---

## Status

| Step | Scope | State |
|---|---|---|
| 1 | Base app: window, menus, open & display image | ✅ |
| 2 | Video: playback, dual-pane layout | ✅ |
| 3 | ONNX inference: segmentation overlay, settings, model switching | ✅ |
| 4 | Polish: worker-thread inference, GPU, drag-and-drop, FPS, recording | future |

Inference currently runs on the GUI thread; for CPU-only video it plays in bursts via the
backpressure mechanism. Moving inference to a worker thread (and optional GPU execution
providers) is the main remaining item.
