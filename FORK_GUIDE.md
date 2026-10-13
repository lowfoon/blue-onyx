# Linux fork: build, test, and deploy

This is the first development iteration of [lowfoon/blue-onyx](https://github.com/lowfoon/blue-onyx), based on upstream commit `1cd4d93`. See [the source analysis](REVERSE_ENGINEERING.md) for the architecture, confirmed bugs, remaining work, and validation limits. [Intel-focused CI](https://github.com/lowfoon/blue-onyx/actions/runs/38373206951) passed all seven jobs, including CPU, CUDA and OpenVINO container builds/lifecycle checks. Actual GPU inference and NVR integration still require on-device testing.

## Intel OpenVINO

```sh
# Use the group ID of the actual render device; include other groups if your driver requires them.
RENDER_GID=$(stat -c '%g' /dev/dri/renderD128) \
  docker compose -f deployment/docker-compose.yml up -d --build
```

This builds the existing OpenVINO source pipeline, mounts the render device, adds the host render group, and persists compiled-model cache, models, and config. The initial native build is substantially slower than CPU/CUDA's prebuilt-runtime path. Confirm `OpenVINO(GPU 0)` in `/stats` and actual Intel GPU activity before accepting the deployment.

## CPU or NVIDIA

Run from the repository root on Linux x86_64 with Docker Compose:

```sh
# CPU reference deployment
docker compose -f deployment/docker-compose.cpu.yml up -d --build

# NVIDIA deployment: requires working host driver and NVIDIA Container Toolkit
docker compose -f deployment/docker-compose.cuda.yml up -d --build
```

The Intel OpenVINO deployment is the primary target; NVIDIA remains an optional alternative. The NVIDIA image is paired with ONNX Runtime 1.29's CUDA 12 build, CUDA 12.9, and cuDNN 9. Follow [NVIDIA's Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to configure the host runtime. Verify `nvidia-smi` on the host and GPU visibility in a vendor test container before testing this service.

The CUDA build fails detector initialization when provider registration fails. Check `/stats` for `CUDA(GPU 0)`, the container logs for provider errors, and host GPU utilization during requests. A provider label alone does not prove all graph nodes execute on GPU.

## Experimental WebGPU / Radeon

The `webgpu` build uses the native ONNX Runtime plugin through Vulkan. Its dedicated image packages hash-verified ONNX Runtime **1.31.0** and WebGPU plugin **0.4.0**; the other images retain their existing runtimes. Linux x86_64 Radeon 660M inference has been prototyped with the medium RT-DETR v2 model. Other models and GPUs need their own validation.

Provide an existing model directory containing `rt-detrv2-m.onnx` (and its class YAML when required):

```sh
RENDER_GID=$(stat -c '%g' /dev/dri/renderD128) \
  BLUE_ONYX_MODELS_DIR=/srv/blueonyx/models \
  docker compose -f deployment/docker-compose.webgpu.yml up -d --build
```

The example explicitly selects the RT-DETR v2 adapter and binds to loopback port **32170** for a separate trial deployment. Set `BLUE_ONYX_BIND_IP` and `BLUE_ONYX_HOST_PORT` if needed. It shares the render device, runs as an ordinary container user, persists configuration and shader cache, and mounts the supplied model directory read-only.

Check readiness and confirm `WebGPU(GPU 0)` in `/stats`, and `WebGPU`/`canUseGPU: true` in detection responses. Core/plugin registration errors, missing plugin files and unavailable GPU indices fail initialization rather than silently selecting CPU. Ordinary unsupported-node CPU partitioning remains possible; measure actual graph execution and timing. `--force-cpu` bypasses plugin initialization for comparison. Graph capture is disabled because the tested graph has CPU shape operations.

For a native Linux build:

```sh
python3 scripts/fetch_webgpu_runtime.py --output /tmp/blueonyx-webgpu-ort
BLUE_ONYX_ORT_LIB_DIR=/tmp/blueonyx-webgpu-ort \
  cargo build --locked --release --bin blue_onyx --features webgpu
ORT_DYLIB_PATH=/tmp/blueonyx-webgpu-ort/libonnxruntime.so \
  BLUE_ONYX_WEBGPU_PLUGIN_PATH=/tmp/blueonyx-webgpu-ort/libonnxruntime_providers_webgpu.so \
  target/release/blue_onyx --model /srv/blueonyx/models/rt-detrv2-m.onnx \
    --object-detection-model-type rt-detrv2 --port 32170
```

Select only one GPU feature per build. CPU-node fallback does not mean the whole graph is on CPU; readiness/provider labels alone do not establish GPU throughput or model accuracy.

Do not run CPU and GPU examples simultaneously on the same port. The Compose files use named volumes for configuration and models. First startup downloads `rf-detr-n.onnx` and its YAML; internet access is required for that download. Later starts reuse the volume. Initial downloads can exceed the 180-second healthcheck start period on a slow connection; inspect logs and allow extra time. Docker marking a container unhealthy does not automatically restart it.

## Reach the service

Compose binds to `127.0.0.1` by default. For a Blue Iris / Agent DVR machine elsewhere on a trusted LAN, set `BLUE_ONYX_BIND_IP` to the Linux host's LAN address, e.g.:

```sh
RENDER_GID=$(stat -c '%g' /dev/dri/renderD128) \
  BLUE_ONYX_BIND_IP=192.168.1.20 \
  docker compose -f deployment/docker-compose.yml up -d
```

Set `RENDER_GID` as in the Intel example when using that Compose file. Restrict access to the intended clients. The inherited configuration/restart routes are unauthenticated, so an unrestricted published port also exposes administration. Remote authentication is roadmap work.

```sh
curl -f http://127.0.0.1:32168/health/live
curl -f http://127.0.0.1:32168/health/ready
curl -F 'image=@assets/dog_bike_car.jpg' -F 'min_confidence=0.3' \
  http://127.0.0.1:32168/v1/vision/detection
```

Set the NVR's AI server URL to the Linux host and port 32168. Use the compatible `/v1/vision/detection` route. Requests accept JPEG uploads. There is no JSON image-URL fetch API.

## Models and configuration

The images use:

| Setting | Container path |
| --- | --- |
| `BLUE_ONYX_CONFIG_PATH` | `/app/config/blue_onyx_config.json` |
| `BLUE_ONYX_MODEL_DIR` | `/app/models` |
| `ORT_DYLIB_PATH` | `/app/ort-libs/libonnxruntime.so` |

The web UI at `/config` can save and restart. CLI overrides still use inherited default-value comparison: explicitly specifying a value equal to the CLI default may not override a saved nondefault setting. Until this is corrected, change that setting in the config file/UI and recreate or restart as appropriate.

Custom models are resolved exactly where `--model` points. Mount the model directory and pass matching adapter/class options. For example, add a bind mount `/srv/camera-models:/custom-models:ro` and a Compose command:

```yaml
command:
  - --model
  - /custom-models/delivery.onnx
  - --object-detection-model-type
  - yolo5
  - --object-classes
  - /custom-models/delivery.yaml
```

A companion `.yaml` is used when an explicit class file is not provided. The class file requires `NAMES` with labels in output-index order. `--force-cpu` bypasses the compiled GPU provider. `--gpu-index` chooses the index inside the container's visible GPU set.

## Native build and tests

Install Rust stable and download the official ONNX Runtime 1.29 archive for the build host. Point `BLUE_ONYX_ORT_LIB_DIR` at its `lib` directory to avoid building the native runtime from source:

```sh
export BLUE_ONYX_ORT_LIB_DIR=/absolute/path/to/onnxruntime/lib
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
cargo fmt --all --check
cargo build --locked --bin blue_onyx
```

For a CUDA Linux build, use the CUDA-enabled runtime directory and add `--features cuda`. For Intel, use matching OpenVINO libraries and `--features openvino`. For native WebGPU, use the pinned runtime/plugin setup above and `--features webgpu`. Select only one GPU feature per build. Keep runtime and provider libraries together, and set `ORT_DYLIB_PATH` to the full runtime-library path when launching from another directory.

On Unix, run an actual CPU inference/lifecycle check:

```sh
ORT_DYLIB_PATH=/absolute/path/to/onnxruntime/lib/libonnxruntime.so \
  python3 scripts/smoke_test.py --binary target/debug/blue_onyx
```

Use `.dylib` on macOS. The script uses a temporary config/model directory and ephemeral loopback port, downloads the default model, checks detection JSON, and verifies clean SIGTERM exit. `--model /path/to/model.onnx` can reuse an existing RF-DETR model and YAML. It does not benchmark GPU performance.

## Before publishing

Run Linux builds and actual NVIDIA/Intel inference on the intended hardware. Migrate update/installer endpoints to this fork and audit the inherited release workflow: it references an absent `install_latest_blue_onyx.ps1`. Document model redistribution licenses. The containers retain the service MIT license and ONNX Runtime license/notices; model and provider redistribution requirements still need their own records. The new Docker publish workflow builds the exact published release tag; manual dispatch only validates builds. No release image is claimed as available yet.
