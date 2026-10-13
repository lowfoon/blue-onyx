# Linux fork: build, test, and deploy

This is the first development iteration of [lowfoon/blue-onyx](https://github.com/lowfoon/blue-onyx), based on upstream commit `1cd4d93`. See [the source analysis](REVERSE_ENGINEERING.md) for the architecture, confirmed bugs, remaining work, and validation limits.

## CPU or NVIDIA

Run from the repository root on Linux x86_64 with Docker Compose:

```sh
# CPU reference deployment
docker compose -f deployment/docker-compose.cpu.yml up -d --build

# NVIDIA deployment: requires working host driver and NVIDIA Container Toolkit
docker compose -f deployment/docker-compose.cuda.yml up -d --build
```

The NVIDIA image is paired with ONNX Runtime 1.29's CUDA 12 build, CUDA 12.9, and cuDNN 9. Follow [NVIDIA's Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to configure the host runtime. Verify `nvidia-smi` on the host and GPU visibility in a vendor test container before testing this service.

The CUDA build fails detector initialization when provider registration fails. Check `/stats` for `CUDA(GPU 0)`, the container logs for provider errors, and host GPU utilization during requests. A provider label alone does not prove all graph nodes execute on GPU.

Do not run CPU and GPU examples simultaneously on the same port. The Compose files use named volumes for configuration and models. First startup downloads `rf-detr-n.onnx` and its YAML; internet access is required for that download. Later starts reuse the volume. Initial downloads can exceed the 180-second healthcheck start period on a slow connection; inspect logs and allow extra time. Docker marking a container unhealthy does not automatically restart it.

## Intel OpenVINO

```sh
# Use the group ID of the actual render device; include other groups if your driver requires them.
RENDER_GID=$(stat -c '%g' /dev/dri/renderD128) \
  docker compose -f deployment/docker-compose.yml up -d --build
```

This builds the existing OpenVINO source pipeline, mounts the render device, adds the host render group, and persists compiled-model cache, models, and config. The initial native build is substantially slower than CPU/CUDA's prebuilt-runtime path. Confirm `OpenVINO(GPU 0)` in `/stats` and actual Intel GPU activity before accepting the deployment.

## Reach the service

Compose binds to `127.0.0.1` by default. For a Blue Iris / Agent DVR machine elsewhere on a trusted LAN, set `BLUE_ONYX_BIND_IP` to the Linux host's LAN address, e.g.:

```sh
BLUE_ONYX_BIND_IP=192.168.1.20 \
  docker compose -f deployment/docker-compose.cuda.yml up -d
```

Restrict access to the intended clients. The inherited configuration/restart routes are unauthenticated, so an unrestricted published port also exposes administration. Remote authentication is roadmap work.

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

For a CUDA Linux build, use the CUDA-enabled runtime directory and add `--features cuda`. For Intel, use matching OpenVINO libraries and `--features openvino`. Select only one GPU feature per build. Keep runtime and provider libraries together, and set `ORT_DYLIB_PATH` to the full runtime-library path when launching from another directory.

On Unix, run an actual CPU inference/lifecycle check:

```sh
ORT_DYLIB_PATH=/absolute/path/to/onnxruntime/lib/libonnxruntime.so \
  python3 scripts/smoke_test.py --binary target/debug/blue_onyx
```

Use `.dylib` on macOS. The script uses a temporary config/model directory and ephemeral loopback port, downloads the default model, checks detection JSON, and verifies clean SIGTERM exit. `--model /path/to/model.onnx` can reuse an existing RF-DETR model and YAML. It does not benchmark GPU performance.

## Before publishing

Run Linux builds and actual NVIDIA/Intel inference on the intended hardware. Migrate update/installer endpoints to this fork and audit the inherited release workflow: it references an absent `install_latest_blue_onyx.ps1`. Document model redistribution licenses. The containers retain the service MIT license and ONNX Runtime license/notices; model and provider redistribution requirements still need their own records. The new Docker publish workflow builds the exact published release tag; manual dispatch only validates builds. No release image is claimed as available yet.
