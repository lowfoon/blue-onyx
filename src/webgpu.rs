//! Native WebGPU plugin registration. A missing plugin/device fails initialization.
use anyhow::{Context, bail};
use ort::{environment::Environment, memory::DeviceType, session::builder::SessionBuilder};
use std::{path::PathBuf, sync::Mutex};
use tracing::info;

// ORT registrations belong to the process environment, not an individual model.
// Reuse the registration across configuration/model restarts, and retry failures.
static REGISTERED_PLUGIN: Mutex<Option<PathBuf>> = Mutex::new(None);

pub(crate) fn configure_session(
    builder: SessionBuilder,
    gpu_index: i32,
) -> anyhow::Result<SessionBuilder> {
    let index = usize::try_from(gpu_index).context("WebGPU index must be non-negative")?;
    let path = std::env::var_os("BLUE_ONYX_WEBGPU_PLUGIN_PATH")
        .context("Set BLUE_ONYX_WEBGPU_PLUGIN_PATH to libonnxruntime_providers_webgpu.so")?;
    let path = std::fs::canonicalize(PathBuf::from(path))
        .context("WebGPU plugin library is missing or inaccessible")?;
    let env =
        Environment::current().context("Failed to initialize the ONNX Runtime environment")?;
    {
        let mut registered = REGISTERED_PLUGIN
            .lock()
            .map_err(|_| anyhow::anyhow!("WebGPU registration lock was poisoned"))?;
        if let Some(existing) = registered.as_ref() {
            if existing != &path {
                bail!(
                    "WebGPU plugin path changed; restart the process to load a different library"
                );
            }
        } else {
            env.register_ep_library("blue_onyx_webgpu", &path)
                .context("Failed to register WebGPU plugin; check core/plugin compatibility")?;
            *registered = Some(path);
        }
    }

    // Linux hardware discovery uses PCI device IDs. Software adapters have no
    // physical PCI ID; do not advertise them as hardware GPU acceleration.
    let devices: Vec<_> = env
        .devices()
        .filter(|device| {
            matches!(device.ep(), Ok("WebGpuExecutionProvider"))
                && device.hardware_device().ty() == DeviceType::GPU
                && device.hardware_device().id() != 0
        })
        .collect();
    let count = devices.len();
    let device = devices.into_iter().nth(index).with_context(|| {
        format!("WebGPU GPU index {index} is unavailable ({count} physical GPU devices discovered)")
    })?;
    info!(
        gpu_index,
        device_id = device.hardware_device().id(),
        "Configuring native WebGPU execution provider"
    );
    let options = vec![
        (
            "WebGpuExecutionProvider.preferredLayout".to_owned(),
            "NHWC".to_owned(),
        ),
        (
            "WebGpuExecutionProvider.enableGraphCapture".to_owned(),
            "0".to_owned(),
        ),
    ];
    builder
        .with_devices([device], Some(&options))
        .map_err(ort::Error::<()>::from)
        .context("Failed to configure the selected WebGPU device")
}
