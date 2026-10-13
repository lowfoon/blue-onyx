use blue_onyx::{
    blue_onyx_service as create_blue_onyx_service, cli::Cli, init_logging,
    system_info::system_info, update_log_level,
};
use tracing::{error, info, warn};

fn main() -> anyhow::Result<()> {
    let Some(mut current_args) = Cli::from_config_and_args()? else {
        return Ok(());
    };
    let _guard = init_logging(current_args.log_level, &mut current_args.log_path)?;
    system_info()?; // Print the configuration being used
    current_args.print_config();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    loop {
        let (service, cancellation_token, _restart_token) =
            create_blue_onyx_service(current_args.clone())?;
        let (should_restart, worker_handle) = rt.block_on(async {
            tokio::pin!(service);
            tokio::select! {
                biased;
                signal = &mut shutdown => {
                    signal?;
                    info!("Shutdown signal received, draining requests");
                    cancellation_token.cancel();
                    service.await.map(|(_, handle)| (false, handle))
                }
                result = &mut service => result,
            }
        })?;
        // The server has released its queue sender; the worker can now finish.
        if let Some(handle) = worker_handle
            && let Err(error) = handle.join()
        {
            error!(?error, "Worker thread panicked");
        }
        if should_restart {
            info!("Restarting server with updated configuration...");

            // Reload configuration for restart
            let new_args = Cli::from_config_and_args()?.expect("Should always have args");

            // Check if log level changed and update dynamically
            if new_args.log_level != current_args.log_level {
                info!(
                    old_level = ?current_args.log_level,
                    new_level = ?new_args.log_level,
                    "Log level change detected, applying dynamically"
                );

                if let Err(e) = update_log_level(new_args.log_level) {
                    warn!("Failed to update log level dynamically: {}", e);
                }
            }

            current_args = new_args;
            continue;
        } else {
            break; // Normal shutdown
        }
    }

    Ok(())
}

async fn shutdown_signal() -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result?,
            _ = terminate.recv() => {},
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;
    Ok(())
}
