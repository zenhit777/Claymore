use claymore_scheduler::SchedulerConfig;

fn main() {
    let command = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "profile".to_owned());
    match command.as_str() {
        "profile" => {
            let config = SchedulerConfig::default();
            println!("Claymore v0.4.0");
            println!("scheduler_config_valid={}", config.validate());
            println!("cuda_driver_feature={}", cfg!(feature = "cuda-driver"));
            println!("status=experimental_runtime");
        }
        "device" => show_device(),
        "help" | "--help" | "-h" => {
            println!("Claymore commands:");
            println!("  profile   Show the active static runtime profile (default)");
            println!("  device    Query CUDA device 0 (requires --features cuda-driver)");
        }
        other => {
            eprintln!("unknown command: {other}");
            eprintln!("run 'claymore-cli help' for usage");
            std::process::exit(2);
        }
    }
}

#[cfg(feature = "cuda-driver")]
fn show_device() {
    match claymore_cuda::CudaDriver::new(0) {
        Ok(driver) => {
            let info = driver.device_info();
            println!("device_name={}", info.name);
            println!("device_ordinal={}", info.ordinal);
            println!("device_total_bytes={}", info.total_memory_bytes);
            match driver.memory_info() {
                Ok((free, total)) => {
                    println!("memory_free_bytes={free}");
                    println!("memory_total_bytes={total}");
                }
                Err(error) => eprintln!("memory_query_error={error}"),
            }
        }
        Err(error) => {
            eprintln!("cuda_device_error={error}");
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "cuda-driver"))]
fn show_device() {
    eprintln!("CUDA driver support is disabled.");
    eprintln!("Rebuild with: cargo run -p claymore-cli --features cuda-driver -- device");
    std::process::exit(2);
}
