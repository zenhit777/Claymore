use claymore_runtime::HardwareProfile;

fn main() {
    let command = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "profile".to_owned());
    match command.as_str() {
        "profile" => {
            show_profile();
        }
        "device" => show_device(),
        "help" | "--help" | "-h" => {
            println!("Claymore commands:");
            println!("  profile   Load and validate the hardware profile (default)");
            println!("            Override the embedded profile with CLAYMORE_HARDWARE_PROFILE=/path/to/profile.json");
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

fn show_profile() {
    const EMBEDDED_PROFILE: &str =
        include_str!("../../../config/rtx5090-7950x3d.json");
    let profile = match std::env::var("CLAYMORE_HARDWARE_PROFILE") {
        Ok(path) => HardwareProfile::load(path),
        Err(_) => HardwareProfile::from_json(EMBEDDED_PROFILE),
    };
    let profile = match profile {
        Ok(profile) => profile,
        Err(error) => {
            eprintln!("hardware_profile_error={error}");
            std::process::exit(1);
        }
    };
    let memory = match profile.memory_budgets() {
        Ok(memory) => memory,
        Err(error) => {
            eprintln!("hardware_profile_error={error}");
            std::process::exit(1);
        }
    };
    let scheduler = match profile.scheduler_config() {
        Ok(scheduler) => scheduler,
        Err(error) => {
            eprintln!("hardware_profile_error={error}");
            std::process::exit(1);
        }
    };

    println!("Claymore v0.5.0");
    println!("profile={}", profile.profile);
    println!("gpu={}", profile.hardware.gpu);
    println!("vram_total_gib={}", profile.hardware.vram_gib);
    println!("vram_soft_budget_bytes={}", memory.vram_soft_bytes);
    println!("vram_hard_ceiling_bytes={}", memory.vram_hard_bytes);
    println!("vram_safety_reserve_gib={}", profile.vram.safety_reserve_gib);
    println!("expert_cache_target_bytes={}", memory.expert_cache_target_bytes);
    println!("expert_cache_max_bytes={}", memory.expert_cache_max_bytes);
    println!("host_budget_bytes={}", memory.host_bytes);
    println!("pinned_host_budget_bytes={}", memory.pinned_host_bytes);
    println!("cpu_workers={}", scheduler.cpu_workers);
    println!("max_prefetches={}", scheduler.max_prefetches);
    println!("lookahead={}", scheduler.lookahead);
    println!("cuda_driver_feature={}", cfg!(feature = "cuda-driver"));
    println!("status=profile_validated");
}
