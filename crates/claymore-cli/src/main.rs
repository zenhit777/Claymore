use claymore_scheduler::SchedulerConfig;

fn main() {
    let config = SchedulerConfig::default();
    println!("Claymore scaffold v0.2");
    println!("scheduler_config_valid={}", config.validate());
    println!("status=policy_scaffold_only");
}
