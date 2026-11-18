//! Scheduling policy primitives. No CUDA streams are created here.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    pub cpu_workers: usize,
    pub max_prefetches: usize,
    pub lookahead: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            cpu_workers: 10,
            max_prefetches: 2,
            lookahead: 1,
        }
    }
}

impl SchedulerConfig {
    pub fn validate(&self) -> bool {
        (1..=12).contains(&self.cpu_workers) && self.max_prefetches <= 2
    }
}
