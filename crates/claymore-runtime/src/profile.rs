//! Validated hardware profile loading for runtime and scheduler configuration.
use std::path::Path;

use claymore_memory::MemoryBudgets;
use claymore_scheduler::SchedulerConfig;
use serde::Deserialize;
use thiserror::Error;

const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum HardwareProfileError {
    #[error("failed to read hardware profile {path}: {reason}")]
    Read { path: String, reason: String },
    #[error("invalid hardware profile JSON: {0}")]
    Parse(String),
    #[error("invalid hardware profile: {0}")]
    Invalid(&'static str),
    #[error("hardware profile byte conversion overflow")]
    SizeOverflow,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct HardwareProfile {
    pub profile: String,
    pub hardware: HardwareInfo,
    pub precision: PrecisionPolicy,
    pub vram: VramProfile,
    pub host_memory: HostMemoryProfile,
    pub scheduler: SchedulerProfile,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct HardwareInfo {
    pub gpu: String,
    pub vram_gib: u64,
    pub cpu: String,
    pub ram_gib: u64,
    pub environment: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct PrecisionPolicy {
    pub policy: String,
    pub preferred: String,
    pub sensitive_accumulation: String,
    pub fallback: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct VramProfile {
    pub soft_budget_gib: u64,
    pub hard_ceiling_gib: u64,
    pub safety_reserve_gib: u64,
    pub model_working_set_gib: u64,
    pub state_kv_canvas_gib: u64,
    pub cuda_workspace_gib: u64,
    pub transfer_gib: u64,
    pub allocator_slack_gib: u64,
    pub expert_cache_target_gib: u64,
    pub expert_cache_max_gib: u64,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct HostMemoryProfile {
    pub claymore_budget_gib: u64,
    pub weights_pool_target_gib: u64,
    pub pinned_ceiling_gib: u64,
    pub staging_and_state_gib: u64,
    pub system_reserve_gib: u64,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct SchedulerProfile {
    pub cpu_workers: usize,
    pub cpu_workers_min: usize,
    pub cpu_workers_max: usize,
    pub cuda_compute_streams: usize,
    pub transfer_streams: usize,
    pub max_prefetches: usize,
    pub lookahead: usize,
    pub tick_us: u64,
}

impl HardwareProfile {
    pub fn from_json(json: &str) -> Result<Self, HardwareProfileError> {
        let profile: Self = serde_json::from_str(json)
            .map_err(|error| HardwareProfileError::Parse(error.to_string()))?;
        profile.validate()?;
        Ok(profile)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, HardwareProfileError> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path).map_err(|error| HardwareProfileError::Read {
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
        Self::from_json(&json)
    }

    pub fn validate(&self) -> Result<(), HardwareProfileError> {
        if self.profile.trim().is_empty()
            || self.hardware.gpu.trim().is_empty()
            || self.hardware.cpu.trim().is_empty()
            || self.hardware.environment.trim().is_empty()
        {
            return Err(HardwareProfileError::Invalid("profile and hardware identifiers must not be empty"));
        }
        if self.hardware.vram_gib == 0 || self.hardware.ram_gib == 0 {
            return Err(HardwareProfileError::Invalid("physical memory capacities must be positive"));
        }
        if self.vram.soft_budget_gib == 0
            || self.vram.soft_budget_gib > self.vram.hard_ceiling_gib
            || self.vram.hard_ceiling_gib.saturating_add(self.vram.safety_reserve_gib) > self.hardware.vram_gib
        {
            return Err(HardwareProfileError::Invalid("VRAM soft/hard/reserve limits are inconsistent"));
        }
        let planned_working_set = self.vram.model_working_set_gib
            .saturating_add(self.vram.state_kv_canvas_gib)
            .saturating_add(self.vram.cuda_workspace_gib)
            .saturating_add(self.vram.transfer_gib)
            .saturating_add(self.vram.allocator_slack_gib);
        if planned_working_set > self.vram.soft_budget_gib {
            return Err(HardwareProfileError::Invalid("planned VRAM working set exceeds the soft budget"));
        }
        if self.vram.expert_cache_target_gib > self.vram.expert_cache_max_gib
            || self.vram.expert_cache_max_gib > self.vram.model_working_set_gib
        {
            return Err(HardwareProfileError::Invalid("expert cache must fit inside the model working set"));
        }
        let host_committed = self.host_memory.claymore_budget_gib
            .saturating_add(self.host_memory.system_reserve_gib);
        let host_working_set = self.host_memory.weights_pool_target_gib
            .saturating_add(self.host_memory.staging_and_state_gib);
        if host_committed > self.hardware.ram_gib
            || host_working_set > self.host_memory.claymore_budget_gib
            || self.host_memory.pinned_ceiling_gib > self.host_memory.claymore_budget_gib
        {
            return Err(HardwareProfileError::Invalid("host memory budgets exceed their enclosing limits"));
        }
        if self.scheduler.cpu_workers_min == 0
            || self.scheduler.cpu_workers_min > self.scheduler.cpu_workers
            || self.scheduler.cpu_workers > self.scheduler.cpu_workers_max
            || self.scheduler.cpu_workers_max > 12
            || self.scheduler.max_prefetches > 2
            || self.scheduler.cuda_compute_streams == 0
            || self.scheduler.transfer_streams == 0
            || self.scheduler.tick_us == 0
        {
            return Err(HardwareProfileError::Invalid("scheduler limits are inconsistent"));
        }
        if !["bf16", "fp16", "fp32"].contains(&self.precision.preferred.as_str())
            || !["bf16", "fp16", "fp32"].contains(&self.precision.sensitive_accumulation.as_str())
            || !["bf16", "fp16", "fp32"].contains(&self.precision.fallback.as_str())
        {
            return Err(HardwareProfileError::Invalid("precision values must be bf16, fp16 or fp32"));
        }
        Ok(())
    }

    pub fn memory_budgets(&self) -> Result<MemoryBudgets, HardwareProfileError> {
        self.validate()?;
        Ok(MemoryBudgets {
            vram_soft_bytes: gib(self.vram.soft_budget_gib)?,
            vram_hard_bytes: gib(self.vram.hard_ceiling_gib)?,
            host_bytes: gib(self.host_memory.claymore_budget_gib)?,
            pinned_host_bytes: gib(self.host_memory.pinned_ceiling_gib)?,
            expert_cache_target_bytes: gib(self.vram.expert_cache_target_gib)?,
            expert_cache_max_bytes: gib(self.vram.expert_cache_max_gib)?,
        })
    }

    pub fn scheduler_config(&self) -> Result<SchedulerConfig, HardwareProfileError> {
        self.validate()?;
        Ok(SchedulerConfig {
            cpu_workers: self.scheduler.cpu_workers,
            max_prefetches: self.scheduler.max_prefetches,
            lookahead: self.scheduler.lookahead,
        })
    }
}

fn gib(value: u64) -> Result<u64, HardwareProfileError> {
    value.checked_mul(GIB).ok_or(HardwareProfileError::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILE: &str = include_str!("../../../config/rtx5090-7950x3d.json");

    #[test]
    fn reference_rtx5090_profile_loads_and_converts_to_runtime_budgets() {
        let profile = HardwareProfile::from_json(PROFILE).unwrap();
        let memory = profile.memory_budgets().unwrap();
        let scheduler = profile.scheduler_config().unwrap();

        assert_eq!(memory.vram_soft_bytes, 26 * GIB);
        assert_eq!(memory.vram_hard_bytes, 28 * GIB);
        assert_eq!(memory.expert_cache_target_bytes, 14 * GIB);
        assert_eq!(memory.expert_cache_max_bytes, 16 * GIB);
        assert_eq!(memory.host_bytes, 88 * GIB);
        assert_eq!(memory.pinned_host_bytes, 8 * GIB);
        assert_eq!(scheduler.cpu_workers, 10);
        assert_eq!(scheduler.max_prefetches, 2);
        assert_eq!(scheduler.lookahead, 1);
    }

    #[test]
    fn profile_rejects_expert_cache_outside_model_working_set() {
        let mut profile = HardwareProfile::from_json(PROFILE).unwrap();
        profile.vram.expert_cache_max_gib = profile.vram.model_working_set_gib + 1;
        assert_eq!(
            profile.validate(),
            Err(HardwareProfileError::Invalid(
                "expert cache must fit inside the model working set"
            ))
        );
    }

    #[test]
    fn profile_rejects_working_set_over_soft_vram_budget() {
        let mut profile = HardwareProfile::from_json(PROFILE).unwrap();
        profile.vram.allocator_slack_gib += 1;
        assert_eq!(
            profile.validate(),
            Err(HardwareProfileError::Invalid(
                "planned VRAM working set exceeds the soft budget"
            ))
        );
    }
}
