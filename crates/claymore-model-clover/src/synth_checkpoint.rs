//! Deterministic synthetic Clover checkpoints for M11.4 CPU tests.
//!
//! The writer emits the exact tensor set that [`crate::CloverCpuBackend`]
//! resolves for a given (reduced) [`crate::CloverConfig`], using real
//! Safetensors files so every production read path is exercised. Values are
//! deterministic in tensor name and element index; individual tensors can be
//! overridden for numerics fixtures. This module is test infrastructure, not a
//! model checkpoint: it must never be presented as model evidence.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::config::{CloverConfig, LayerType};

pub const SYNTH_EMBED: &str = "model.embed_tokens.weight";
pub const SYNTH_FINAL_NORM: &str = "model.norm.weight";
pub const SYNTH_LM_HEAD: &str = "lm_head.weight";
pub const SYNTH_MODEL_SHARD: &str = "model-00001-of-00001.safetensors";
pub const SYNTH_CONV_SIDECAR: &str = "bd_gdn_conv_right_weight.safetensors";
pub const SYNTH_SCALAR_SIDECAR: &str = "bd_gdn_softmax_scalars.safetensors";

pub fn bf16_bits(value: f32) -> u16 {
    claymore_tensor::Bf16::from_f32(value).to_bits()
}

/// Deterministic small BF16 pattern in `[-0.5, 0.5]` per tensor element.
pub fn synth_value(name: &str, index: usize) -> u16 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_1b99);
    }
    hash ^= index as u64;
    hash = hash.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    hash ^= hash >> 29;
    let scaled = ((hash >> 32) % 1024) as f32 / 1023.0 - 0.5;
    bf16_bits(scaled)
}

fn gdn_tensor_names(layer: usize) -> Vec<(String, &'static str)> {
    let prefix = format!("model.layers.{layer}.linear_attn");
    [
        ("conv1d", "conv1d.weight"),
        ("in_proj_qkv", "in_proj_qkv.weight"),
        ("in_proj_z", "in_proj_z.weight"),
        ("in_proj_a", "in_proj_a.weight"),
        ("in_proj_b", "in_proj_b.weight"),
        ("A_log", "A_log"),
        ("dt_bias", "dt_bias"),
        ("norm", "norm.weight"),
        ("out_proj", "out_proj.weight"),
        ("conv_right", "conv_right_weight"),
        ("logit_scale", "bd_softmax_logit_scale"),
        ("history_bias", "bd_softmax_history_bias"),
        ("history_c", "bd_softmax_history_c"),
    ]
    .into_iter()
    .map(|(suffix, leaf)| (format!("{prefix}.{leaf}"), suffix))
    .collect()
}

fn attention_tensor_names(layer: usize) -> Vec<(String, &'static str)> {
    let prefix = format!("model.layers.{layer}.self_attn");
    [
        ("q_proj", "q_proj.weight"),
        ("k_proj", "k_proj.weight"),
        ("v_proj", "v_proj.weight"),
        ("o_proj", "o_proj.weight"),
        ("q_norm", "q_norm.weight"),
        ("k_norm", "k_norm.weight"),
    ]
    .into_iter()
    .map(|(suffix, leaf)| (format!("{prefix}.{leaf}"), suffix))
    .collect()
}

fn mlp_tensor_names(layer: usize) -> Vec<(String, &'static str)> {
    let prefix = format!("model.layers.{layer}.mlp");
    [
        ("gate", "gate.weight"),
        ("gate_up", "experts.gate_up_proj"),
        ("down", "experts.down_proj"),
        ("shared_gate", "shared_expert.gate_proj.weight"),
        ("shared_up", "shared_expert.up_proj.weight"),
        ("shared_down", "shared_expert.down_proj.weight"),
        ("shared_gate_proj", "shared_expert_gate.weight"),
    ]
    .into_iter()
    .map(|(suffix, leaf)| (format!("{prefix}.{leaf}"), suffix))
    .collect()
}

/// `(tensor name, shape, sidecar shard, BF16 bit patterns)`.
type SynthEntry = (String, Vec<usize>, Option<String>, Vec<u16>);

/// Ordered synthetic checkpoint inventory: name -> (shape, sidecar file, bits).
#[derive(Debug, Clone)]
pub struct SynthCloverCheckpoint {
    entries: Vec<SynthEntry>,
    values: BTreeMap<String, Vec<u16>>,
}

impl SynthCloverCheckpoint {
    pub fn new(config: &CloverConfig) -> Self {
        let hidden = config.hidden_size;
        let key_width = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_width = config.linear_num_value_heads * config.linear_value_head_dim;
        let channels = 2 * key_width + value_width;
        let kernel = config.linear_conv_kernel_dim;
        let query_width = config.num_attention_heads * config.head_dim;
        let kv_width = config.num_key_value_heads * config.head_dim;
        let vocab = config.vocab_size;

        let mut checkpoint = Self {
            entries: Vec::new(),
            values: BTreeMap::new(),
        };
        let push = |entries: &mut Vec<SynthEntry>,
                    name: &str,
                    shape: Vec<usize>,
                    sidecar: Option<&str>| {
            let count: usize = shape.iter().product();
            let bits = (0..count).map(|index| synth_value(name, index)).collect();
            entries.push((name.to_owned(), shape, sidecar.map(str::to_owned), bits));
        };

        push(
            &mut checkpoint.entries,
            SYNTH_EMBED,
            vec![vocab, hidden],
            None,
        );
        for (layer, layer_type) in config.layer_types.iter().copied().enumerate() {
            push(
                &mut checkpoint.entries,
                &format!("model.layers.{layer}.input_layernorm.weight"),
                vec![hidden],
                None,
            );
            match layer_type {
                LayerType::LinearAttention => {
                    let names = gdn_tensor_names(layer);
                    let shapes = |suffix: &str| -> (Vec<usize>, Option<&'static str>) {
                        match suffix {
                            "conv1d" => (vec![channels, 1, kernel], None),
                            "in_proj_qkv" => (vec![channels, hidden], None),
                            "in_proj_z" => (vec![value_width, hidden], None),
                            "in_proj_a" | "in_proj_b" => {
                                (vec![config.linear_num_value_heads, hidden], None)
                            }
                            "A_log" | "dt_bias" | "logit_scale" | "history_bias" | "history_c" => (
                                vec![config.linear_num_value_heads],
                                Some(SYNTH_SCALAR_SIDECAR),
                            ),
                            "norm" => (vec![config.linear_value_head_dim], None),
                            "out_proj" => (vec![hidden, value_width], None),
                            "conv_right" => {
                                (vec![channels, 1, kernel - 1], Some(SYNTH_CONV_SIDECAR))
                            }
                            _ => unreachable!("unknown GDN suffix {suffix}"),
                        }
                    };
                    for (name, suffix) in names {
                        let (shape, sidecar) = shapes(suffix);
                        push(&mut checkpoint.entries, name.as_str(), shape, sidecar);
                    }
                }
                LayerType::FullAttention => {
                    let names = attention_tensor_names(layer);
                    let shapes = |suffix: &str| -> Vec<usize> {
                        match suffix {
                            "q_proj" => vec![query_width * 2, hidden],
                            "k_proj" | "v_proj" => vec![kv_width, hidden],
                            "o_proj" => vec![hidden, query_width],
                            "q_norm" | "k_norm" => vec![config.head_dim],
                            _ => unreachable!("unknown attention suffix {suffix}"),
                        }
                    };
                    for (name, suffix) in names {
                        push(&mut checkpoint.entries, name.as_str(), shapes(suffix), None);
                    }
                }
            }
            push(
                &mut checkpoint.entries,
                &format!("model.layers.{layer}.post_attention_layernorm.weight"),
                vec![hidden],
                None,
            );
            let names = mlp_tensor_names(layer);
            let shapes = |suffix: &str| -> Vec<usize> {
                match suffix {
                    "gate" => vec![config.num_experts, hidden],
                    "gate_up" => vec![config.num_experts, 2 * config.moe_intermediate_size, hidden],
                    "down" => vec![config.num_experts, hidden, config.moe_intermediate_size],
                    "shared_gate" | "shared_up" => {
                        vec![config.shared_expert_intermediate_size, hidden]
                    }
                    "shared_down" => {
                        vec![hidden, config.shared_expert_intermediate_size]
                    }
                    "shared_gate_proj" => vec![1, hidden],
                    _ => unreachable!("unknown mlp suffix {suffix}"),
                }
            };
            for (name, suffix) in names {
                push(&mut checkpoint.entries, name.as_str(), shapes(suffix), None);
            }
        }
        push(
            &mut checkpoint.entries,
            SYNTH_FINAL_NORM,
            vec![hidden],
            None,
        );
        push(
            &mut checkpoint.entries,
            SYNTH_LM_HEAD,
            vec![vocab, hidden],
            None,
        );
        checkpoint
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries
            .iter()
            .find(|(entry, _, _, _)| entry == name)
            .map(|(_, shape, _, _)| shape.as_slice())
    }

    pub fn set(&mut self, name: &str, values: Vec<u16>) -> &mut Self {
        let expected: usize = self
            .shape(name)
            .expect("unknown tensor name")
            .iter()
            .product();
        assert_eq!(values.len(), expected, "value count for {name}");
        self.values.insert(name.to_owned(), values);
        self
    }

    pub fn set_f32(&mut self, name: &str, values: &[f32]) -> &mut Self {
        self.set(name, values.iter().copied().map(bf16_bits).collect())
    }

    pub fn values(&self, name: &str) -> Vec<u16> {
        if let Some(values) = self.values.get(name) {
            return values.clone();
        }
        let index = self
            .entries
            .iter()
            .position(|(entry, _, _, _)| entry == name)
            .expect("unknown tensor name");
        let shape = &self.entries[index].1;
        let count: usize = shape.iter().product();
        (0..count)
            .map(|element| synth_value(name, element))
            .collect()
    }

    /// Move one tensor to the end of the model-shard payload region so callers
    /// can force a mid-stream I/O failure by truncating the shard.
    pub fn move_last(&mut self, name: &str) -> &mut Self {
        let index = self
            .entries
            .iter()
            .position(|(entry, _, _, _)| entry == name)
            .expect("unknown tensor name");
        let entry = self.entries.remove(index);
        self.entries.push(entry);
        self
    }

    pub fn payload_bytes(&self) -> u64 {
        self.entries
            .iter()
            .map(|(_, shape, _, _)| shape.iter().product::<usize>() as u64 * 2)
            .sum()
    }

    /// Write the index, the single model shard, and both GDN sidecars.
    pub fn write(&self, root: &Path) {
        let mut shards: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (index, (_, _, sidecar, _)) in self.entries.iter().enumerate() {
            let shard = sidecar
                .clone()
                .unwrap_or_else(|| SYNTH_MODEL_SHARD.to_owned());
            shards.entry(shard).or_default().push(index);
        }
        let mut weight_map = String::new();
        let mut total_size = 0u64;
        for (shard, indices) in &shards {
            let mut offset = 0usize;
            let mut header = String::from("{");
            let mut payload = Vec::new();
            for (position, &entry) in indices.iter().enumerate() {
                let (name, shape, _, _) = &self.entries[entry];
                if position != 0 {
                    header.push(',');
                }
                let bits = self.values(name);
                let length = bits.len() * 2;
                let shape_text = shape
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                header.push_str(&format!(
                    r#""{name}":{{"dtype":"BF16","shape":[{shape_text}],"data_offsets":[{offset},{}]}}"#,
                    offset + length
                ));
                for bit in bits {
                    payload.extend_from_slice(&bit.to_le_bytes());
                }
                offset += length;
                total_size += length as u64;
            }
            header.push('}');
            let _ = offset;
            // Rebuild the weight map entries for this shard.
            for &entry in indices {
                let (name, _, _, _) = &self.entries[entry];
                if !weight_map.is_empty() {
                    weight_map.push(',');
                }
                weight_map.push_str(&format!(r#""{name}":"{shard}""#));
            }
            let mut header = header.into_bytes();
            while header.len() % 8 != 0 {
                header.push(b' ');
            }
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(&header);
            bytes.extend_from_slice(&payload);
            fs::write(root.join(shard), bytes).expect("write shard");
        }
        let index = format!(
            r#"{{"metadata":{{"total_size":{total_size}}},"weight_map":{{{weight_map}}}}}"#
        );
        fs::write(root.join("model.safetensors.index.json"), index).expect("write index");
        assert_eq!(
            total_size,
            self.payload_bytes(),
            "synthetic payload bookkeeping"
        );
    }
}

/// Reduced but internally consistent Clover config used by M11.4 CPU tests.
///
/// Attention geometry stays at the pinned production values (head_dim 256,
/// partial factor 0.25, theta 10 000 000) so the RoPE path is exercised at the
/// shipped configuration while all other dimensions stay tiny.
pub fn tiny_clover_config() -> CloverConfig {
    const CONFIG: &str = include_str!("../../../docs/audits/fixtures/clover-config.json");
    let mut config = CloverConfig::from_json(CONFIG).unwrap();
    config.hidden_size = 4;
    config.num_hidden_layers = 2;
    config.layer_types = vec![LayerType::LinearAttention, LayerType::FullAttention];
    config.num_experts = 4;
    config.num_experts_per_tok = 2;
    config.vocab_size = 9;
    config.mask_token_id = 7;
    config.eos_token_id = 0;
    config.block_size = 4;
    config.max_position_embeddings = 32;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.moe_intermediate_size = 2;
    config.shared_expert_intermediate_size = 2;
    config.linear_num_key_heads = 1;
    config.linear_key_head_dim = 2;
    config.linear_num_value_heads = 2;
    config.linear_value_head_dim = 1;
    config.linear_conv_kernel_dim = 4;
    config
}
