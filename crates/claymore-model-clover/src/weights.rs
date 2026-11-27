//! Checkpoint-backed CPU weight access and bounded dense projections.
//!
//! The store reads BF16 matrix tiles directly from the validated Safetensors
//! inventory. It never loads a complete shard or complete matrix just to run a
//! dense projection.

use claymore_checkpoint::safetensors::MAX_CHUNK_BYTES;
use claymore_tensor::Bf16;
use thiserror::Error;

use crate::{
    CloverCheckpointError, CloverCheckpointInventory, CloverExpertWeights,
    CloverSharedExpertWeights,
};

#[derive(Debug, Error)]
pub enum CloverCpuWeightError {
    #[error(transparent)]
    Checkpoint(#[from] CloverCheckpointError),
    #[error("checkpoint tensor {0:?} is missing")]
    MissingTensor(String),
    #[error("tensor {tensor:?} must be a rank-2 matrix, got shape {shape:?}")]
    InvalidMatrixShape { tensor: String, shape: Vec<usize> },
    #[error("input has {actual} elements, expected {expected}")]
    InputShape { actual: usize, expected: usize },
    #[error("matrix row width {0} exceeds the bounded BF16 read capacity")]
    MatrixRowTooWide(usize),
    #[error("token ID {token_id} is outside embedding vocabulary size {vocabulary}")]
    TokenOutOfRange { token_id: u32, vocabulary: usize },
    #[error("normalization tensor {tensor:?} must have shape [{expected}], got {actual:?}")]
    InvalidVectorShape {
        tensor: String,
        expected: usize,
        actual: Vec<usize>,
    },
    #[error("RMSNorm epsilon must be finite and positive, got {0}")]
    InvalidEpsilon(f32),
    #[error("matrix dimensions overflow addressable memory")]
    SizeOverflow,
    #[error("expert index {expert} is outside the declared expert count {experts}")]
    ExpertOutOfRange { expert: usize, experts: usize },
}

/// Read-only access to validated checkpoint weights for CPU reference kernels.
///
/// This type borrows the inventory, whose shard handles remain the sole
/// authority for locating tensor payloads. Each projection reads at most
/// 8 MiB of BF16 weights at a time.
pub struct CloverCpuWeightStore<'a> {
    inventory: &'a CloverCheckpointInventory,
}

impl<'a> CloverCpuWeightStore<'a> {
    pub fn new(inventory: &'a CloverCheckpointInventory) -> Self {
        Self { inventory }
    }

    pub fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.inventory
            .tensor(name)
            .map(|tensor| tensor.shape.as_slice())
    }

    /// Read a range of BF16 elements while preserving their stored bit patterns.
    pub fn read_bf16(
        &self,
        name: &str,
        element_offset: u64,
        element_count: usize,
    ) -> Result<Vec<Bf16>, CloverCpuWeightError> {
        Ok(self
            .inventory
            .read_bf16_range(name, element_offset, element_count)?
            .into_iter()
            .map(Bf16::from_bits)
            .collect())
    }

    /// Gather BF16 token embeddings from a checkpoint matrix
    /// [vocabulary, hidden_size], preserving the order and duplicates of IDs.
    pub fn embed_tokens(
        &self,
        tensor_name: &str,
        token_ids: &[u32],
    ) -> Result<Vec<Bf16>, CloverCpuWeightError> {
        let shape = self
            .tensor_shape(tensor_name)
            .ok_or_else(|| CloverCpuWeightError::MissingTensor(tensor_name.to_owned()))?;
        if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
            return Err(CloverCpuWeightError::InvalidMatrixShape {
                tensor: tensor_name.to_owned(),
                shape: shape.to_vec(),
            });
        }
        let (vocabulary, hidden_size) = (shape[0], shape[1]);
        let output_len = token_ids
            .len()
            .checked_mul(hidden_size)
            .ok_or(CloverCpuWeightError::SizeOverflow)?;
        let mut output = Vec::with_capacity(output_len);
        for &token_id in token_ids {
            let row = usize::try_from(token_id)
                .ok()
                .filter(|&id| id < vocabulary)
                .ok_or(CloverCpuWeightError::TokenOutOfRange {
                    token_id,
                    vocabulary,
                })?;
            let offset = row
                .checked_mul(hidden_size)
                .and_then(|value| u64::try_from(value).ok())
                .ok_or(CloverCpuWeightError::SizeOverflow)?;
            output.extend(self.read_bf16(tensor_name, offset, hidden_size)?);
        }
        Ok(output)
    }

    /// Apply Clover's zero-centered RMSNorm to [tokens, hidden_size].
    ///
    /// The checkpoint scale is interpreted as (1 + weight), reductions run
    /// in FP32, and output values are rounded to BF16 once.
    pub fn rms_norm_zero_centered(
        &self,
        tensor_name: &str,
        input: &[Bf16],
        token_count: usize,
        hidden_size: usize,
        epsilon: f32,
    ) -> Result<Vec<Bf16>, CloverCpuWeightError> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(CloverCpuWeightError::InvalidEpsilon(epsilon));
        }
        let shape = self
            .tensor_shape(tensor_name)
            .ok_or_else(|| CloverCpuWeightError::MissingTensor(tensor_name.to_owned()))?;
        if shape != [hidden_size] {
            return Err(CloverCpuWeightError::InvalidVectorShape {
                tensor: tensor_name.to_owned(),
                expected: hidden_size,
                actual: shape.to_vec(),
            });
        }
        let expected = token_count
            .checked_mul(hidden_size)
            .ok_or(CloverCpuWeightError::SizeOverflow)?;
        if token_count == 0 || hidden_size == 0 || input.len() != expected {
            return Err(CloverCpuWeightError::InputShape {
                actual: input.len(),
                expected,
            });
        }
        let scale = self.read_bf16(tensor_name, 0, hidden_size)?;
        let mut output = Vec::with_capacity(input.len());
        for row in input.chunks_exact(hidden_size) {
            let mean_square = row
                .iter()
                .map(|value| {
                    let value = value.to_f32();
                    value * value
                })
                .sum::<f32>()
                / hidden_size as f32;
            let inverse_rms = (mean_square + epsilon).sqrt().recip();
            for (value, weight) in row.iter().zip(&scale) {
                output.push(Bf16::from_f32(
                    value.to_f32() * inverse_rms * (1.0 + weight.to_f32()),
                ));
            }
        }
        Ok(output)
    }

    /// Execute the bias-free BF16 router projection and widen its BF16-rounded
    /// results to FP32, matching Clover's router input contract.
    pub fn router_logits(
        &self,
        tensor_name: &str,
        input: &[Bf16],
        token_count: usize,
    ) -> Result<Vec<f32>, CloverCpuWeightError> {
        Ok(self
            .linear(tensor_name, input, token_count)?
            .into_iter()
            .map(Bf16::to_f32)
            .collect())
    }

    /// Execute a bias-free BF16 linear projection with checkpoint matrix
    /// [out_features, in_features] and input [tokens, in_features].
    ///
    /// The accumulation order is deterministic and FP32; each output element
    /// is rounded to BF16 exactly once. Matrix rows are streamed in bounded
    /// tiles, so even the vocabulary projection does not require a full weight
    /// matrix allocation.
    pub fn linear(
        &self,
        tensor_name: &str,
        input: &[Bf16],
        token_count: usize,
    ) -> Result<Vec<Bf16>, CloverCpuWeightError> {
        let shape = self
            .tensor_shape(tensor_name)
            .ok_or_else(|| CloverCpuWeightError::MissingTensor(tensor_name.to_owned()))?;
        if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
            return Err(CloverCpuWeightError::InvalidMatrixShape {
                tensor: tensor_name.to_owned(),
                shape: shape.to_vec(),
            });
        }
        let (output_width, input_width) = (shape[0], shape[1]);
        let expected_input = token_count
            .checked_mul(input_width)
            .ok_or(CloverCpuWeightError::SizeOverflow)?;
        if token_count == 0 || input.len() != expected_input {
            return Err(CloverCpuWeightError::InputShape {
                actual: input.len(),
                expected: expected_input,
            });
        }

        let output_len = token_count
            .checked_mul(output_width)
            .ok_or(CloverCpuWeightError::SizeOverflow)?;
        let mut output = vec![Bf16::from_bits(0); output_len];
        let max_elements = MAX_CHUNK_BYTES / std::mem::size_of::<u16>();
        let rows_per_chunk = max_elements / input_width;
        if rows_per_chunk == 0 {
            return Err(CloverCpuWeightError::MatrixRowTooWide(input_width));
        }

        let mut row_start = 0usize;
        while row_start < output_width {
            let rows = rows_per_chunk.min(output_width - row_start);
            let elements = rows
                .checked_mul(input_width)
                .ok_or(CloverCpuWeightError::SizeOverflow)?;
            let offset = row_start
                .checked_mul(input_width)
                .and_then(|value| u64::try_from(value).ok())
                .ok_or(CloverCpuWeightError::SizeOverflow)?;
            let weights = self.read_bf16(tensor_name, offset, elements)?;

            for token in 0..token_count {
                let input_row = &input[token * input_width..(token + 1) * input_width];
                for local_row in 0..rows {
                    let weight_row =
                        &weights[local_row * input_width..(local_row + 1) * input_width];
                    let mut accumulator = 0.0f32;
                    for inner in 0..input_width {
                        accumulator = input_row[inner]
                            .to_f32()
                            .mul_add(weight_row[inner].to_f32(), accumulator);
                    }
                    output[token * output_width + row_start + local_row] =
                        Bf16::from_f32(accumulator);
                }
            }
            row_start += rows;
        }

        Ok(output)
    }

    /// Materialize one routed expert from Clover's stacked checkpoint tensors.
    /// Each tensor range is validated and read through the inventory's bounded
    /// BF16 reader; no other expert's weights are loaded.
    pub fn routed_expert_weights(
        &self,
        layer: usize,
        expert: usize,
        num_experts: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<CloverExpertWeights, CloverCpuWeightError> {
        if num_experts == 0 || expert >= num_experts {
            return Err(CloverCpuWeightError::ExpertOutOfRange {
                expert,
                experts: num_experts,
            });
        }
        if hidden_size == 0 || intermediate_size == 0 {
            return Err(CloverCpuWeightError::SizeOverflow);
        }
        let gate_name = format!("model.layers.{layer}.mlp.experts.gate_up_proj");
        let down_name = format!("model.layers.{layer}.mlp.experts.down_proj");
        let gate_rows = intermediate_size
            .checked_mul(2)
            .ok_or(CloverCpuWeightError::SizeOverflow)?;
        let gate_elements = gate_rows
            .checked_mul(hidden_size)
            .ok_or(CloverCpuWeightError::SizeOverflow)?;
        let down_elements = hidden_size
            .checked_mul(intermediate_size)
            .ok_or(CloverCpuWeightError::SizeOverflow)?;
        for (name, expected) in [
            (&gate_name, vec![num_experts, gate_rows, hidden_size]),
            (&down_name, vec![num_experts, hidden_size, intermediate_size]),
        ] {
            let actual = self
                .tensor_shape(name)
                .ok_or_else(|| CloverCpuWeightError::MissingTensor(name.clone()))?;
            if actual != expected {
                return Err(CloverCpuWeightError::InvalidMatrixShape {
                    tensor: name.clone(),
                    shape: actual.to_vec(),
                });
            }
        }
        let gate_offset = expert
            .checked_mul(gate_elements)
            .and_then(|value| u64::try_from(value).ok())
            .ok_or(CloverCpuWeightError::SizeOverflow)?;
        let down_offset = expert
            .checked_mul(down_elements)
            .and_then(|value| u64::try_from(value).ok())
            .ok_or(CloverCpuWeightError::SizeOverflow)?;
        Ok(CloverExpertWeights {
            hidden_size,
            intermediate_size,
            gate_up_proj: self.read_bf16(&gate_name, gate_offset, gate_elements)?,
            down_proj: self.read_bf16(&down_name, down_offset, down_elements)?,
        })
    }

    /// Materialize the per-layer shared SwiGLU expert and its learned gate.
    pub fn shared_expert_weights(
        &self,
        layer: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<CloverSharedExpertWeights, CloverCpuWeightError> {
        if hidden_size == 0 || intermediate_size == 0 {
            return Err(CloverCpuWeightError::SizeOverflow);
        }
        let prefix = format!("model.layers.{layer}.mlp");
        let gate_name = format!("{prefix}.shared_expert.gate_proj.weight");
        let up_name = format!("{prefix}.shared_expert.up_proj.weight");
        let down_name = format!("{prefix}.shared_expert.down_proj.weight");
        let shared_gate_name = format!("{prefix}.shared_expert_gate.weight");
        for (name, expected) in [
            (&gate_name, vec![intermediate_size, hidden_size]),
            (&up_name, vec![intermediate_size, hidden_size]),
            (&down_name, vec![hidden_size, intermediate_size]),
            (&shared_gate_name, vec![1, hidden_size]),
        ] {
            let actual = self
                .tensor_shape(name)
                .ok_or_else(|| CloverCpuWeightError::MissingTensor(name.clone()))?;
            if actual != expected {
                return Err(CloverCpuWeightError::InvalidMatrixShape {
                    tensor: name.clone(),
                    shape: actual.to_vec(),
                });
            }
        }
        let projection_elements = intermediate_size
            .checked_mul(hidden_size)
            .ok_or(CloverCpuWeightError::SizeOverflow)?;
        let down_elements = hidden_size
            .checked_mul(intermediate_size)
            .ok_or(CloverCpuWeightError::SizeOverflow)?;
        Ok(CloverSharedExpertWeights {
            hidden_size,
            intermediate_size,
            gate_proj: self.read_bf16(&gate_name, 0, projection_elements)?,
            up_proj: self.read_bf16(&up_name, 0, projection_elements)?,
            down_proj: self.read_bf16(&down_name, 0, down_elements)?,
            shared_gate: self.read_bf16(&shared_gate_name, 0, hidden_size)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

    fn write_tensor(path: &Path, name: &str, shape: &[usize], payload: &[u8]) {
        let shape = shape
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"BF16","shape":[{shape}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut header = header.into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(payload);
        fs::write(path, bytes).unwrap();
    }

    fn write_fixture(root: &Path) {
        let model = root.join("model-00001-of-00001.safetensors");
        let conv = root.join("bd_gdn_conv_right_weight.safetensors");
        let scalars = root.join("bd_gdn_softmax_scalars.safetensors");
        write_tensor(
            &model,
            "weight",
            &[2, 2],
            &[0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40],
        );
        write_tensor(&conv, "conv", &[2], &[0, 0, 0, 0]);
        write_tensor(&scalars, "scale", &[2], &[0, 0, 0, 0]);
        fs::write(
            root.join("model.safetensors.index.json"),
            br#"{"metadata":{"total_size":16},"weight_map":{"weight":"model-00001-of-00001.safetensors","conv":"bd_gdn_conv_right_weight.safetensors","scale":"bd_gdn_softmax_scalars.safetensors"}}"#,
        )
        .unwrap();
    }

    #[test]
    fn checkpoint_linear_streams_bf16_rows_and_matches_reference() {
        let root: PathBuf = std::env::temp_dir().join(format!(
            "claymore-clover-cpu-weights-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        write_fixture(&root);

        let inventory = CloverCheckpointInventory::inspect(&root).unwrap();
        let store = CloverCpuWeightStore::new(&inventory);
        assert_eq!(store.tensor_shape("weight"), Some([2, 2].as_slice()));
        let embedding = store.embed_tokens("weight", &[1]).unwrap();
        assert_eq!(embedding, [Bf16::from_f32(3.0), Bf16::from_f32(4.0)]);
        assert!(matches!(
            store.embed_tokens("weight", &[2]),
            Err(CloverCpuWeightError::TokenOutOfRange { .. })
        ));
        let input = [Bf16::from_f32(5.0), Bf16::from_f32(6.0)];
        let output = store.linear("weight", &input, 1).unwrap();
        assert_eq!(output, [Bf16::from_f32(17.0), Bf16::from_f32(39.0)]);
        assert_eq!(
            store.router_logits("weight", &input, 1).unwrap(),
            [Bf16::from_f32(17.0).to_f32(), Bf16::from_f32(39.0).to_f32()]
        );
        let normalized = store
            .rms_norm_zero_centered(
                "scale",
                &[Bf16::from_f32(3.0), Bf16::from_f32(4.0)],
                1,
                2,
                1e-6,
            )
            .unwrap();
        let denominator = (12.5f32 + 1e-6).sqrt();
        assert_eq!(
            normalized,
            [
                Bf16::from_f32(3.0 / denominator),
                Bf16::from_f32(4.0 / denominator)
            ]
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn checkpoint_linear_rejects_bad_shapes_and_missing_tensors() {
        let root = std::env::temp_dir().join(format!(
            "claymore-clover-cpu-weights-errors-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        write_fixture(&root);

        let inventory = CloverCheckpointInventory::inspect(&root).unwrap();
        let store = CloverCpuWeightStore::new(&inventory);
        assert!(matches!(
            store.linear("absent", &[Bf16::from_bits(0)], 1),
            Err(CloverCpuWeightError::MissingTensor(_))
        ));
        assert!(matches!(
            store.linear("conv", &[Bf16::from_bits(0)], 1),
            Err(CloverCpuWeightError::InvalidMatrixShape { .. })
        ));
        assert!(matches!(
            store.linear("weight", &[Bf16::from_bits(0)], 1),
            Err(CloverCpuWeightError::InputShape { .. })
        ));

        fs::remove_dir_all(root).unwrap();
    }

    fn write_multi_tensor(path: &Path, tensors: &[(&str, &[usize], &[u16])]) {
        let mut header = String::from("{");
        let mut offset = 0usize;
        let mut payload = Vec::new();
        for (index, (name, shape, values)) in tensors.iter().enumerate() {
            if index != 0 { header.push(','); }
            let shape = shape.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
            let start = offset;
            offset += values.len() * 2;
            header.push_str(&format!(
                r#""{name}":{{"dtype":"BF16","shape":[{shape}],"data_offsets":[{start},{offset}]}}"#
            ));
            for value in *values { payload.extend_from_slice(&value.to_le_bytes()); }
        }
        header.push('}');
        let mut header = header.into_bytes();
        while header.len() % 8 != 0 { header.push(b' '); }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&payload);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn gathers_one_routed_expert_and_shared_expert_from_stacked_safetensors() {
        let root = std::env::temp_dir().join(format!(
            "claymore-clover-moe-weights-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let bits = |value: f32| (value.to_bits() >> 16) as u16;
        let gate_up = (1..=8).map(|v| bits(v as f32)).collect::<Vec<_>>();
        let down = (9..=12).map(|v| bits(v as f32)).collect::<Vec<_>>();
        let model = root.join("model-00001-of-00001.safetensors");
        write_multi_tensor(
            &model,
            &[
                (
                    "model.layers.0.mlp.experts.gate_up_proj",
                    &[2, 2, 2],
                    &gate_up,
                ),
                ("model.layers.0.mlp.experts.down_proj", &[2, 2, 1], &down),
                (
                    "model.layers.0.mlp.shared_expert.gate_proj.weight",
                    &[1, 2],
                    &[bits(13.0), bits(14.0)],
                ),
                (
                    "model.layers.0.mlp.shared_expert.up_proj.weight",
                    &[1, 2],
                    &[bits(15.0), bits(16.0)],
                ),
                (
                    "model.layers.0.mlp.shared_expert.down_proj.weight",
                    &[2, 1],
                    &[bits(17.0), bits(18.0)],
                ),
                (
                    "model.layers.0.mlp.shared_expert_gate.weight",
                    &[1, 2],
                    &[bits(19.0), bits(20.0)],
                ),
            ],
        );
        write_tensor(
            &root.join("bd_gdn_conv_right_weight.safetensors"),
            "conv",
            &[2],
            &[0, 0, 0, 0],
        );
        write_tensor(
            &root.join("bd_gdn_softmax_scalars.safetensors"),
            "scale",
            &[2],
            &[0, 0, 0, 0],
        );
        fs::write(root.join("model.safetensors.index.json"), br#"{"metadata":{"total_size":48},"weight_map":{
            "model.layers.0.mlp.experts.gate_up_proj":"model-00001-of-00001.safetensors",
            "model.layers.0.mlp.experts.down_proj":"model-00001-of-00001.safetensors",
            "model.layers.0.mlp.shared_expert.gate_proj.weight":"model-00001-of-00001.safetensors",
            "model.layers.0.mlp.shared_expert.up_proj.weight":"model-00001-of-00001.safetensors",
            "model.layers.0.mlp.shared_expert.down_proj.weight":"model-00001-of-00001.safetensors",
            "model.layers.0.mlp.shared_expert_gate.weight":"model-00001-of-00001.safetensors",
            "conv":"bd_gdn_conv_right_weight.safetensors","scale":"bd_gdn_softmax_scalars.safetensors"
        }}"#).unwrap();

        let inventory = CloverCheckpointInventory::inspect(&root).unwrap();
        let store = CloverCpuWeightStore::new(&inventory);
        let expert = store.routed_expert_weights(0, 1, 2, 2, 1).unwrap();
        assert_eq!(
            expert.gate_up_proj.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            [5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(
            expert.down_proj.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            [11.0, 12.0]
        );
        let shared = store.shared_expert_weights(0, 2, 1).unwrap();
        assert_eq!(
            shared.gate_proj.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            [13.0, 14.0]
        );
        assert_eq!(
            shared.up_proj.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            [15.0, 16.0]
        );
        assert_eq!(
            shared.down_proj.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            [17.0, 18.0]
        );
        assert_eq!(
            shared.shared_gate.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            [19.0, 20.0]
        );
        assert!(matches!(
            store.routed_expert_weights(0, 2, 2, 2, 1),
            Err(CloverCpuWeightError::ExpertOutOfRange { .. })
        ));
        fs::remove_dir_all(root).unwrap();
    }

}
