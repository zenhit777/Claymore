//! Checkpoint-backed CPU weight access and bounded dense projections.
//!
//! The store reads BF16 matrix tiles directly from the validated Safetensors
//! inventory. It never loads a complete shard or complete matrix just to run a
//! dense projection.

use claymore_checkpoint::safetensors::MAX_CHUNK_BYTES;
use claymore_tensor::Bf16;
use thiserror::Error;

use crate::{CloverCheckpointError, CloverCheckpointInventory};

#[derive(Debug, Error)]
pub enum CloverCpuWeightError {
    #[error(transparent)]
    Checkpoint(#[from] CloverCheckpointError),
    #[error("checkpoint tensor {0:?} is missing")]
    MissingTensor(String),
    #[error("tensor {tensor:?} must be a rank-2 matrix, got shape {shape:?}")]
    InvalidMatrixShape {
        tensor: String,
        shape: Vec<usize>,
    },
    #[error("input has {actual} elements, expected {expected}")]
    InputShape { actual: usize, expected: usize },
    #[error("matrix row width {0} exceeds the bounded BF16 read capacity")]
    MatrixRowTooWide(usize),
    #[error("matrix dimensions overflow addressable memory")]
    SizeOverflow,
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
        self.inventory.tensor(name).map(|tensor| tensor.shape.as_slice())
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
                    let weight_row = &weights
                        [local_row * input_width..(local_row + 1) * input_width];
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
            .map(usize::to_string)
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
        let input = [Bf16::from_f32(5.0), Bf16::from_f32(6.0)];
        let output = store.linear("weight", &input, 1).unwrap();
        assert_eq!(
            output,
            [Bf16::from_f32(17.0), Bf16::from_f32(39.0)]
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
}
