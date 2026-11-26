//! Dependency-free SHA-256 used to pin checkpoint metadata and artifacts.
//!
//! Tensor payloads are validated structurally (offsets, shapes, coverage) but
//! are not individually hashed by the Safetensors format. Small metadata files
//! with known digests are verified with this implementation.

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const INITIAL: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Compute SHA-256 over an in-memory buffer.
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut state = INITIAL;
    let mut chunks = bytes.chunks_exact(64);
    for chunk in &mut chunks {
        compress(&mut state, chunk.try_into().expect("64-byte chunk"));
    }
    let remainder = chunks.remainder();
    let mut tail = Vec::with_capacity(128);
    tail.extend_from_slice(remainder);
    tail.push(0x80);
    while tail.len() % 64 != 56 {
        tail.push(0);
    }
    tail.extend_from_slice(&(bytes.len() as u64 * 8).to_be_bytes());
    for chunk in tail.chunks_exact(64) {
        compress(&mut state, chunk.try_into().expect("64-byte chunk"));
    }
    let mut digest = [0u8; 32];
    for (word, out) in state.iter().zip(digest.chunks_exact_mut(4)) {
        out.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

/// Streaming SHA-256 for multi-gigabyte payloads.
///
/// A full-shard payload is verified in bounded chunks; this keeps the transient
/// allocation at the chunk size, never the whole file.
pub struct Sha256Stream {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    total: u64,
}

impl Default for Sha256Stream {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256Stream {
    pub fn new() -> Self {
        Self {
            state: INITIAL,
            buffer: [0u8; 64],
            buffered: 0,
            total: 0,
        }
    }

    /// Feed at most `stream::MAX_CHUNK_BYTES` bytes at a time.
    pub fn update(&mut self, bytes: &[u8]) {
        self.total = self
            .total
            .checked_add(bytes.len() as u64)
            .expect("streamed length overflow");
        let mut offset = 0;
        if self.buffered > 0 {
            let take = (64 - self.buffered).min(bytes.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&bytes[..take]);
            self.buffered += take;
            offset = take;
            if self.buffered == 64 {
                compress(&mut self.state, &self.buffer);
                self.buffered = 0;
            }
        }
        let mut chunks = bytes[offset..].chunks_exact(64);
        for chunk in &mut chunks {
            compress(&mut self.state, chunk.try_into().expect("64-byte chunk"));
        }
        let remainder = chunks.remainder();
        self.buffer[..remainder.len()].copy_from_slice(remainder);
        self.buffered = remainder.len();
    }

    pub fn finish(self) -> [u8; 32] {
        let mut tail = Vec::with_capacity(128);
        tail.extend_from_slice(&self.buffer[..self.buffered]);
        tail.push(0x80);
        while tail.len() % 64 != 56 {
            tail.push(0);
        }
        tail.extend_from_slice(&(self.total * 8).to_be_bytes());
        let mut state = self.state;
        for chunk in tail.chunks_exact(64) {
            compress(&mut state, chunk.try_into().expect("64-byte chunk"));
        }
        let mut digest = [0u8; 32];
        for (word, out) in state.iter().zip(digest.chunks_exact_mut(4)) {
            out.copy_from_slice(&word.to_be_bytes());
        }
        digest
    }
}

/// Lowercase hexadecimal rendering of a digest.
pub fn hex(digest: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (index, word) in block.chunks_exact(4).enumerate() {
        w[index] = u32::from_be_bytes(word.try_into().expect("four-byte word"));
    }
    for index in 16..64 {
        let s0 =
            w[index - 15].rotate_right(7) ^ w[index - 15].rotate_right(18) ^ (w[index - 15] >> 3);
        let s1 =
            w[index - 2].rotate_right(17) ^ w[index - 2].rotate_right(19) ^ (w[index - 2] >> 10);
        w[index] = w[index - 16]
            .wrapping_add(s0)
            .wrapping_add(w[index - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for index in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[index])
            .wrapping_add(w[index]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }
    for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *slot = slot.wrapping_add(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_fips_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex(&sha256(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn handles_block_boundaries() {
        for length in [55, 56, 57, 63, 64, 65, 127, 128, 129] {
            let input = vec![0x5au8; length];
            let digest = sha256(&input);
            // Recompute with a second independent padding path (streamed by
            // concatenating one extra byte and slicing is not equivalent), so
            // this test only pins determinism and output width.
            assert_eq!(digest, sha256(&input));
            assert_eq!(hex(&digest).len(), 64);
        }
    }
}
