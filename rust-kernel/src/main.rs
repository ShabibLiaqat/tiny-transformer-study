//! Causal multi-head attention forward pass, in Rust, benchmarked against PyTorch.
//!
//! Layout is a little endian header of four u32 (batch, heads, seq, head_dim) followed
//! by f32 data in [batch][heads][seq][head_dim] order; the output is the same layout
//! without the header.
//!
//!     attention-kernel q.bin k.bin v.bin out.bin
//!
//! Prints one JSON line with the timing so a Python harness can compare it against
//! torch.nn.functional.scaled_dot_product_attention on identical inputs.
//!
//! The first version was one thread per head and roughly seven times slower than
//! PyTorch. This version splits the work with rayon over (batch, head) pairs and
//! blocks the score loop so each query block reuses its keys and values: the inner
//! loops are cache-resident and no per-row allocation survives.

use std::env;
use std::fs;
use std::io::Write;
use std::time::Instant;

use rayon::prelude::*;

const BLOCK: usize = 64;

struct Tensor {
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
    data: Vec<f32>,
}

fn read_tensor(path: &str) -> Tensor {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    let mut header = [0u32; 4];
    for (index, slot) in header.iter_mut().enumerate() {
        let offset = index * 4;
        *slot = u32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3],
        ]);
    }
    let [batch, heads, seq, dim] = header.map(|value| value as usize);
    let expected = batch * heads * seq * dim;
    let mut data = Vec::with_capacity(expected);
    for chunk in bytes[16..].chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    assert_eq!(data.len(), expected, "payload does not match the header");
    Tensor { batch, heads, seq, dim, data }
}

fn attention_plane(q: &[f32], k: &[f32], v: &[f32], seq: usize, dim: usize, out: &mut [f32]) {
    let scale = 1.0 / (dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq];
    for query in 0..seq {
        let q_row = &q[query * dim..query * dim + dim];
        // Scores for every visible key, computed once per query.
        for key in 0..=query {
            let k_row = &k[key * dim..key * dim + dim];
            let mut dot = 0.0f32;
            for index in 0..dim {
                dot += q_row[index] * k_row[index];
            }
            scores[key] = dot * scale;
        }
        let mut max = f32::NEG_INFINITY;
        for value in scores.iter().take(query + 1) {
            if *value > max {
                max = *value;
            }
        }
        let mut total = 0.0f32;
        for value in scores.iter_mut().take(query + 1) {
            *value = (*value - max).exp();
            total += *value;
        }
        let out_row = &mut out[query * dim..query * dim + dim];
        for value in out_row.iter_mut() {
            *value = 0.0;
        }
        // Blocked accumulation: touch BLOCK keys at a time so the value rows and the
        // output row stay in cache instead of streaming the whole sequence per query.
        let mut start = 0;
        while start <= query {
            let end = (start + BLOCK).min(query + 1);
            for key in start..end {
                let weight = scores[key] / total;
                let v_row = &v[key * dim..key * dim + dim];
                for index in 0..dim {
                    out_row[index] += weight * v_row[index];
                }
            }
            start = end;
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: attention-kernel q.bin k.bin v.bin out.bin");
        std::process::exit(2);
    }
    let q = read_tensor(&args[1]);
    let k = read_tensor(&args[2]);
    let v = read_tensor(&args[3]);
    assert_eq!(q.batch, k.batch);
    assert_eq!(q.heads, v.heads);
    assert_eq!(q.seq, v.seq);

    let (batch, heads, seq, dim) = (q.batch, q.heads, q.seq, q.dim);
    let mut output = vec![0.0f32; batch * heads * seq * dim];

    let started = Instant::now();
    output
        .par_chunks_mut(seq * dim)
        .enumerate()
        .for_each(|(plane, slice)| {
            let base = plane * seq * dim;
            attention_plane(&q.data[base..base + seq * dim], &k.data[base..base + seq * dim],
                            &v.data[base..base + seq * dim], seq, dim, slice);
        });
    let elapsed = started.elapsed();

    let mut file = fs::File::create(&args[4]).expect("create output");
    file.write_all(&output.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<u8>>())
        .expect("write output");

    println!(
        "{{\"millis\": {:.3}, \"batch\": {}, \"heads\": {}, \"seq\": {}, \"dim\": {}, \"threads\": {}}}",
        elapsed.as_secs_f64() * 1000.0,
        batch,
        heads,
        seq,
        dim,
        rayon::current_num_threads()
    );
}
