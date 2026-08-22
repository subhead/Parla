// Wrapper autour de `parakeet_rs::ParakeetTDT` pour reproduire l'API
// publique de WhisperEngine (load paresseux + transcribe_samples).
//
// Reference VoiceInk : FluidAudio expose `AsrManager.transcribe(samples,
// decoderState)`. Ici on s'appuie sur parakeet-rs qui encapsule deja le
// chargement ONNX, le mel spectrogram, l'encodeur, le joint net et le
// decoder TDT. On expose juste un handle avec rechargement paresseux si
// le repertoire change.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use parakeet_rs::{ParakeetTDT, Transcriber};
use parking_lot::Mutex;

pub const PARAKEET_MAX_CHUNK_SAMPLES: usize = 60 * 16_000;
pub const PARAKEET_OVERLAP_SAMPLES: usize = 16_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParakeetChunk {
    pub start: usize,
    pub end: usize,
}

/// Plan source-preserving Parakeet requests. VAD ranges are timing hints only.
pub fn plan_parakeet_chunks(
    total_samples: usize,
    vad_ranges: Option<&[(usize, usize)]>,
) -> Vec<ParakeetChunk> {
    if total_samples == 0 {
        return Vec::new();
    }
    if total_samples <= PARAKEET_MAX_CHUNK_SAMPLES {
        return vec![ParakeetChunk {
            start: 0,
            end: total_samples,
        }];
    }
    let mut chunks = Vec::new();
    let mut ranges: Vec<(usize, usize)> = vad_ranges
        .unwrap_or_default()
        .iter()
        .filter_map(|&(start, end)| {
            let start = start.min(total_samples);
            let end = end.min(total_samples);
            (start < end).then_some((start, end))
        })
        .collect();
    ranges.sort_unstable();
    let mut grouped_ranges = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        if let Some((_, grouped_end)) = grouped_ranges.last_mut() {
            if start <= *grouped_end {
                *grouped_end = (*grouped_end).max(end);
                continue;
            }
        }
        grouped_ranges.push((start, end));
    }
    let mut start = 0;
    while start < total_samples {
        let hard_end = (start + PARAKEET_MAX_CHUNK_SAMPLES).min(total_samples);
        // Only use the end of a grouped speech region as a VAD boundary. A
        // boundary is useful only when it is close to the hard limit; otherwise
        // fixed-size chunking with overlap gives better source coverage.
        let near_limit = hard_end.saturating_sub(PARAKEET_OVERLAP_SAMPLES * 5);
        let end = grouped_ranges
            .iter()
            .map(|&(_, speech_end)| speech_end)
            .filter(|&candidate| {
                candidate >= near_limit && candidate < hard_end && candidate > start
            })
            .min_by_key(|candidate| hard_end.abs_diff(*candidate))
            .unwrap_or(hard_end);
        chunks.push(ParakeetChunk { start, end });
        if end >= total_samples {
            break;
        }
        let next_start = if end == hard_end {
            end.saturating_sub(PARAKEET_OVERLAP_SAMPLES)
        } else {
            end
        };
        // Defensive guard for malformed hints or future boundary changes.
        start = next_start.max(start + 1).min(total_samples);
    }
    chunks
}

fn normalized_word(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn clean_words(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out = String::new();
    for word in words {
        if !out.is_empty()
            && !word
                .chars()
                .next()
                .is_some_and(|c| ",.!?;:%)]}".contains(c))
        {
            out.push(' ');
        }
        out.push_str(word);
    }
    out
}

pub fn merge_parakeet_transcripts(transcripts: &[String]) -> String {
    let mut merged = String::new();
    for current in transcripts {
        if current.trim().is_empty() {
            continue;
        }
        if merged.trim().is_empty() {
            merged = clean_words(current);
            continue;
        }
        let prior_words: Vec<&str> = merged.split_whitespace().collect();
        let current_words: Vec<&str> = current.split_whitespace().collect();
        let max = prior_words.len().min(current_words.len());
        let exact_overlap = (1..=max)
            .rev()
            .find(|&count| prior_words[prior_words.len() - count..] == current_words[..count]);
        let overlap = exact_overlap
            .or_else(|| {
                (1..=max).rev().find(|&count| {
                    prior_words[prior_words.len() - count..]
                        .iter()
                        .zip(&current_words[..count])
                        .all(|(a, b)| {
                            normalized_word(a) == normalized_word(b)
                                && !normalized_word(a).is_empty()
                        })
                })
            })
            .unwrap_or(0);
        let suffix = current_words[overlap..].join(" ");
        if !suffix.is_empty() {
            merged = clean_words(&format!("{merged} {suffix}"));
        }
    }
    merged
}

pub fn transcribe_parakeet_chunks<F>(
    samples: &[f32],
    vad_ranges: Option<&[(usize, usize)]>,
    mut transcribe: F,
) -> Result<String>
where
    F: FnMut(&[f32]) -> Result<String>,
{
    let chunks = plan_parakeet_chunks(samples.len(), vad_ranges);
    let mut texts = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        texts.push(transcribe(&samples[chunk.start..chunk.end])?);
    }
    Ok(merge_parakeet_transcripts(&texts))
}

struct Loaded {
    path: PathBuf,
    engine: ParakeetTDT,
}

pub struct ParakeetEngine {
    current: Mutex<Option<Loaded>>,
}

impl ParakeetEngine {
    pub fn new() -> Self {
        Self {
            current: Mutex::new(None),
        }
    }

    /// Charge le modele (repertoire contenant config.json + *.onnx + vocab.txt)
    /// si ce n'est pas deja celui-ci qui est charge. Bloquant.
    pub fn ensure_loaded(&self, model_dir: &Path) -> Result<()> {
        let mut guard = self.current.lock();
        if let Some(cur) = guard.as_ref() {
            if cur.path == model_dir {
                return Ok(());
            }
        }
        // Libere l'ancien avant de charger le nouveau.
        *guard = None;

        // Selection de l'execution provider ONNX. parakeet-rs prend un
        // `Option<ExecutionConfig>` ; `None` retombe sur `ExecutionProvider::Cpu`
        // (le `#[default]` de l'enum). Il faut donc lui passer explicitement le
        // provider GPU, sinon compiler avec `cuda-onnx` / `directml-onnx` ne
        // fait qu'exposer le variant sans jamais l'utiliser (inference CPU).
        // Chaque provider GPU retombe automatiquement sur CPU s'il echoue a
        // s'initialiser (parakeet-rs enchaine [GPU, CPU.error_on_failure()]).
        #[cfg(feature = "cuda-onnx")]
        let cfg = Some(
            parakeet_rs::ExecutionConfig::new()
                .with_execution_provider(parakeet_rs::ExecutionProvider::Cuda),
        );
        #[cfg(all(not(feature = "cuda-onnx"), feature = "directml-onnx"))]
        let cfg = Some(
            parakeet_rs::ExecutionConfig::new()
                .with_execution_provider(parakeet_rs::ExecutionProvider::DirectML),
        );
        #[cfg(all(not(feature = "cuda-onnx"), not(feature = "directml-onnx")))]
        let cfg: Option<parakeet_rs::ExecutionConfig> = None;

        let engine = ParakeetTDT::from_pretrained(model_dir, cfg)
            .map_err(|e| anyhow!("parakeet load: {e:?}"))?;
        *guard = Some(Loaded {
            path: model_dir.to_path_buf(),
            engine,
        });
        Ok(())
    }

    /// Transcrit un buffer PCM Float32 mono 16 kHz. Bloquant (inference).
    /// `language` est accepte mais non-utilise : parakeet TDT v2 est anglais
    /// uniquement, v3 detecte la langue automatiquement.
    pub fn transcribe_samples(&self, samples: &[f32], _language: Option<&str>) -> Result<String> {
        let mut guard = self.current.lock();
        let loaded = guard
            .as_mut()
            .ok_or_else(|| anyhow!("modele Parakeet non charge"))?;
        // Pas de timestamps pour l'usage dictee : on veut juste le texte.
        // parakeet-rs attend un Vec<f32>. Une copie est inevitable ici.
        let result = loaded
            .engine
            .transcribe_samples(samples.to_vec(), 16000, 1, None)
            .map_err(|e| anyhow!("parakeet transcribe: {e:?}"))?;
        Ok(result.text)
    }

    /// Libere la memoire (ONNX session). Utile quand l'utilisateur change de
    /// modele ou desactive la source.
    pub fn unload(&self) {
        *self.current.lock() = None;
    }
}

impl Default for ParakeetEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_plan_is_bounded_and_overlapped() {
        let chunks = plan_parakeet_chunks(121 * 16_000, None);
        assert_eq!(
            chunks[0],
            ParakeetChunk {
                start: 0,
                end: 60 * 16_000
            }
        );
        assert_eq!(chunks[1].start, 59 * 16_000);
        assert!(chunks
            .iter()
            .all(|c| c.end - c.start <= PARAKEET_MAX_CHUNK_SAMPLES));
    }

    #[test]
    fn short_input_is_one_full_chunk_even_with_vad_ranges() {
        let total = 40 * 16_000;
        assert_eq!(
            plan_parakeet_chunks(total, Some(&[(1_000, 2_000), (20_000, 30_000)])),
            vec![ParakeetChunk {
                start: 0,
                end: total
            }]
        );
    }

    #[test]
    fn vad_plan_uses_boundary_and_preserves_timing() {
        let boundary = 58 * 16_000;
        let chunks = plan_parakeet_chunks(
            90 * 16_000,
            Some(&[
                (10 * 16_000, 20 * 16_000),
                (30 * 16_000, 40 * 16_000),
                (boundary, 59 * 16_000),
            ]),
        );
        assert_eq!(chunks[0].end, boundary);
        assert_eq!(chunks[1].start, boundary);
        assert!(chunks.iter().all(|c| c.end > c.start));
        assert!(chunks
            .iter()
            .all(|c| c.end - c.start <= PARAKEET_MAX_CHUNK_SAMPLES));
    }

    #[test]
    fn vad_ranges_are_grouped_before_boundary_selection() {
        let chunks = plan_parakeet_chunks(
            90 * 16_000,
            Some(&[
                (10 * 16_000, 20 * 16_000),
                (20 * 16_000, 30 * 16_000),
                (29 * 16_000, 59 * 16_000),
            ]),
        );
        assert_eq!(chunks[0].end, 59 * 16_000);
    }

    #[test]
    fn unsuitable_vad_boundary_uses_fixed_overlap() {
        let chunks = plan_parakeet_chunks(90 * 16_000, Some(&[(0, 31 * 16_000)]));
        assert_eq!(chunks[0].end, 60 * 16_000);
        assert_eq!(chunks[1].start, 59 * 16_000);
    }

    #[test]
    fn malformed_vad_ranges_cannot_make_empty_chunks_or_loop() {
        let chunks = plan_parakeet_chunks(
            121 * 16_000,
            Some(&[
                (usize::MAX, usize::MAX),
                (10, 10),
                (90, 20),
                (0, usize::MAX),
            ]),
        );
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| c.start < c.end));
        assert!(chunks.iter().all(|c| c.end <= 121 * 16_000));
        assert_eq!(chunks.last().unwrap().end, 121 * 16_000);
    }

    #[test]
    fn fixed_fallback_overlaps_by_one_second_and_covers_source() {
        let total = 121 * 16_000 + 123;
        let chunks = plan_parakeet_chunks(total, Some(&[(1, 2)]));
        assert!(chunks
            .iter()
            .all(|c| c.end - c.start <= PARAKEET_MAX_CHUNK_SAMPLES));
        assert_eq!(chunks[1].start, chunks[0].end - PARAKEET_OVERLAP_SAMPLES);
        assert_eq!(chunks.last().unwrap().end, total);
    }

    #[test]
    fn merge_only_removes_contiguous_boundary_overlap() {
        let texts = vec![
            "Hello world, repeated word".into(),
            "WORLD repeated word again".into(),
        ];
        assert_eq!(
            merge_parakeet_transcripts(&texts),
            "Hello world, repeated word again"
        );
        assert_eq!(
            merge_parakeet_transcripts(&["one one two".into(), "three one".into()]),
            "one one two three one"
        );
        assert_eq!(
            merge_parakeet_transcripts(&["Hello,   WORLD".into(), " world!   Again".into()]),
            "Hello, WORLD Again"
        );
        assert_eq!(
            merge_parakeet_transcripts(&["Hello, brave WORLD".into(), " world! Again".into()]),
            "Hello, brave WORLD Again"
        );
        assert_eq!(
            merge_parakeet_transcripts(&["Hello".into(), "hello".into()]),
            "Hello"
        );
        assert_eq!(
            merge_parakeet_transcripts(&["alpha beta".into(), "gamma alpha".into()]),
            "alpha beta gamma alpha"
        );
        assert_eq!(
            merge_parakeet_transcripts(&["one".into(), String::new(), "two".into()]),
            "one two"
        );
    }

    #[test]
    fn executor_is_serial_and_propagates_failure() {
        let samples = vec![0.0; PARAKEET_MAX_CHUNK_SAMPLES + 1];
        let mut calls = 0;
        let error = transcribe_parakeet_chunks(&samples, None, |_| {
            calls += 1;
            anyhow::bail!("chunk failed")
        });
        assert!(error.is_err());
        assert_eq!(calls, 1);
    }
}

#[derive(Default)]
pub struct ParakeetEngineState(pub Arc<ParakeetEngine>);
