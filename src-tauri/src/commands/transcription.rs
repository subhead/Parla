// Tauri commands for local manual transcription.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, State};
use tracing::info;

use crate::commands::parakeet::ParakeetModelManagerState;
use crate::transcription::{
    audio::read_audio_as_f32,
    model_manager::ModelManager,
    parakeet::ParakeetEngine,
    whisper::{WhisperEngine, WhisperParams},
};

pub struct WhisperEngineState(pub Arc<WhisperEngine>);

impl Default for WhisperEngineState {
    fn default() -> Self {
        Self(Arc::new(WhisperEngine::new()))
    }
}

#[derive(Debug, Deserialize)]
pub struct TranscribeRequest {
    pub wav_path: String,
    #[serde(flatten)]
    pub source: ManualTranscriptionSource,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "source", rename_all = "lowercase")]
pub enum ManualTranscriptionSource {
    Whisper {
        model_id: String,
        #[serde(default)]
        language: Option<String>,
        #[serde(default)]
        initial_prompt: Option<String>,
        #[serde(default)]
        n_threads: Option<usize>,
    },
    Parakeet {
        model_id: String,
        #[serde(default)]
        language: Option<String>,
    },
}

#[derive(Debug, Serialize)]
pub struct TranscribeResponse {
    pub text: String,
    pub model_id: String,
    pub duration_ms: u64,
}

trait WhisperTranscriber {
    fn whisper(
        &self,
        model_path: &std::path::Path,
        samples: &[f32],
        params: &WhisperParams,
    ) -> anyhow::Result<String>;
}

trait ParakeetTranscriber {
    fn parakeet(
        &self,
        model_path: &std::path::Path,
        samples: &[f32],
        language: Option<&str>,
    ) -> anyhow::Result<String>;
}

struct WhisperEngineTranscriber {
    engine: Arc<WhisperEngine>,
}

impl WhisperTranscriber for WhisperEngineTranscriber {
    fn whisper(
        &self,
        model_path: &std::path::Path,
        samples: &[f32],
        params: &WhisperParams,
    ) -> anyhow::Result<String> {
        self.engine.load(model_path)?;
        self.engine.transcribe_samples(samples, params)
    }
}

struct ParakeetEngineTranscriber {
    engine: Arc<ParakeetEngine>,
}

impl ParakeetTranscriber for ParakeetEngineTranscriber {
    fn parakeet(
        &self,
        model_path: &std::path::Path,
        samples: &[f32],
        language: Option<&str>,
    ) -> anyhow::Result<String> {
        self.engine.ensure_loaded(model_path)?;
        self.engine.transcribe_samples(samples, language)
    }
}

enum TranscriptionRoute<'a> {
    Whisper {
        transcriber: &'a dyn WhisperTranscriber,
        model: Option<PathBuf>,
    },
    Parakeet {
        transcriber: &'a dyn ParakeetTranscriber,
        model: Option<PathBuf>,
    },
}

fn dispatch_transcription(
    req: &TranscribeRequest,
    route: TranscriptionRoute<'_>,
) -> Result<String, String> {
    // Check selected model before touching WAV. This avoids decoding work when model is unavailable.
    let model = match (&req.source, &route) {
        (
            ManualTranscriptionSource::Whisper { model_id, .. },
            TranscriptionRoute::Whisper { model, .. },
        ) => model
            .clone()
            .ok_or_else(|| format!("modele Whisper non disponible: {model_id}"))?,
        (
            ManualTranscriptionSource::Parakeet { model_id, .. },
            TranscriptionRoute::Parakeet { model, .. },
        ) => model
            .clone()
            .ok_or_else(|| format!("modele Parakeet incomplet ou indisponible: {model_id}"))?,
        (ManualTranscriptionSource::Parakeet { .. }, TranscriptionRoute::Whisper { .. }) => {
            return Err("route Parakeet appelee avec un transcripteur Whisper".into())
        }
        (ManualTranscriptionSource::Whisper { .. }, TranscriptionRoute::Parakeet { .. }) => {
            return Err("route Whisper appelee avec un transcripteur Parakeet".into())
        }
    };

    let audio_path = PathBuf::from(&req.wav_path);
    if !audio_path.exists() {
        return Err(format!("audio file not found: {}", req.wav_path));
    }
    if !audio_path.is_file() {
        return Err(format!(
            "audio path is not a regular file: {}",
            req.wav_path
        ));
    }
    let samples = read_audio_as_f32(&audio_path)
        .map_err(|e| format!("could not read audio {}: {e}", audio_path.display()))?;

    match (&req.source, route) {
        (
            ManualTranscriptionSource::Whisper {
                language,
                initial_prompt,
                n_threads,
                ..
            },
            TranscriptionRoute::Whisper { transcriber, .. },
        ) => transcriber
            .whisper(
                &model,
                &samples,
                &WhisperParams {
                    language: language.clone(),
                    initial_prompt: initial_prompt.clone(),
                    n_threads: n_threads.unwrap_or(0),
                },
            )
            .map_err(|e| e.to_string()),
        (
            ManualTranscriptionSource::Parakeet { language, .. },
            TranscriptionRoute::Parakeet { transcriber, .. },
        ) => crate::transcription::parakeet::transcribe_parakeet_chunks(&samples, None, |chunk| {
            transcriber.parakeet(&model, chunk, language.as_deref())
        })
        .map_err(|e| e.to_string()),
        _ => unreachable!("route validated against transcription source"),
    }
}

#[tauri::command]
pub async fn transcribe_wav(
    app: AppHandle,
    engine_state: State<'_, WhisperEngineState>,
    models_state: State<'_, super::models::ModelManagerState>,
    req: TranscribeRequest,
) -> Result<TranscribeResponse, String> {
    let req = Arc::new(req);
    let model_id = match &req.source {
        ManualTranscriptionSource::Whisper { model_id, .. }
        | ManualTranscriptionSource::Parakeet { model_id, .. } => model_id.clone(),
    };
    let start = std::time::Instant::now();
    let text = match &req.source {
        ManualTranscriptionSource::Whisper { model_id, .. } => {
            let models: Arc<ModelManager> = models_state.0.clone();
            let whisper_model = models.path_if_present(model_id);
            let transcriber = WhisperEngineTranscriber {
                engine: engine_state.0.clone(),
            };
            let req = req.clone();
            tokio::task::spawn_blocking(move || {
                dispatch_transcription(
                    &req,
                    TranscriptionRoute::Whisper {
                        transcriber: &transcriber,
                        model: whisper_model,
                    },
                )
            })
            .await
            .map_err(|e| format!("tache transcription panic: {e}"))??
        }
        ManualTranscriptionSource::Parakeet { model_id, .. } => {
            let parakeet_models = app.state::<ParakeetModelManagerState>().0.clone();
            let parakeet_model = parakeet_models.path_for_id(model_id);
            let transcriber = ParakeetEngineTranscriber {
                engine: app
                    .state::<crate::transcription::parakeet::ParakeetEngineState>()
                    .0
                    .clone(),
            };
            let req = req.clone();
            tokio::task::spawn_blocking(move || {
                dispatch_transcription(
                    &req,
                    TranscriptionRoute::Parakeet {
                        transcriber: &transcriber,
                        model: parakeet_model,
                    },
                )
            })
            .await
            .map_err(|e| format!("tache transcription panic: {e}"))??
        }
    };

    let duration_ms = start.elapsed().as_millis() as u64;
    info!(
        model_id,
        chars = text.len(),
        duration_ms,
        "Transcription terminee"
    );

    Ok(TranscribeResponse {
        text,
        model_id,
        duration_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::ops::Deref;
    use std::sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    };

    struct Fake {
        whisper: Mutex<Option<WhisperParams>>,
        parakeet_calls: Mutex<Vec<usize>>,
        parakeet_first_samples: Mutex<Vec<f32>>,
        parakeet_results: Mutex<VecDeque<Result<String, String>>>,
    }
    impl WhisperTranscriber for Fake {
        fn whisper(
            &self,
            _: &std::path::Path,
            samples: &[f32],
            params: &WhisperParams,
        ) -> anyhow::Result<String> {
            *self.whisper.lock().unwrap() = Some(params.clone());
            assert!(!samples.is_empty());
            Ok("whisper text".into())
        }
    }

    impl ParakeetTranscriber for Fake {
        fn parakeet(
            &self,
            _: &std::path::Path,
            samples: &[f32],
            _: Option<&str>,
        ) -> anyhow::Result<String> {
            self.parakeet_calls.lock().unwrap().push(samples.len());
            self.parakeet_first_samples.lock().unwrap().push(samples[0]);
            match self.parakeet_results.lock().unwrap().pop_front() {
                Some(Ok(text)) => Ok(text),
                Some(Err(error)) => anyhow::bail!(error),
                None => Ok("parakeet text".into()),
            }
        }
    }

    struct TempWav(PathBuf);

    impl Deref for TempWav {
        type Target = std::path::Path;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl AsRef<std::path::Path> for TempWav {
        fn as_ref(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempWav {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn wav(seconds: usize) -> TempWav {
        wav_with_markers(seconds, &[])
    }

    fn wav_with_markers(seconds: usize, markers: &[(usize, i16)]) -> TempWav {
        static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);

        let path = loop {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let counter = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
            let candidate = std::env::temp_dir().join(format!(
                "parla-manual-{}-{timestamp}-{counter}.wav",
                std::process::id()
            ));
            if std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
                .is_ok()
            {
                break candidate;
            }
        };
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let temp = TempWav(path);
        let mut writer = hound::WavWriter::create(&temp, spec).unwrap();
        for sample in 0..seconds * 16_000 {
            let second = sample / 16_000;
            let value = markers
                .iter()
                .rev()
                .find(|&&(marker, _)| marker <= second)
                .map_or(0, |&(_, value)| value);
            writer.write_sample(value).unwrap();
        }
        writer.finalize().unwrap();
        temp
    }

    #[test]
    fn dispatcher_forwards_whisper_only_parameters() {
        let path = wav(1);
        let fake = Fake {
            whisper: Mutex::new(None),
            parakeet_calls: Mutex::new(Vec::new()),
            parakeet_first_samples: Mutex::new(Vec::new()),
            parakeet_results: Mutex::new(VecDeque::new()),
        };
        let req = TranscribeRequest {
            wav_path: path.to_string_lossy().into(),
            source: ManualTranscriptionSource::Whisper {
                model_id: "w".into(),
                language: Some("de".into()),
                initial_prompt: Some("prompt".into()),
                n_threads: Some(7),
            },
        };
        assert_eq!(
            dispatch_transcription(
                &req,
                TranscriptionRoute::Whisper {
                    transcriber: &fake,
                    model: Some(PathBuf::from("w")),
                },
            )
            .unwrap(),
            "whisper text"
        );
        let params = fake.whisper.lock().unwrap().clone().unwrap();
        assert_eq!(params.language.as_deref(), Some("de"));
        assert_eq!(params.initial_prompt.as_deref(), Some("prompt"));
        assert_eq!(params.n_threads, 7);
        assert!(fake.parakeet_calls.lock().unwrap().is_empty());
    }

    #[test]
    fn dispatcher_reads_wav_for_parakeet_route() {
        let path = wav(1);
        let fake = Fake {
            whisper: Mutex::new(None),
            parakeet_calls: Mutex::new(Vec::new()),
            parakeet_first_samples: Mutex::new(Vec::new()),
            parakeet_results: Mutex::new(VecDeque::from([Ok("parakeet text".into())])),
        };
        let req = TranscribeRequest {
            wav_path: path.to_string_lossy().into(),
            source: ManualTranscriptionSource::Parakeet {
                model_id: "p".into(),
                language: None,
            },
        };
        assert_eq!(
            dispatch_transcription(
                &req,
                TranscriptionRoute::Parakeet {
                    transcriber: &fake,
                    model: Some(PathBuf::from("p")),
                },
            )
            .unwrap(),
            "parakeet text"
        );
        assert!(fake.whisper.lock().unwrap().is_none());
        assert_eq!(*fake.parakeet_calls.lock().unwrap(), vec![16_000]);
    }

    #[test]
    fn dispatcher_chunks_long_wav_in_order_and_merges_boundary_text() {
        let path = wav_with_markers(121, &[(0, 1000), (59, 2000), (118, 3000)]);
        let fake = Fake {
            whisper: Mutex::new(None),
            parakeet_calls: Mutex::new(Vec::new()),
            parakeet_first_samples: Mutex::new(Vec::new()),
            parakeet_results: Mutex::new(VecDeque::from([
                Ok("alpha beta".into()),
                Ok("BETA gamma".into()),
                Ok("delta".into()),
            ])),
        };
        let req = TranscribeRequest {
            wav_path: path.to_string_lossy().into(),
            source: ManualTranscriptionSource::Parakeet {
                model_id: "p".into(),
                language: None,
            },
        };
        let text = dispatch_transcription(
            &req,
            TranscriptionRoute::Parakeet {
                transcriber: &fake,
                model: Some(PathBuf::from("p")),
            },
        )
        .unwrap();
        let calls = fake.parakeet_calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0], 60 * 16_000);
        assert_eq!(calls[1], 60 * 16_000);
        assert_eq!(calls[2], 2 * 16_000);
        assert!(calls.iter().all(|&samples| samples <= 60 * 16_000));
        let first_samples = fake.parakeet_first_samples.lock().unwrap().clone();
        assert_eq!(first_samples.len(), 3);
        assert!((first_samples[0] - 1000.0 / 32_768.0).abs() < 0.001);
        assert!((first_samples[1] - 2000.0 / 32_768.0).abs() < 0.001);
        assert!((first_samples[2] - 3000.0 / 32_768.0).abs() < 0.001);
        assert_eq!(text, "alpha beta gamma delta");
    }

    #[test]
    fn dispatcher_exactly_sixty_seconds_calls_once() {
        let path = wav(60);
        let fake = Fake {
            whisper: Mutex::new(None),
            parakeet_calls: Mutex::new(Vec::new()),
            parakeet_first_samples: Mutex::new(Vec::new()),
            parakeet_results: Mutex::new(VecDeque::from([Ok("exact".into())])),
        };
        let req = TranscribeRequest {
            wav_path: path.to_string_lossy().into(),
            source: ManualTranscriptionSource::Parakeet {
                model_id: "p".into(),
                language: None,
            },
        };
        assert_eq!(
            dispatch_transcription(
                &req,
                TranscriptionRoute::Parakeet {
                    transcriber: &fake,
                    model: Some(PathBuf::from("p")),
                }
            )
            .unwrap(),
            "exact"
        );
        assert_eq!(*fake.parakeet_calls.lock().unwrap(), vec![60 * 16_000]);
    }

    #[test]
    fn dispatcher_second_chunk_failure_returns_no_partial_output() {
        let path = wav(121);
        let fake = Fake {
            whisper: Mutex::new(None),
            parakeet_calls: Mutex::new(Vec::new()),
            parakeet_first_samples: Mutex::new(Vec::new()),
            parakeet_results: Mutex::new(VecDeque::from([
                Ok("partial".into()),
                Err("second chunk failed".into()),
            ])),
        };
        let req = TranscribeRequest {
            wav_path: path.to_string_lossy().into(),
            source: ManualTranscriptionSource::Parakeet {
                model_id: "p".into(),
                language: None,
            },
        };
        let error = dispatch_transcription(
            &req,
            TranscriptionRoute::Parakeet {
                transcriber: &fake,
                model: Some(PathBuf::from("p")),
            },
        )
        .unwrap_err();
        assert!(error.contains("second chunk failed"));
        assert_eq!(
            *fake.parakeet_calls.lock().unwrap(),
            vec![60 * 16_000, 60 * 16_000]
        );
    }

    #[test]
    fn dispatcher_rejects_missing_wav_and_selected_model() {
        let fake = Fake {
            whisper: Mutex::new(None),
            parakeet_calls: Mutex::new(Vec::new()),
            parakeet_first_samples: Mutex::new(Vec::new()),
            parakeet_results: Mutex::new(VecDeque::new()),
        };
        let req = TranscribeRequest {
            wav_path: "missing.wav".into(),
            source: ManualTranscriptionSource::Whisper {
                model_id: "w".into(),
                language: None,
                initial_prompt: None,
                n_threads: None,
            },
        };
        assert!(dispatch_transcription(
            &req,
            TranscriptionRoute::Whisper {
                transcriber: &fake,
                model: Some(PathBuf::from("w")),
            },
        )
        .unwrap_err()
        .contains("not found"));
        let path = wav(1);
        let req = TranscribeRequest {
            wav_path: path.to_string_lossy().into(),
            source: ManualTranscriptionSource::Whisper {
                model_id: "w".into(),
                language: None,
                initial_prompt: None,
                n_threads: None,
            },
        };
        assert!(dispatch_transcription(
            &req,
            TranscriptionRoute::Whisper {
                transcriber: &fake,
                model: None,
            },
        )
        .unwrap_err()
        .contains("modele Whisper non disponible"));
        let req = TranscribeRequest {
            wav_path: path.to_string_lossy().into(),
            source: ManualTranscriptionSource::Parakeet {
                model_id: "p".into(),
                language: None,
            },
        };
        assert!(dispatch_transcription(
            &req,
            TranscriptionRoute::Parakeet {
                transcriber: &fake,
                model: None,
            },
        )
        .unwrap_err()
        .contains("incomplet"));
    }
}
