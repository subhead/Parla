// Module transcription - pipeline Whisper local + model manager.
//
// Reference VoiceInk :
// - Transcription/Core/VoiceInkEngine.swift (orchestration)
// - Transcription/Core/TranscriptionPipeline.swift (etapes apres capture)
// - Transcription/Core/Whisper/LibWhisper.swift (params whisper.cpp)
// - Transcription/Core/Whisper/WhisperModelManager.swift (download + storage)
// - Models/PredefinedModels.swift (catalogue de modeles)

pub mod audio;
pub mod cloud;
pub mod engine;
pub mod model;
pub mod model_manager;
pub mod parakeet;
pub mod parakeet_model_manager;
pub mod pipeline;
pub mod vad;
pub mod whisper;

pub use model_manager::{ModelManager, ModelState};
