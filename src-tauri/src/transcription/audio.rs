//! Decode manual-transcription audio into the model's mono 16 kHz input.

use std::path::Path;

use crate::audio::resampler::MonoResampler;
use anyhow::{anyhow, Context, Result};
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};

pub const MODEL_SAMPLE_RATE: u32 = 16_000;
const MAX_INPUT_BYTES: u64 = 256 * 1024 * 1024;
const MAX_DECODED_DURATION_SECONDS: u64 = 2 * 60 * 60;
const MAX_DECODED_SAMPLES: usize = 32_000_000;
const MAX_OUTPUT_SAMPLES: usize = 32_000_000;
const MAX_COMPRESSED_PACKET_BYTES: usize = 16 * 1024 * 1024;
const MAX_CHANNELS: usize = 32;

/// Decode, downmix, and high-quality resample an audio file for local transcription.
pub fn read_audio_as_f32(path: &Path) -> Result<Vec<f32>> {
    let input_bytes = std::fs::metadata(path)
        .with_context(|| format!("could not inspect audio file {}", path.display()))?
        .len();
    if input_bytes > MAX_INPUT_BYTES {
        return Err(anyhow!(
            "audio file is too large (maximum input size is {} MiB)",
            MAX_INPUT_BYTES / (1024 * 1024)
        ));
    }
    let file = std::fs::File::open(path)
        .with_context(|| format!("could not open audio file {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }
    let probed = get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|error| anyhow!("could not identify audio format: {error}"))?;
    let mut format = probed.format;
    let mut decoder_errors = Vec::new();
    let selected = format.tracks().iter().find_map(|track| {
        let channels = track.codec_params.channels?.count();
        let sample_rate = track.codec_params.sample_rate?;
        if track.codec_params.codec == CODEC_TYPE_NULL
            || channels == 0
            || channels > MAX_CHANNELS
            || sample_rate == 0
            || track.codec_params.max_frames_per_packet.is_none()
        {
            return None;
        }
        match get_codecs().make(&track.codec_params, &DecoderOptions::default()) {
            Ok(decoder) => Some((
                track.id,
                sample_rate,
                channels,
                track.codec_params.max_frames_per_packet,
                decoder,
            )),
            Err(error) => {
                decoder_errors.push(format!("track {}: {error}", track.id));
                None
            }
        }
    });
    let (track_id, sample_rate, channels, max_frames_per_packet, mut decoder) = selected
        .ok_or_else(|| match decoder_errors.last() {
            Some(error) => {
                anyhow!("could not create audio decoder for any supported audio track ({error})")
            }
            None => anyhow!("audio file contains no supported audio track"),
        })?;
    if sample_rate == 0 {
        return Err(anyhow!("audio file has an invalid sample rate: 0"));
    }
    let mut mono = Vec::new();
    let max_duration_samples = (sample_rate as u64)
        .checked_mul(MAX_DECODED_DURATION_SECONDS)
        .and_then(|samples| usize::try_from(samples).ok())
        .unwrap_or(usize::MAX)
        .min(MAX_DECODED_SAMPLES)
        .min(
            (MAX_OUTPUT_SAMPLES as u64)
                .saturating_mul(sample_rate as u64)
                .checked_div(MODEL_SAMPLE_RATE as u64)
                .and_then(|samples| usize::try_from(samples).ok())
                .unwrap_or(usize::MAX),
        );

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(symphonia::core::errors::Error::ResetRequired) => {
                return Err(anyhow!("audio decoder requires unsupported reset"))
            }
            Err(error) => return Err(anyhow!("could not read audio packet: {error}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let packet_frame_cap = ensure_packet_within_decode_cap(
            packet.buf().len(),
            max_frames_per_packet,
            mono.len(),
            max_duration_samples,
        )?;
        let decoded = decoder
            .decode(&packet)
            .map_err(|error| anyhow!("could not decode audio packet: {error}"))?;
        append_downmixed(
            &decoded,
            channels,
            &mut mono,
            max_duration_samples,
            packet_frame_cap,
        )?;
    }

    if mono.is_empty() {
        return Err(anyhow!("audio file contains no samples"));
    }
    let mut resampler = MonoResampler::new(sample_rate)
        .map_err(|error| anyhow!("could not create audio resampler: {error}"))?;
    let mut resampled = Vec::new();
    resampler
        .process(&mono, &mut resampled)
        .and_then(|_| resampler.flush(&mut resampled))
        .map_err(|error| anyhow!("could not resample audio: {error}"))?;
    Ok(resampled)
}

fn append_downmixed(
    buffer: &AudioBufferRef<'_>,
    channels: usize,
    output: &mut Vec<f32>,
    sample_cap: usize,
    packet_frame_cap: u64,
) -> Result<()> {
    let remaining_frames = sample_cap.saturating_sub(output.len());
    if u64::try_from(remaining_frames).unwrap_or(u64::MAX) < packet_frame_cap {
        return Err(anyhow!(
            "decoded audio exceeds supported duration or sample limit"
        ));
    }
    let remaining_interleaved_samples = checked_interleaved_capacity(remaining_frames, channels)?;
    // Check decoder-reported capacity before SampleBuffer::new allocates. Symphonia does not
    // expose a caller-provided allocation limit for Decoder::decode, so this is the last guard
    // available before converting decoder output into our own buffer.
    if u64::try_from(buffer.capacity()).unwrap_or(u64::MAX) > packet_frame_cap {
        return Err(anyhow!(
            "decoded audio exceeds supported duration or sample limit"
        ));
    }
    let interleaved_capacity = checked_interleaved_capacity(buffer.capacity(), channels)?;
    if interleaved_capacity > remaining_interleaved_samples {
        return Err(anyhow!(
            "decoded audio exceeds supported duration or sample limit"
        ));
    }
    let mut samples = SampleBuffer::<f32>::new(buffer.capacity() as u64, *buffer.spec());
    samples.copy_interleaved_ref(buffer.clone());
    let frames = samples.samples().len() / channels;
    if frames > remaining_frames {
        return Err(anyhow!(
            "decoded audio exceeds supported duration or sample limit"
        ));
    }
    for frame in samples.samples().chunks_exact(channels) {
        output.push(frame.iter().sum::<f32>() / channels as f32);
    }
    Ok(())
}

fn checked_interleaved_capacity(frames: usize, channels: usize) -> Result<usize> {
    frames.checked_mul(channels).ok_or_else(|| {
        anyhow!("decoded audio allocation exceeds supported duration or sample limit")
    })
}

fn ensure_packet_within_decode_cap(
    packet_bytes: usize,
    max_frames_per_packet: Option<u64>,
    decoded_samples: usize,
    sample_cap: usize,
) -> Result<u64> {
    if packet_bytes > MAX_COMPRESSED_PACKET_BYTES {
        return Err(anyhow!(
            "compressed audio packet exceeds supported size limit"
        ));
    }
    let remaining_samples =
        u64::try_from(sample_cap.saturating_sub(decoded_samples)).unwrap_or(u64::MAX);
    let packet_frames = max_frames_per_packet.ok_or_else(|| {
        anyhow!("audio track does not declare a reliable maximum packet frame count")
    })?;
    if packet_frames > remaining_samples {
        return Err(anyhow!(
            "audio packet exceeds remaining supported duration or sample limit"
        ));
    }
    Ok(packet_frames)
}

#[cfg(test)]
mod tests {
    use super::{
        checked_interleaved_capacity, ensure_packet_within_decode_cap, MonoResampler, MAX_CHANNELS,
        MAX_COMPRESSED_PACKET_BYTES, MAX_DECODED_SAMPLES,
    };

    #[test]
    fn resampler_rejects_zero_sample_rate() {
        assert!(MonoResampler::new(0).is_err());
    }

    #[test]
    fn decoded_sample_cap_is_bounded() {
        assert_eq!(MAX_DECODED_SAMPLES, 32_000_000);
    }

    #[test]
    fn packet_is_rejected_before_decode_when_duration_exceeds_remaining_cap() {
        assert!(ensure_packet_within_decode_cap(100, Some(101), 900, 1_000).is_err());
        assert_eq!(
            ensure_packet_within_decode_cap(100, Some(100), 900, 1_000).unwrap(),
            100
        );
    }

    #[test]
    fn packet_without_declared_frame_limit_is_rejected() {
        assert!(ensure_packet_within_decode_cap(100, None, 0, 1_000_000).is_err());
    }

    #[test]
    fn packet_frame_cap_uses_remaining_samples() {
        assert!(ensure_packet_within_decode_cap(100, Some(1_000), 900, 1_000).is_err());
    }

    #[test]
    fn oversized_compressed_packet_is_rejected_before_decode() {
        assert!(ensure_packet_within_decode_cap(
            MAX_COMPRESSED_PACKET_BYTES + 1,
            Some(1),
            0,
            1_000,
        )
        .is_err());
        assert!(
            ensure_packet_within_decode_cap(MAX_COMPRESSED_PACKET_BYTES, Some(1), 0, 1_000,)
                .is_ok()
        );
    }

    #[test]
    fn channel_count_is_bounded() {
        assert_eq!(MAX_CHANNELS, 32);
    }

    #[test]
    fn interleaved_capacity_checks_frame_channel_multiplication() {
        assert_eq!(checked_interleaved_capacity(10, 2).unwrap(), 20);
        assert!(checked_interleaved_capacity(usize::MAX, 2).is_err());
    }
}
