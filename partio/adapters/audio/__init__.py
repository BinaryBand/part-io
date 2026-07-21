"""Audio matching adapters."""

from partio.adapters.audio.clips import (
    audio_duration_seconds,
    extract_audio_clip,
    play_audio_segment,
)
from partio.adapters.audio.evaluation import (
    AudioManifestEvaluation,
    evaluate_match_manifest,
    load_match_labels,
)
from partio.adapters.audio.matcher import AudioMatch, find_audio_sample_matches

__all__ = [
    "AudioManifestEvaluation",
    "AudioMatch",
    "audio_duration_seconds",
    "evaluate_match_manifest",
    "extract_audio_clip",
    "find_audio_sample_matches",
    "load_match_labels",
    "play_audio_segment",
]
