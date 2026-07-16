"""Audio matching adapters."""

from part_io.adapters.audio.clips import extract_audio_clip, play_audio_segment
from part_io.adapters.audio.evaluation import (
    AudioManifestEvaluation,
    evaluate_match_manifest,
    load_match_labels,
)
from part_io.adapters.audio.matcher import AudioMatch, find_audio_sample_matches

__all__ = [
    "AudioManifestEvaluation",
    "AudioMatch",
    "evaluate_match_manifest",
    "extract_audio_clip",
    "find_audio_sample_matches",
    "load_match_labels",
    "play_audio_segment",
]
