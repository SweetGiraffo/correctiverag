"""
Audio Loader using Faster-Whisper ASR.
Transcribes audio files into timestamped text chunks with start/end metadata.
"""

from typing import List, Optional
import os
from .base import BaseLoader, IngestedChunk, Modality


class WhisperModelCache:
    """Singleton cache for faster-whisper model to prevent repeated initialization."""
    _model = None
    _model_size = None

    @classmethod
    def get_model(cls, model_size: str = "base"):
        if cls._model is None or cls._model_size != model_size:
            from faster_whisper import WhisperModel
            cls._model = WhisperModel(model_size, device="cpu", compute_type="int8")
            cls._model_size = model_size
        return cls._model


class AudioLoader(BaseLoader):
    """
    Transcribes audio files (.mp3, .wav, .m4a) using Faster-Whisper ASR
    and chunks speech while preserving timestamps in metadata.
    """

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size

    def load(self, file_path: str) -> List[IngestedChunk]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        filename = os.path.basename(file_path)
        chunks: List[IngestedChunk] = []

        try:
            model = WhisperModelCache.get_model(self.model_size)
            segments, info = model.transcribe(file_path, beam_size=5)

            current_text = []
            start_ts = 0.0
            end_ts = 0.0
            chunk_idx = 0

            for seg in segments:
                if not current_text:
                    start_ts = seg.start
                current_text.append(seg.text.strip())
                end_ts = seg.end

                # Flush chunk when reaching ~60 words or sentence threshold
                accumulated = " ".join(current_text)
                if len(accumulated.split()) >= 60:
                    chunks.append(
                        IngestedChunk(
                            modality=Modality.AUDIO,
                            text=accumulated,
                            metadata={
                                "source_file": filename,
                                "file_path": os.path.abspath(file_path),
                                "start_time": round(start_ts, 2),
                                "end_time": round(end_ts, 2),
                                "duration_seconds": round(end_ts - start_ts, 2),
                                "language": info.language,
                                "chunk_index": chunk_idx
                            }
                        )
                    )
                    current_text = []
                    chunk_idx += 1

            if current_text:
                accumulated = " ".join(current_text)
                chunks.append(
                    IngestedChunk(
                        modality=Modality.AUDIO,
                        text=accumulated,
                        metadata={
                            "source_file": filename,
                            "file_path": os.path.abspath(file_path),
                            "start_time": round(start_ts, 2),
                            "end_time": round(end_ts, 2),
                            "duration_seconds": round(end_ts - start_ts, 2),
                            "language": info.language,
                            "chunk_index": chunk_idx
                        }
                    )
                )

            # If no spoken segments detected (e.g. ambient audio, tones, or silence)
            if not chunks:
                chunks.append(
                    IngestedChunk(
                        modality=Modality.AUDIO,
                        text=f"[Audio file: {filename}]",
                        metadata={
                            "source_file": filename,
                            "file_path": os.path.abspath(file_path),
                            "duration_seconds": round(getattr(info, "duration", 0.0), 2),
                            "status": "audio_recorded"
                        }
                    )
                )

        except Exception as e:
            # Fallback if whisper fails or codec is missing
            chunks.append(
                IngestedChunk(
                    modality=Modality.AUDIO,
                    text=f"[Audio transcription failed for {filename}: {str(e)}]",
                    metadata={
                        "source_file": filename,
                        "file_path": os.path.abspath(file_path),
                        "error": str(e)
                    }
                )
            )

        return chunks
