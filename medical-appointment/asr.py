"""Speech recognition with word timestamps.

Qwen3-ASR produces the text and Qwen3-ForcedAligner times every word, giving a list of timed words. Audio is cut
at the quietest point before each chunk limit and the chunks are recognised and aligned as one batch, so decoding
time follows the longest chunk rather than the whole conversation.
"""

import io
import logging
import time
from typing import List, Tuple

import numpy as np

from settings import Settings
from transcript import Word

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def load_audio(audio_bytes: bytes) -> np.ndarray:
    from faster_whisper.audio import decode_audio

    return decode_audio(io.BytesIO(audio_bytes), sampling_rate=SAMPLE_RATE)


def chunk_bounds(audio: np.ndarray, max_s: float, search_s: float = 8.0) -> List[Tuple[int, int]]:
    limit = int(max_s * SAMPLE_RATE)
    frame = int(0.02 * SAMPLE_RATE)
    bounds, start = [], 0
    while len(audio) - start > limit:
        hi = start + limit
        lo = max(start + frame, hi - int(search_s * SAMPLE_RATE))
        frames = (hi - lo) // frame
        energy = (audio[lo:lo + frames * frame].reshape(frames, frame) ** 2).mean(axis=1)
        cut = lo + int(np.argmin(energy)) * frame + frame // 2
        bounds.append((start, cut))
        start = cut
    bounds.append((start, len(audio)))
    return bounds


def parse_device(device: str) -> Tuple[str, int]:
    kind, _, index = device.partition(':')
    return kind, int(index) if index else 0


def torch_dtype(settings: Settings):
    import torch

    if not settings.device.startswith('cuda'):
        torch.set_num_threads(settings.cpu_threads)
    if settings.torch_dtype != 'auto':
        return getattr(torch, settings.torch_dtype)
    if not settings.device.startswith('cuda'):
        return torch.float32
    if torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    return torch.float16


class QwenASR:
    def __init__(self, settings: Settings):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(settings.qwen_asr_model)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            settings.qwen_asr_model, dtype=torch_dtype(settings)
        ).to(settings.device).eval()
        self.prompt = settings.asr_prompt or None
        self.max_new_tokens = settings.asr_max_new_tokens

    def transcribe_texts(self, pieces: List[np.ndarray]) -> List[str]:
        inputs = self.processor.apply_transcription_request(
            audio=list(pieces), language='English', prompt=[self.prompt] * len(pieces) if self.prompt else None
        )
        inputs = inputs.to(self.model.device, self.model.dtype)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output[:, inputs['input_ids'].shape[1]:]
        return [self.processor.decode(row, return_format='transcription_only').strip() for row in generated]


class QwenAligner:
    def __init__(self, settings: Settings):
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(settings.aligner_model)
        self.model = AutoModelForTokenClassification.from_pretrained(
            settings.aligner_model, dtype=torch_dtype(settings)
        ).to(settings.device).eval()

    def align_batch(self, pieces: List[np.ndarray], texts: List[str], offsets: List[float]) -> List[List[Word]]:
        tokens = [text.split() for text in texts]
        keep = [i for i, t in enumerate(tokens) if t]
        results: List[List[Word]] = [[] for _ in texts]
        if not keep:
            return results

        inputs, word_lists = self.processor.prepare_forced_aligner_inputs(
            audio=[pieces[i] for i in keep], transcript=[' '.join(tokens[i]) for i in keep], language='English'
        )
        inputs = inputs.to(self.model.device, self.model.dtype)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
        timed = self.processor.decode_forced_alignment(
            logits=logits,
            input_ids=inputs['input_ids'],
            word_lists=word_lists,
            timestamp_token_id=self.model.config.timestamp_token_id,
        )
        for sample, i in enumerate(keep):
            results[i] = self._attach(tokens[i], timed[sample], offsets[i])
        return results

    def _attach(self, tokens: List[str], timed: List[dict], offset: float) -> List[Word]:
        # The aligner drops punctuation-only tokens; keep the original spelling and hang those on the previous word.
        words: List[Word] = []
        cursor = 0
        for token in tokens:
            pieces = self.processor.split_words_for_alignment(token, 'English')
            if not pieces or cursor >= len(timed):
                if words:
                    words[-1].text = f'{words[-1].text} {token}'
                continue
            first, last = timed[cursor], timed[min(cursor + len(pieces), len(timed)) - 1]
            cursor += len(pieces)
            words.append(Word(token, offset + first['start_time'], offset + last['end_time']))
        return words


class Transcriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.qwen = QwenASR(settings)
        self.aligner = QwenAligner(settings)

    def __call__(self, audio: np.ndarray) -> List[Word]:
        started = time.perf_counter()
        bounds = chunk_bounds(audio, self.settings.chunk_s)
        pieces = [audio[a:b] for a, b in bounds]
        offsets = [a / SAMPLE_RATE for a, _ in bounds]
        batch = max(1, self.settings.asr_batch)

        texts: List[str] = []
        for i in range(0, len(pieces), batch):
            texts.extend(self.qwen.transcribe_texts(pieces[i:i + batch]))
        words = []
        for i in range(0, len(pieces), batch):
            for chunk_words in self.aligner.align_batch(pieces[i:i + batch], texts[i:i + batch], offsets[i:i + batch]):
                words.extend(chunk_words)

        logger.info('transcribed %.1f s of audio into %d words in %.1f s (%d chunks)',
                    len(audio) / SAMPLE_RATE, len(words), time.perf_counter() - started, len(bounds))
        return words
