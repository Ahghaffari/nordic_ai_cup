"""Audio -> timed transcript -> one LLM pass over all questions -> answers and evidence spans."""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from answering import PREFILL, build_messages, build_schema, lexical_best_line, parse_results
from settings import Settings
from spans import QuoteMatcher, Span, clamp
from transcript import Transcript

logger = logging.getLogger(__name__)

LEXICAL_YES = 0.5
# Below this much of the budget left there is no point starting: transcription alone costs more.
MIN_SECONDS_TO_TRY = 20.0


@dataclass
class Decision:
    answer: bool
    p_yes: float
    span: Optional[Span]
    line: Optional[int]
    quote: str
    source: str


class Pipeline:
    def __init__(self, settings: Optional[Settings] = None, load_asr: bool = True, load_llm: bool = True):
        self.settings = settings or Settings()
        self.lock = threading.Lock()
        self.transcriber = None
        self.engine = None
        if load_asr:
            from asr import Transcriber

            self.transcriber = Transcriber(self.settings)
        if load_llm:
            from llm import LlamaEngine

            self.engine = LlamaEngine(self.settings)

    def warmup(self) -> None:
        from asr import SAMPLE_RATE

        started = time.perf_counter()
        t = np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
        tone = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        if self.transcriber is not None:
            try:
                self.transcriber(tone)
            except Exception:
                logger.exception('asr warmup failed')
        if self.engine is not None:
            transcript = Transcript([], self.settings.line_pause_s, self.settings.line_max_words)
            self.answer(transcript, ['Was a vaccine given?'], deadline=None)
        logger.info('warmup done in %.1f s', time.perf_counter() - started)

    def transcribe(self, audio_bytes: bytes) -> Transcript:
        from asr import load_audio

        s = self.settings
        cache = Path(s.transcript_cache) / s.asr_tag / f'{hashlib.sha1(audio_bytes).hexdigest()}.json' if s.transcript_cache else None
        if cache is not None and cache.exists():
            return Transcript.load(cache, s.line_pause_s, s.line_max_words)

        started = time.perf_counter()
        words = self.transcriber(load_audio(audio_bytes))
        transcript = Transcript(words, s.line_pause_s, s.line_max_words)
        self._release_cached_blocks()
        if cache is not None:
            transcript.save(cache, {'asr_tag': s.asr_tag, 'seconds': time.perf_counter() - started})
        return transcript

    @staticmethod
    def _release_cached_blocks() -> None:
        """Hand the allocator's spare blocks back after transcription.

        Every longer conversation than the last reserves more, and the card is shared with the answering model,
        so the high-water mark is what a long evaluation conversation would collide with.
        """
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.exception('could not release cached blocks')

    def generate(self, transcript: Transcript, questions: List[str], deadline: Optional[float]) -> dict:
        """One generation covering every question: line, verbatim quote and yes/no for each."""
        s = self.settings
        raw = {'text': '', 'p_yes': []}
        if self.engine is None or not questions:
            return raw
        try:
            raw = self.engine.generate_json(
                build_messages(transcript, questions, s.llm_examples),
                build_schema(len(questions)),
                max_tokens=s.llm_max_tokens,
                deadline=deadline,
                prefill=PREFILL,
            )
        except Exception:
            logger.exception('llm failed, answering lexically')
            return raw

        return raw

    def answer(self, transcript: Transcript, questions: List[str], deadline: Optional[float]) -> List[Decision]:
        return self.decide(transcript, questions, self.generate(transcript, questions, deadline))

    def decide(self, transcript: Transcript, questions: List[str], raw: dict) -> List[Decision]:
        s = self.settings
        results = parse_results(raw.get('text', ''), len(questions), raw.get('p_yes'))
        parsed = sum(r is not None for r in results)
        if parsed < len(questions) and self.engine is not None:
            logger.warning('parsed %d/%d answers (finish=%s), generation tail: %r',
                           parsed, len(questions), raw.get('finish_reason'), raw.get('text', '')[-300:])
        matcher = QuoteMatcher(transcript)

        decisions: List[Decision] = []
        for index, (question, result) in enumerate(zip(questions, results)):
            if result is not None:
                p_yes = result['p_yes']
                line = result.get('from')
                quote = str(result.get('quote') or '')
                span = matcher.evidence(quote, line, result.get('to'), s.span_mode, s.min_quote_ratio, s.pad_start, s.pad_end)
                source = 'llm'
            else:
                line, score = lexical_best_line(transcript, question)
                p_yes = 0.6 if score >= LEXICAL_YES else 0.1
                quote = ''
                span = matcher.line_span(line)
                source = 'lexical'

            answer = p_yes >= s.yes_threshold
            if not answer and not s.span_for_no:
                span = None
            decisions.append(Decision(answer, p_yes, span, line, quote, source))
        return decisions

    def run(self, audio_bytes: bytes, questions: List[str], arrived: Optional[float] = None) -> List[Decision]:
        """`arrived` is when the request reached the server, not when it reached the model.

        The budget has to run from arrival. Starting it after the lock gives a request that queued behind a slow
        one a fresh full budget on top of the wait, so one overrun becomes a run of timeouts, and five in a row
        ends the attempt. Counting the wait means a late request answers with whatever it has instead.
        """
        arrived = time.monotonic() if arrived is None else arrived
        with self.lock:
            deadline = arrived + self.settings.budget_s
            if time.monotonic() > deadline - MIN_SECONDS_TO_TRY:
                logger.warning('request already %.1f s old at the front of the queue, guessing without the audio',
                               time.monotonic() - arrived)
                return [Decision(True, 1.0, None, None, '', 'queued-out') for _ in questions]
            transcript = self.transcribe(audio_bytes)
            return self.answer(transcript, questions, deadline)
