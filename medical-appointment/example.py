"""Speech recognition + LLM verification behind the /predict endpoint.

Per request:
  1. decode the MP3 and transcribe it into timed words (Qwen3-ASR + Qwen3-ForcedAligner by default)
  2. ask a local LLM about all questions at once: line, verbatim quote and yes/no for each
  3. read P(yes) from the logits, answer yes above MED_YES_THRESHOLD
  4. snap each quote onto the timed words to get the evidence span

Anything that fails falls back to a lexical guess; nothing here raises. MED_PIPELINE=0 serves the original
baseline from example_baseline.py instead, without loading any model.
"""

import logging
import os
import time

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import audio_duration_seconds, decode_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PIPELINE = None
if os.environ.get('MED_PIPELINE', '1') != '0':
    from pipeline import Pipeline
    from settings import Settings

    SETTINGS = Settings()
    logger.info('settings: %s', SETTINGS.as_dict())
    PIPELINE = Pipeline(SETTINGS)
    if SETTINGS.warmup:
        PIPELINE.warmup()
else:
    import example_baseline

    logger.info('MED_PIPELINE=0: serving the baseline')


def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    if PIPELINE is None:
        return example_baseline.predict(request)

    started = time.perf_counter()
    arrived = time.monotonic()
    count = len(request.questions)
    answers, evidence_start, evidence_end = [True] * count, [None] * count, [None] * count

    try:
        audio_bytes = decode_audio(request.audio_base64)
        duration = audio_duration_seconds(audio_bytes)
        decisions = PIPELINE.run(audio_bytes, request.questions, arrived)
        for i, decision in enumerate(decisions):
            answers[i] = bool(decision.answer)
            if decision.span is not None:
                evidence_start[i], evidence_end[i] = decision.span
        logger.info(
            '%s (%.1f s): %d yes, %d spans, %d/%d answered by the llm, in %.1f s',
            request.audio_filename,
            duration if duration is not None else float('nan'),
            sum(answers),
            sum(s is not None for s in evidence_start),
            sum(d.source == 'llm' for d in decisions),
            count,
            time.perf_counter() - started,
        )
    except Exception:
        logger.exception('pipeline failed for %s, answering yes without evidence', request.audio_filename)

    return ASRQuestionResponseDto(answers=answers, evidence_start=evidence_start, evidence_end=evidence_end)
