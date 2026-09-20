"""Exercise the whole pipeline on a short clip with small models on CPU.

    python tools/smoke_test.py --part asr            # Qwen3-ASR-0.6B + forced aligner on the first 20 s
    python tools/smoke_test.py --part llm            # Qwen3.5-0.8B over that transcript, decisions and spans
    python tools/smoke_test.py --part all

It checks that every stage runs and that the response passes utils.validate_response, not answer quality.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SMOKE_ENV = {
    'MED_DEVICE': 'cpu',
    'MED_CPU_THREADS': '4',
    'MED_ASR': 'qwen3',
    'MED_QWEN_ASR_MODEL': '/mnt/data/nordicai/medical-appointment/models/Qwen3-ASR-0.6B-hf',
    'MED_LLM': '/mnt/data/nordicai/medical-appointment/models/gguf/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf',
    'MED_LLM_CTX': '4096',
    'MED_LLM_MAX_TOKENS': '600',
}
for key, value in SMOKE_ENV.items():
    os.environ.setdefault(key, value)

from asr import SAMPLE_RATE, Transcriber, load_audio  # noqa: E402
from dtos import ASRQuestionResponseDto  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from settings import Settings  # noqa: E402
from transcript import Transcript  # noqa: E402
from utils import gold_evidence, group_questions_by_conversation, load_sample_audio, validate_response  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['asr', 'llm', 'all'], default='all')
    parser.add_argument('--seconds', type=float, default=20.0)
    parser.add_argument('--conversation', type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    settings = Settings()
    audio_filename, rows = group_questions_by_conversation()[args.conversation]
    audio = load_audio(load_sample_audio(audio_filename))[:int(args.seconds * SAMPLE_RATE)]
    cache = ROOT / 'storage' / 'transcripts' / 'smoke' / f'{Path(audio_filename).stem}_{int(args.seconds)}s.json'

    if args.part in ('asr', 'all'):
        started = time.perf_counter()
        transcriber = Transcriber(settings)
        print(f'asr models loaded in {time.perf_counter() - started:.1f} s')
        started = time.perf_counter()
        words = transcriber(audio)
        print(f'transcribed {args.seconds:.0f} s in {time.perf_counter() - started:.1f} s, {len(words)} words')
        transcript = Transcript(words, settings.line_pause_s, settings.line_max_words)
        transcript.save(cache, {'smoke': True})
        for w in words[:12]:
            print(f'  {w.start:6.2f}-{w.end:6.2f} {w.text}')
        print(transcript.render())
        assert words and all(w.end >= w.start for w in words), 'no words or reversed timings'
        assert all(a.start <= b.start + 1e-6 for a, b in zip(words, words[1:])), 'word starts not monotonic'

    if args.part in ('llm', 'all'):
        transcript = Transcript.load(cache, settings.line_pause_s, settings.line_max_words)
        inside = [r for r in rows if (gold_evidence(r) or (0, 0))[1] <= args.seconds] or rows[:3]
        questions = [r['question'] for r in inside] + [r['question'] for r in rows if r['question_type'] != 'positive'][:2]
        pipeline = Pipeline(settings, load_asr=False, load_llm=True)
        started = time.perf_counter()
        decisions = pipeline.answer(transcript, questions, deadline=time.monotonic() + 120)
        print(f'answered {len(questions)} questions in {time.perf_counter() - started:.1f} s')
        for question, d in zip(questions, decisions):
            print(f'  {"yes" if d.answer else "no ":<3} p={d.p_yes:.2f} {d.source:<7} L{d.line} span={d.span} "{d.quote}" | {question}')
        response = ASRQuestionResponseDto(
            answers=[d.answer for d in decisions],
            evidence_start=[d.span[0] if d.span else None for d in decisions],
            evidence_end=[d.span[1] if d.span else None for d in decisions],
        )
        validate_response(response, expected_count=len(questions))
        assert any(d.source == 'llm' for d in decisions), 'llm produced nothing parseable'
        print('response valid')


if __name__ == '__main__':
    main()
