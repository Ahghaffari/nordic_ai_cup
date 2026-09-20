"""Transcribe the supplied conversations once and cache the timed words under transcripts/<asr tag>/<conversation>.json.

    MED_ASR=qwen3 python tools/transcribe_train.py
    MED_ASR=whisper MED_ALIGNER=none python tools/transcribe_train.py --limit 5
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr import SAMPLE_RATE, Transcriber, load_audio  # noqa: E402
from settings import TRANSCRIPTS, Settings  # noqa: E402
from transcript import Transcript  # noqa: E402
from utils import group_questions_by_conversation, load_sample_audio  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    settings = Settings()
    out_dir = TRANSCRIPTS / settings.asr_tag
    print(f'asr tag {settings.asr_tag} -> {out_dir}', flush=True)
    transcriber = Transcriber(settings)
    transcriber(load_audio(load_sample_audio(group_questions_by_conversation()[0][0]))[:SAMPLE_RATE * 5])

    conversations = group_questions_by_conversation()
    if args.limit:
        conversations = conversations[:args.limit]

    total_audio = total_time = worst = 0.0
    for audio_filename, _ in conversations:
        path = out_dir / f'{Path(audio_filename).stem}.json'
        if path.exists() and not args.force:
            continue
        audio = load_audio(load_sample_audio(audio_filename))
        started = time.perf_counter()
        words = transcriber(audio)
        seconds = time.perf_counter() - started
        duration = len(audio) / SAMPLE_RATE
        Transcript(words).save(path, {'asr_tag': settings.asr_tag, 'seconds': seconds, 'audio_seconds': duration})
        total_audio += duration
        total_time += seconds
        worst = max(worst, seconds)
        print(f'{audio_filename:<32} {duration:6.1f} s audio  {len(words):4d} words  {seconds:6.1f} s', flush=True)

    if total_audio:
        print(f'\n{total_audio / 60:.1f} min audio in {total_time:.0f} s '
              f'(real-time factor {total_time / total_audio:.3f}, worst conversation {worst:.1f} s)')


if __name__ == '__main__':
    main()
