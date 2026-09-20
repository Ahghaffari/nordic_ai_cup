"""Score the answering half on cached transcripts, without the server and without re-running speech recognition.

    python tools/eval_offline.py                         # LLM over every cached conversation, then score
    python tools/eval_offline.py --limit 5
    python tools/eval_offline.py --rescore runs/eval_.../raw.jsonl --pad-start 0.1   # re-score saved generations

Raw generations go to runs/eval_<utc>/raw.jsonl so thresholds, pads and span rules can be swept without the LLM.
Scoring uses the functions from utils.py, so the numbers match local_evaluator.py.
"""

import argparse
import collections
import dataclasses
import json
import logging
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_evaluator import ACCURACY_WEIGHT, TIOU_WEIGHT  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from settings import RUNS, TRANSCRIPTS, Settings  # noqa: E402
from transcript import Transcript  # noqa: E402
from utils import gold_evidence, group_questions_by_conversation, temporal_iou  # noqa: E402


def score(pipeline: Pipeline, conversations, raws, verbose: bool = False) -> dict:
    by_type = collections.defaultdict(lambda: [0, 0])
    correct = total = 0
    tious = []
    for (audio_filename, rows, transcript), raw in zip(conversations, raws):
        questions = [row['question'] for row in rows]
        decisions = pipeline.decide(transcript, questions, raw)
        for row, decision in zip(rows, decisions):
            label = int(row['label'])
            ok = int(decision.answer) == label
            correct += ok
            total += 1
            by_type[row['question_type']][0] += ok
            by_type[row['question_type']][1] += 1
            gold = gold_evidence(row)
            iou = 0.0
            if label == 1 and gold is not None:
                iou = temporal_iou(gold, decision.span if decision.answer else None)
                tious.append(iou)
            if verbose:
                mark = 'ok   ' if ok else 'WRONG'
                print(f'  {mark} {row["question_type"]:<13} p={decision.p_yes:.2f} tIoU={iou:.2f} '
                      f'L{decision.line} "{decision.quote[:60]}" | {row["question"]}')
    accuracy = correct / total if total else 0.0
    mean_tiou = statistics.mean(tious) if tious else 0.0
    return {
        'accuracy': accuracy,
        'tiou': mean_tiou,
        'score': ACCURACY_WEIGHT * accuracy + TIOU_WEIGHT * mean_tiou,
        'by_type': {k: v[0] / v[1] for k, v in by_type.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default=None, help='transcript cache tag (defaults to the current MED_ASR settings)')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--exclude', nargs='*', default=[], help='conversations to leave out (the prompt example)')
    parser.add_argument('--rescore', type=Path, default=None, help='raw.jsonl from an earlier run')
    parser.add_argument('--pad-start', type=float, default=None)
    parser.add_argument('--pad-end', type=float, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    settings = Settings()
    if args.pad_start is not None:
        settings.pad_start = args.pad_start
    if args.pad_end is not None:
        settings.pad_end = args.pad_end
    tag = args.tag or settings.asr_tag

    conversations = []
    for audio_filename, rows in group_questions_by_conversation():
        path = TRANSCRIPTS / tag / f'{Path(audio_filename).stem}.json'
        if path.exists() and audio_filename not in args.exclude:
            conversations.append((audio_filename, rows, Transcript.load(path, settings.line_pause_s, settings.line_max_words)))
    if args.limit:
        conversations = conversations[:args.limit]
    if not conversations:
        print(f'no cached transcripts under {TRANSCRIPTS / tag}; run tools/transcribe_train.py first')
        return

    if args.rescore:
        saved = {r['audio_filename']: r for r in map(json.loads, args.rescore.read_text().splitlines())}
        conversations = [c for c in conversations if c[0] in saved]
        raws = [saved[c[0]] for c in conversations]
        pipeline = Pipeline(settings, load_asr=False, load_llm=False)
    else:
        pipeline = Pipeline(settings, load_asr=False, load_llm=True)
        out_dir = RUNS / f'eval_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}'
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'settings.json').write_text(json.dumps({**settings.as_dict(), 'transcript_tag': tag}, indent=1))
        raws = []
        with open(out_dir / 'raw.jsonl', 'w') as f:
            for audio_filename, rows, transcript in conversations:
                questions = [row['question'] for row in rows]
                raw = pipeline.generate(transcript, questions, deadline=None)
                raw['audio_filename'] = audio_filename
                raws.append(raw)
                f.write(json.dumps(raw) + '\n')
                f.flush()
                print(f'{audio_filename:<32} {raw["seconds"]:5.1f} s  '
                      f'{raw["prompt_tokens"]} + {raw["completion_tokens"]} tokens', flush=True)
        print(f'raw generations saved to {out_dir / "raw.jsonl"}')
        seconds = [r['seconds'] for r in raws]
        print(f'llm seconds per conversation: mean {statistics.mean(seconds):.1f}, worst {max(seconds):.1f}')

    print(f'\ntranscripts {tag}, {len(conversations)} conversations, pads {settings.pad_start:+.2f}/{settings.pad_end:+.2f}')
    print(f'{"span mode":>10} {"threshold":>9}  {"acc":>5}  {"tIoU":>5}  {"score":>5}  positive  hard_neg  off_topic')
    for mode in ('quote', 'line_quote', 'lines'):
        for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
            variant = Pipeline(dataclasses.replace(settings, yes_threshold=threshold, span_mode=mode), load_asr=False, load_llm=False)
            result = score(variant, conversations, raws)
            t = result['by_type']
            print(f'{mode:>10} {threshold:>9.2f}  {result["accuracy"]:.3f}  {result["tiou"]:.3f}  {result["score"]:.3f}  '
                  f'{t.get("positive", 0):8.3f}  {t.get("hard_negative", 0):8.3f}  {t.get("off_topic", 0):9.3f}')

    if args.verbose:
        print(f'\nper question at threshold {settings.yes_threshold}, span mode {settings.span_mode}:')
        score(pipeline, conversations, raws, verbose=True)


if __name__ == '__main__':
    main()
