"""How well can spans built from our word timings match the annotations, and which padding helps?

For every annotated span it finds the best window of consecutive transcript words and the best single line
(or pair of adjacent lines). Those are upper bounds for word-level and line-level evidence. It then sweeps a
start/end pad on the best word windows, which is the padding to use in MED_PAD_START / MED_PAD_END.

    python tools/span_oracle.py --tag qwen3-Qwen3-ASR-1.7B-hf-align_qwen3
"""

import argparse
import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import TRANSCRIPTS, Settings  # noqa: E402
from transcript import Transcript  # noqa: E402
from utils import gold_evidence, group_questions_by_conversation, temporal_iou  # noqa: E402


def best_word_window(transcript: Transcript, gold, margin: float = 3.0):
    near = [i for i, w in enumerate(transcript.words) if w.end > gold[0] - margin and w.start < gold[1] + margin]
    best = (0.0, None)
    for a in near:
        for b in near:
            if b < a:
                continue
            span = (transcript.words[a].start, transcript.words[b].end)
            iou = temporal_iou(gold, span)
            if iou > best[0]:
                best = (iou, span)
    return best


def best_lines(transcript: Transcript, gold, max_lines: int = 8):
    best = 0.0
    lines = transcript.lines
    for i in range(len(lines)):
        for j in range(i, min(i + max_lines, len(lines))):
            best = max(best, temporal_iou(gold, (lines[i].start, lines[j].end)))
    return best


def best_line_start_word_end(transcript: Transcript, gold, max_lines: int = 8):
    """Span from the start of a line to the end of any word within the next few lines."""
    best = 0.0
    words, lines = transcript.words, transcript.lines
    for i, line in enumerate(lines):
        if line.start > gold[1] or line.start < gold[0] - 20:
            continue
        last = lines[min(i + max_lines, len(lines)) - 1].last
        for w in range(line.first, last + 1):
            best = max(best, temporal_iou(gold, (line.start, words[w].end)))
    return best


def structure(transcript: Transcript, gold, tolerance: float = 0.2) -> dict:
    lines, words = transcript.lines, transcript.words
    starts_line = any(abs(line.start - gold[0]) <= tolerance for line in lines)
    ends_line = any(abs(line.end - gold[1]) <= tolerance for line in lines)
    starts_word = any(abs(w.start - gold[0]) <= tolerance for w in words)
    ends_word = any(abs(w.end - gold[1]) <= tolerance for w in words)
    covered = sum(1 for line in lines if line.end > gold[0] + 0.05 and line.start < gold[1] - 0.05)
    return {'starts_line': starts_line, 'ends_line': ends_line, 'starts_word': starts_word,
            'ends_word': ends_word, 'lines': covered}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default=Settings().asr_tag)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    settings = Settings()

    word_ious, line_ious, start_ious, windows, shapes, missing = [], [], [], [], [], 0
    conversations = group_questions_by_conversation()
    if args.limit:
        conversations = conversations[:args.limit]
    for audio_filename, rows in conversations:
        path = TRANSCRIPTS / args.tag / f'{Path(audio_filename).stem}.json'
        if not path.exists():
            missing += 1
            continue
        transcript = Transcript.load(path, settings.line_pause_s, settings.line_max_words)
        for row in rows:
            gold = gold_evidence(row)
            if gold is None:
                continue
            iou, span = best_word_window(transcript, gold)
            word_ious.append(iou)
            line_ious.append(best_lines(transcript, gold))
            start_ious.append(best_line_start_word_end(transcript, gold))
            shapes.append(structure(transcript, gold))
            if span is not None:
                windows.append((gold, span))

    if not word_ious:
        print(f'no transcripts under {TRANSCRIPTS / args.tag}')
        return

    print(f'tag {args.tag}: {len(word_ious)} annotated spans ({missing} conversations not transcribed)')
    print(f'  best word window tIoU         mean {statistics.mean(word_ious):.3f}  median {statistics.median(word_ious):.3f}')
    print(f'  best whole lines tIoU         mean {statistics.mean(line_ious):.3f}  median {statistics.median(line_ious):.3f}')
    print(f'  best line start -> word end   mean {statistics.mean(start_ious):.3f}  median {statistics.median(start_ious):.3f}')
    share = lambda key: sum(s[key] for s in shapes) / len(shapes)
    print(f'  gold starts at a line start {share("starts_line"):.0%}, at a word start {share("starts_word"):.0%}; '
          f'ends at a line end {share("ends_line"):.0%}, at a word end {share("ends_word"):.0%} (within 0.2 s)')
    counts = collections.Counter(min(s['lines'], 6) for s in shapes)
    print('  lines touched by gold: ' + ', '.join(f'{k if k < 6 else "6+"}: {counts[k]}' for k in sorted(counts)))
    starts = [span[0] - gold[0] for gold, span in windows]
    ends = [gold[1] - span[1] for gold, span in windows]
    print(f'  gold starts {statistics.median(starts):+.3f} s vs word start, ends {statistics.median(ends):+.3f} s vs word end (median)')

    grid = [x / 20 for x in range(-6, 11)]
    scored = []
    for pad_start in grid:
        for pad_end in grid:
            ious = [temporal_iou(gold, (max(0.0, s - pad_start), max(0.0, e + pad_end))) for gold, (s, e) in windows]
            scored.append((statistics.mean(ious), pad_start, pad_end))
    scored.sort(reverse=True)
    print('  best pads on word windows (tIoU, MED_PAD_START, MED_PAD_END):')
    for iou, pad_start, pad_end in scored[:5]:
        print(f'    {iou:.3f}  {pad_start:+.2f}  {pad_end:+.2f}')


if __name__ == '__main__':
    main()
