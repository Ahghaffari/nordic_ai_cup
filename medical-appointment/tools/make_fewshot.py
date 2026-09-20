"""Turn supplied conversations into worked examples for the prompt.

The annotations mark where a passage starts and stops, which the model does not guess well on its own: it copies
whole sentences where the annotation often stops at the last word that carries the fact. This prints an example
block with the quote taken from the words the annotation actually covers.

    python tools/make_fewshot.py --conversations conversation_sample_4.mp3 conversation_sample_17.mp3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import TRANSCRIPTS, Settings  # noqa: E402
from transcript import Transcript  # noqa: E402
from utils import gold_evidence, group_questions_by_conversation  # noqa: E402


def gold_quote(transcript: Transcript, gold) -> str:
    inside = [w for w in transcript.words if w.end > gold[0] + 0.05 and w.start < gold[1] - 0.05]
    return ' '.join(w.text for w in inside)


def gold_lines(transcript: Transcript, gold):
    touched = [line for line in transcript.lines if line.end > gold[0] + 0.05 and line.start < gold[1] - 0.05]
    return (touched[0].id, touched[-1].id) if touched else (-1, -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default=Settings().asr_tag)
    parser.add_argument('--conversations', nargs='+', required=True)
    parser.add_argument('--questions', type=int, default=6)
    args = parser.parse_args()
    settings = Settings()

    for audio_filename, rows in group_questions_by_conversation():
        if audio_filename not in args.conversations:
            continue
        path = TRANSCRIPTS / args.tag / f'{Path(audio_filename).stem}.json'
        transcript = Transcript.load(path, settings.line_pause_s, settings.line_max_words)
        chosen = rows[:args.questions]

        print(f'--- {audio_filename}')
        print('Transcript:')
        print(transcript.render())
        print('\nQuestions:')
        for i, row in enumerate(chosen, 1):
            print(f'Q{i}: {row["question"]}')
        print('\nExpected:')
        print('{"results": [')
        for i, row in enumerate(chosen, 1):
            gold = gold_evidence(row)
            if gold is None:
                print(f'  {{"q": {i}, "from": -1, "to": -1, "quote": "", "answer": "no"}},')
                continue
            first, last = gold_lines(transcript, gold)
            print(f'  {{"q": {i}, "from": {first}, "to": {last}, "quote": "{gold_quote(transcript, gold)}", "answer": "yes"}},')
        print(']}\n')


if __name__ == '__main__':
    main()
