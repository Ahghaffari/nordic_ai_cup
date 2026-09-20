"""Timed words, the numbered lines shown to the LLM, and text normalisation shared by matching and scoring."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_SENTENCE_END = ('.', '?', '!')


def norm_tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Line:
    id: int
    first: int
    last: int
    start: float
    end: float
    text: str


class Transcript:
    def __init__(self, words: List[Word], pause_s: float = 0.8, max_words: int = 40):
        self.words = words
        self.lines = build_lines(words, pause_s, max_words)

    @property
    def duration(self) -> float:
        return self.words[-1].end if self.words else 0.0

    def render(self) -> str:
        return '\n'.join(f'L{line.id}: {line.text}' for line in self.lines)

    def save(self, path: Path, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'meta': meta, 'words': [asdict(w) for w in self.words]}, indent=1))

    @classmethod
    def load(cls, path: Path, pause_s: float = 0.8, max_words: int = 40) -> 'Transcript':
        data = json.loads(Path(path).read_text())
        return cls([Word(**w) for w in data['words']], pause_s, max_words)


def build_lines(words: List[Word], pause_s: float, max_words: int) -> List[Line]:
    lines: List[Line] = []
    first = 0
    for i, word in enumerate(words):
        last_word = i == len(words) - 1
        gap = 0.0 if last_word else words[i + 1].start - word.end
        if last_word or word.text.endswith(_SENTENCE_END) or gap >= pause_s or i - first + 1 >= max_words:
            text = ' '.join(w.text for w in words[first:i + 1])
            lines.append(Line(len(lines), first, i, words[first].start, word.end, text))
            first = i + 1
    return lines
