"""Turn the LLM's evidence (a line range and a quote) into a start/end second.

Span modes:
  quote       the transcript words matched by the quote
  lines       the whole lines from..to
  line_quote  from the start of the line holding the quote's first word to the quote's last word
Every mode falls back to the other source when its own evidence is missing or unmatched.
"""

from bisect import bisect_right
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from transcript import Transcript, norm_tokens

Span = Tuple[float, float]


class QuoteMatcher:
    def __init__(self, transcript: Transcript):
        self.transcript = transcript
        self.tokens: List[str] = []
        self.owner: List[int] = []
        for index, word in enumerate(transcript.words):
            for token in norm_tokens(word.text):
                self.tokens.append(token)
                self.owner.append(index)

    def match(self, quote: str, line_id: Optional[int] = None) -> Optional[Tuple[int, int, float]]:
        """(first word, last word, similarity) of the transcript window closest to the quote."""
        wanted = norm_tokens(quote)
        if not wanted or not self.tokens:
            return None

        line = self._line(line_id)
        n, size = len(self.tokens), len(wanted)
        wanted_set = set(wanted)
        best = None
        for start in range(n):
            if self.tokens[start] not in wanted_set:
                continue
            for length in range(max(1, size - 2), size + 3):
                end = start + length
                if end > n:
                    break
                window = self.tokens[start:end]
                score = SequenceMatcher(None, wanted, window, autojunk=False).ratio()
                if line is not None and line.first <= self.owner[start] <= line.last:
                    score += 0.02
                if best is None or score > best[2]:
                    best = (self.owner[start], self.owner[end - 1], score)
                if score >= 1.0:
                    return best
        return best

    def _line(self, line_id: Optional[int]):
        if isinstance(line_id, int) and 0 <= line_id < len(self.transcript.lines):
            return self.transcript.lines[line_id]
        return None

    def evidence(
        self,
        quote: str,
        first_line: Optional[int],
        last_line: Optional[int],
        mode: str,
        min_ratio: float,
        pad_start: float = 0.0,
        pad_end: float = 0.0,
    ) -> Optional[Span]:
        words = self.transcript.words
        lines = self.lines_span(first_line, last_line)
        found = self.match(quote, first_line) if quote else None
        if found is not None and found[2] < min_ratio:
            found = None

        if mode == 'lines' and lines is not None:
            start, end = lines
        elif mode == 'line_quote' and found is not None:
            start, end = self.line_of_word(found[0]).start, words[found[1]].end
        elif found is not None:
            start, end = words[found[0]].start, words[found[1]].end
        elif lines is not None:
            start, end = lines
        else:
            return None
        return clamp(start - pad_start, end + pad_end)

    def lines_span(self, first_line: Optional[int], last_line: Optional[int], max_lines: int = 12) -> Optional[Span]:
        first = self._line(first_line)
        if first is None:
            return None
        last = self._line(last_line)
        if last is None or last.id < first.id or last.id - first.id >= max_lines:
            last = first
        return first.start, last.end

    def line_of_word(self, index: int):
        firsts = [line.first for line in self.transcript.lines]
        return self.transcript.lines[max(0, bisect_right(firsts, index) - 1)]

    def line_span(self, line_id: Optional[int]) -> Optional[Span]:
        line = self._line(line_id)
        return None if line is None else clamp(line.start, line.end)


def clamp(start: float, end: float) -> Optional[Span]:
    start = max(0.0, float(start))
    end = max(start, float(end))
    return round(start, 3), round(end, 3)
