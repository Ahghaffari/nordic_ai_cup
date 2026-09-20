"""Prompt, output schema and parsing for verifying the ten statements against one transcript, plus a lexical fallback."""

import json
import re
from typing import Dict, List, Optional, Tuple

from fewshot import EXAMPLES
from transcript import Transcript, norm_tokens

SYSTEM_PROMPT = """You check yes/no questions about a recorded consultation between a doctor and a patient.
You only have an automatic transcript of the recording, split into numbered lines (L0, L1, ...).

For each question decide whether the conversation explicitly establishes what the question asks.
- Answer "yes" only if every detail matches what was said: medicine, dose and unit, frequency, duration, \
body site, side, test, result, vaccine, timing, and who does what.
- Many questions are near-misses of something that was said: the right drug at the wrong dose, the right \
course at the wrong length, the right symptom in the wrong place. If any detail differs or is not stated, answer "no".
- If the subject never comes up, answer "no".
- A question phrased as a statement with a tag ("..., didn't it?", "..., right?") asks whether the statement is true.
- The transcript comes from speech recognition: tolerate misspelled words, but numbers and units must agree.

For every question also give evidence:
- "from" and "to": the first and last line of the passage that establishes the answer. Include every line needed \
(for example a question and the reply that confirms it) and nothing more; "to" equals "from" for a single line. \
For "no", point at the most related passage, or use -1 for both if nothing is related.
- "quote": the exact words, copied from those lines, that state the fact and nothing else. Start where the \
statement starts and stop as soon as it is established: this usually cuts a sentence short, and often skips the \
opening words of the sentence. Copy whole sentences only when the fact needs all of them. Use "" when "from" is -1.
- When the same fact is stated more than once, quote the statement that settles it, not an earlier hint.

The example that follows shows the expected format and, above all, where a quote begins and ends.
"""

def build_messages(transcript: Transcript, questions: List[str], examples: int = 1) -> List[Dict[str, str]]:
    numbered = '\n'.join(f'Q{i + 1}: {q}' for i, q in enumerate(questions))
    fields = '"q", "from", "to", "quote", "answer"'
    user = (
        f'Transcript:\n{transcript.render()}\n\n'
        f'Questions:\n{numbered}\n\n'
        f'Return JSON {{"results": [...]}} with exactly {len(questions)} objects in question order, '
        f'each with the keys {fields}.'
    )
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for example_user, example_assistant in EXAMPLES[:examples]:
        messages += [{'role': 'user', 'content': example_user}, {'role': 'assistant', 'content': example_assistant}]
    return messages + [{'role': 'user', 'content': user}]


def build_schema(count: int) -> dict:
    properties = {
        'q': {'type': 'integer'},
        'from': {'type': 'integer'},
        'to': {'type': 'integer'},
        'quote': {'type': 'string', 'maxLength': 240},
    }
    properties['answer'] = {'type': 'string', 'enum': ['yes', 'no']}
    item = {
        'type': 'object',
        'properties': properties,
        'required': list(properties),
        'additionalProperties': False,
    }
    return {
        'type': 'object',
        'properties': {'results': {'type': 'array', 'items': item, 'minItems': count, 'maxItems': count}},
        'required': ['results'],
        'additionalProperties': False,
    }


_OBJECT = re.compile(r'\{[^{}]*"last"\s*:\s*-?\d+[^{}]*\}')


def parse_refinements(text: str) -> Dict[int, Tuple[int, int]]:
    """{question number (1-based): (first word, last word)} from the refinement pass."""
    items = []
    try:
        items = json.loads(text).get('results', [])
    except (json.JSONDecodeError, AttributeError):
        for match in _REFINE_OBJECT.finditer(text):
            try:
                items.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue

    out: Dict[int, Tuple[int, int]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        q, first, last = _as_int(item.get('q')), _as_int(item.get('first')), _as_int(item.get('last'))
        if q is None or first is None or last is None or last < first:
            continue
        out[q] = (first, last)
    return out


_OBJECT = re.compile(r'\{[^{}]*"answer"\s*:\s*"(?:yes|no)"[^{}]*\}', re.IGNORECASE)
PREFILL = '{"results": [{"q": 1, "from": '


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_results(text: str, count: int, p_yes: Optional[List[float]] = None) -> List[Optional[dict]]:
    """Results by question position; a truncated generation still yields the objects it completed.

    `p_yes` holds one probability per generated answer in generation order and is attached as `p_yes`.
    """
    p_yes = p_yes or []
    items: List[dict] = []
    try:
        items = json.loads(text).get('results', [])
    except (json.JSONDecodeError, AttributeError):
        for match in _OBJECT.finditer(text):
            try:
                items.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue

    results: List[Optional[dict]] = [None] * count
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item['answer'] = str(item.get('answer', '')).strip().lower()
        if item['answer'] not in ('yes', 'no'):
            continue
        for key in ('q', 'from', 'to'):
            item[key] = _as_int(item.get(key))
        item['p_yes'] = p_yes[position] if position < len(p_yes) else (0.9 if item['answer'] == 'yes' else 0.1)
        index = item['q']
        index = index - 1 if index is not None and 1 <= index <= count else position
        if index < count and results[index] is None:
            results[index] = item
    return results


_STOPWORDS = set("""
a an the and or but if of to in on at for with by from as is are was were be been being do does did done has have had
it its this that these those there their they them he she his her him you your i me my we our us will would should
could can may might shall must not no yes any some all about into over after before than then so such very just also
patient doctor visit consultation mentioned mention discussed discuss talk talked said say right didn't isn't wasn't
aren't weren't doesn't don't hasn't haven't won't what which who whom whose when where why how whether any anything
""".split())


def content_tokens(text: str) -> List[str]:
    return [t.rstrip('s') if len(t) > 4 else t for t in norm_tokens(text) if t not in _STOPWORDS]


def lexical_best_line(transcript: Transcript, question: str) -> Tuple[Optional[int], float]:
    wanted = set(content_tokens(question))
    if not wanted or not transcript.lines:
        return None, 0.0
    best, best_score = None, 0.0
    for line in transcript.lines:
        have = set(content_tokens(line.text))
        score = len(wanted & have) / len(wanted)
        if score > best_score:
            best, best_score = line.id, score
    return best, best_score
