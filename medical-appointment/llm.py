"""llama.cpp wrapper: chat template from the GGUF, JSON output, and yes/no probabilities.

The probability of every answer is read from the logits at the step where the model is about to write the
answer value, so one generation yields both the decision and a calibrated-enough score for thresholding.

The JSON-schema grammar is optional: llama-cpp-python checks it against the whole 248k-token vocabulary at every
step, which cuts generation from about 32 to 12 tokens/s on the P100. Without it, the assistant turn is prefilled
with the start of the expected JSON and parsing is tolerant instead.
"""

import json
import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from asr import parse_device
from settings import Settings

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r'\s+')


class AnswerProbe:
    """Logits processor that records P(yes) each time the generated text reaches `"answer":"`."""

    def __init__(self, llm, prompt_len: int, key: str = 'answer'):
        self.llm = llm
        self.prompt_len = prompt_len
        self.marker = f'"{key}":"'
        self.yes_ids = self._first_token_ids(['yes'])
        self.no_ids = self._first_token_ids(['no'])
        self.p_yes: List[float] = []
        self.first_call: Optional[float] = None
        self._last_fired = -1

    def _first_token_ids(self, words) -> List[int]:
        ids = set()
        for word in words:
            tokens = self.llm.tokenize(word.encode(), add_bos=False, special=False)
            if tokens:
                ids.add(tokens[0])
        return sorted(ids)

    def __call__(self, input_ids, scores):
        if self.first_call is None:
            self.first_call = time.perf_counter()
        position = len(input_ids)
        if position - self.prompt_len < 4 or position == self._last_fired:
            return scores
        tail = self.llm.detokenize(list(input_ids[-8:])).decode('utf-8', errors='ignore')
        if _WHITESPACE.sub('', tail).endswith(self.marker):
            logits = np.asarray(scores, dtype=np.float64)
            yes = np.logaddexp.reduce(logits[self.yes_ids])
            no = np.logaddexp.reduce(logits[self.no_ids])
            self.p_yes.append(float(1.0 / (1.0 + np.exp(no - yes))))
            self._last_fired = position
        return scores


class LlamaEngine:
    def __init__(self, settings: Settings):
        import jinja2
        import jinja2.ext
        import jinja2.sandbox
        import llama_cpp

        self.llama_cpp = llama_cpp
        kind, index = parse_device(settings.device)
        gpu_layers = settings.llm_gpu_layers if kind == 'cuda' else 0
        main_gpu = settings.llm_main_gpu if settings.llm_main_gpu >= 0 else index
        split_mode = llama_cpp.LLAMA_SPLIT_MODE_LAYER if settings.llm_split else llama_cpp.LLAMA_SPLIT_MODE_NONE
        self.llm = llama_cpp.Llama(
            model_path=settings.llm_path,
            n_gpu_layers=gpu_layers,
            split_mode=split_mode,
            main_gpu=main_gpu,
            n_ctx=settings.llm_ctx,
            n_batch=settings.llm_batch,
            n_threads=settings.cpu_threads,
            flash_attn=settings.llm_flash_attn,
            seed=0,
            verbose=False,
        )

        template = self.llm.metadata.get('tokenizer.chat_template')
        if not template:
            raise ValueError(f'{settings.llm_path} carries no chat template')
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
        )
        env.globals['raise_exception'] = _raise
        env.globals['strftime_now'] = lambda fmt: datetime.now().strftime(fmt)
        self.template = env.from_string(template)
        self.bos_text = self._token_text(self.llm.token_bos())
        self.eos_text = self._token_text(self.llm.token_eos())
        self.add_bos = str(self.llm.metadata.get('tokenizer.ggml.add_bos_token', 'false')).lower() == 'true'

    def _token_text(self, token: int) -> str:
        if token is None or token < 0:
            return ''
        return self.llm.detokenize([token], special=True).decode('utf-8', errors='ignore')

    def prompt_tokens(self, messages: List[Dict[str, str]]) -> List[int]:
        try:
            prompt = self._render(messages)
        except Exception:
            # Some templates (gemma) refuse a system turn; fold it into the first user message instead.
            prompt = self._render(_merge_system(messages))
        tokens = self.llm.tokenize(prompt.encode('utf-8'), add_bos=False, special=True)
        bos = self.llm.token_bos()
        if self.add_bos and bos >= 0 and (not tokens or tokens[0] != bos):
            tokens = [bos] + tokens
        return tokens

    def _render(self, messages: List[Dict[str, str]]) -> str:
        return self.template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=False,
            bos_token=self.bos_text,
            eos_token=self.eos_text,
        )

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        schema: dict,
        max_tokens: int,
        deadline: Optional[float] = None,
        prefill: str = '',
        use_grammar: bool = False,
    ) -> Dict:
        llama_cpp = self.llama_cpp
        tokens = self.prompt_tokens(messages)
        if prefill and not use_grammar:
            tokens += self.llm.tokenize(prefill.encode('utf-8'), add_bos=False, special=False)
        else:
            prefill = ''
        probe = AnswerProbe(self.llm, len(tokens))
        grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False) if use_grammar else None

        def past_deadline(input_ids, logits) -> bool:
            return deadline is not None and time.monotonic() > deadline

        started = time.perf_counter()
        output = self.llm.create_completion(
            prompt=tokens,
            max_tokens=max_tokens,
            temperature=0.0,
            top_k=1,
            seed=0,
            grammar=grammar,
            logits_processor=llama_cpp.LogitsProcessorList([probe]),
            stopping_criteria=llama_cpp.StoppingCriteriaList([past_deadline]),
        )
        choice = output['choices'][0]
        usage = output.get('usage', {})
        elapsed = time.perf_counter() - started
        prompt_s = (probe.first_call or time.perf_counter()) - started
        logger.info('llm: %s prompt tokens in %.1f s + %s generated in %.1f s (finish=%s)',
                    usage.get('prompt_tokens'), prompt_s, usage.get('completion_tokens'), elapsed - prompt_s,
                    choice.get('finish_reason'))
        return {
            'text': prefill + choice['text'],
            'p_yes': probe.p_yes,
            'finish_reason': choice.get('finish_reason'),
            'prompt_tokens': usage.get('prompt_tokens'),
            'completion_tokens': usage.get('completion_tokens'),
            'seconds': elapsed,
            'prompt_seconds': prompt_s,
        }


def _merge_system(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    system = [m['content'] for m in messages if m['role'] == 'system']
    rest = [m for m in messages if m['role'] != 'system']
    if system and rest:
        rest[0] = {'role': rest[0]['role'], 'content': '\n\n'.join(system + [rest[0]['content']])}
    return rest


def _raise(message):
    raise ValueError(message)
