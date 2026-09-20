"""Every knob of the pipeline, overridable through MED_* environment variables."""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

STORAGE = Path(os.environ.get('MED_STORAGE', '/mnt/data/nordicai/medical-appointment'))
MODELS = STORAGE / 'models'
TRANSCRIPTS = STORAGE / 'transcripts'
RUNS = STORAGE / 'runs'


def _env(name: str, default, cast=str):
    value = os.environ.get(name)
    if value is None or value == '':
        return default
    if cast is bool:
        return value.lower() in ('1', 'true', 'yes', 'on')
    return cast(value)


@dataclass
class Settings:
    # speech recognition: Qwen3-ASR for the text, a forced aligner for the word timings
    asr_backend: str = field(default_factory=lambda: _env('MED_ASR', 'qwen3'))
    qwen_asr_model: str = field(default_factory=lambda: _env('MED_QWEN_ASR_MODEL', str(MODELS / 'Qwen3-ASR-1.7B-hf')))
    asr_prompt: str = field(default_factory=lambda: _env(
        'MED_ASR_PROMPT',
        'A recorded general practice consultation between a doctor and a patient. '
        'Medication names, doses in mg and ml, lab values, vaccinations and follow-up plans.'))
    asr_max_new_tokens: int = field(default_factory=lambda: _env('MED_ASR_MAX_NEW_TOKENS', 1536, int))

    # word timestamps: Qwen3-ForcedAligner over the recognised text
    aligner: str = field(default_factory=lambda: _env('MED_ALIGNER', 'qwen3'))
    aligner_model: str = field(default_factory=lambda: _env('MED_ALIGNER_MODEL', str(MODELS / 'Qwen3-ForcedAligner-0.6B-hf')))
    chunk_s: float = field(default_factory=lambda: _env('MED_CHUNK_S', 30.0, float))
    asr_batch: int = field(default_factory=lambda: _env('MED_ASR_BATCH', 8, int))

    device: str = field(default_factory=lambda: _env('MED_DEVICE', 'cuda:0'))
    torch_dtype: str = field(default_factory=lambda: _env('MED_TORCH_DTYPE', 'auto'))
    cpu_threads: int = field(default_factory=lambda: _env('MED_CPU_THREADS', 8, int))

    # answering LLM (GGUF through llama.cpp)
    llm_path: str = field(default_factory=lambda: _env('MED_LLM', str(MODELS / 'gguf' / 'gemma-4-12B-it-qat-UD-Q4_K_XL.gguf')))
    llm_gpu_layers: int = field(default_factory=lambda: _env('MED_LLM_GPU_LAYERS', -1, int))
    llm_main_gpu: int = field(default_factory=lambda: _env('MED_LLM_MAIN_GPU', -1, int))
    llm_split: bool = field(default_factory=lambda: _env('MED_LLM_SPLIT', False, bool))
    llm_ctx: int = field(default_factory=lambda: _env('MED_LLM_CTX', 4096, int))
    llm_batch: int = field(default_factory=lambda: _env('MED_LLM_BATCH', 512, int))
    llm_flash_attn: bool = field(default_factory=lambda: _env('MED_LLM_FLASH_ATTN', False, bool))
    llm_max_tokens: int = field(default_factory=lambda: _env('MED_LLM_MAX_TOKENS', 1400, int))
    llm_examples: int = field(default_factory=lambda: _env('MED_LLM_EXAMPLES', 1, int))

    # decisions and spans
    yes_threshold: float = field(default_factory=lambda: _env('MED_YES_THRESHOLD', 0.3, float))
    pad_start: float = field(default_factory=lambda: _env('MED_PAD_START', 0.0, float))
    pad_end: float = field(default_factory=lambda: _env('MED_PAD_END', -0.15, float))
    span_mode: str = field(default_factory=lambda: _env('MED_SPAN_MODE', 'line_quote'))
    min_quote_ratio: float = field(default_factory=lambda: _env('MED_MIN_QUOTE_RATIO', 0.6, float))
    span_for_no: bool = field(default_factory=lambda: _env('MED_SPAN_FOR_NO', False, bool))
    line_pause_s: float = field(default_factory=lambda: _env('MED_LINE_PAUSE_S', 0.8, float))
    line_max_words: int = field(default_factory=lambda: _env('MED_LINE_MAX_WORDS', 40, int))

    # serving
    budget_s: float = field(default_factory=lambda: _env('MED_BUDGET_S', 50.0, float))
    warmup: bool = field(default_factory=lambda: _env('MED_WARMUP', True, bool))
    transcript_cache: str = field(default_factory=lambda: _env('MED_TRANSCRIPT_CACHE', ''))

    @property
    def asr_tag(self) -> str:
        name = Path(self.qwen_asr_model).name
        return f'{self.asr_backend}-{name}-align_{self.aligner}-c{self.chunk_s:g}'

    def as_dict(self) -> dict:
        return asdict(self)
