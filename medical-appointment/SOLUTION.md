# Medical appointment — solution

Evaluation score **0.7530**. Validation 0.7863.

## Approach

One request carries a conversation and its ten questions, so the expensive half runs once and the
cheap half ten times: transcribe the audio, then answer every question against that one transcript in
a single generation.

```
MP3 -> 16 kHz mono
    -> chunks cut at the quietest 20 ms frame before each 30 s limit
    -> Qwen3-ASR-1.7B (batched)                      text
    -> Qwen3-ForcedAligner-0.6B                      one timestamp per word
    -> words grouped into numbered lines L0, L1, ... on sentence punctuation,
       a 0.8 s pause, or 40 words
    -> one llama.cpp generation over all ten questions
       returning {from, to, quote, answer} for each
    -> P(yes) read from the logits at each answer
    -> the quote fuzzy-matched back onto the timed words -> start and end second
```

### Why it is built this way

**The model cannot see time, so it quotes and we look the quote up.** Asking a language model for
timestamps directly does not work; asking it to copy the words that establish a fact does. The quote
is matched onto the timed words with a sliding window over normalised tokens, which tolerates the
recogniser's spelling while keeping the alignment exact. Match quality is high in practice: median
similarity 1.02, only 2% below 0.9.

**Separate recogniser and aligner.** Qwen3-ASR has no timestamps of its own and Whisper's DTW timings
are looser than a forced aligner's. Since 60% of the score is the span, the text comes from the
stronger recogniser and a purpose-built aligner times it. Measured against the annotations, gold
starts land 0.04 s from the nearest word start and gold ends 0.18 s before the nearest word end,
which is what `pad_end = -0.15` compensates for.

**One generation, not ten.** Ten separate calls do not fit the 60 s budget. The assistant turn is
prefilled with the opening of the expected JSON rather than constrained by a grammar: llama.cpp
checks a JSON-schema grammar against the whole 248k-token vocabulary at every step, which cuts
generation from 32 to 12 tokens/s. Prefilling produced identical output at full speed, and parsing is
tolerant instead — a truncated generation still yields every object it completed.

**The decision comes free with the generation.** A logits processor watches for the point where the
text reaches `"answer":"` and reads P(yes) from the distribution there, so one greedy pass yields both
the answer and a score to threshold.

**Answering yes is worth more than it looks.** A missed positive loses its accuracy mark *and* scores
zero on the evidence half, which averages over every annotated yes whether or not it was found. A
wrong yes only costs the accuracy mark. Saying yes pays whenever P(yes) > 1 / (2 + 3·tIoU), roughly
0.25–0.30 at realistic tIoU — hence a threshold of 0.10, well below the halfway point.

### Robustness

The budget runs from the moment a request arrives, not from when it reaches the model. Starting the
clock after the lock gives a request that queued behind a slow one a fresh full budget on top of the
wait, so one overrun becomes a run of timeouts and five consecutive timeouts ends an attempt. Nothing
in the request path raises: any failure returns a guess for every question, which is worth half a
mark on average where an error is worth none.

## Results

| | accuracy | mean tIoU | score |
|---|---|---|---|
| supplied baseline (always yes, no spans) | 0.500 | 0.000 | 0.200 |
| this solution, 39 supplied conversations | 0.990 | 0.616 | 0.766 |
| **evaluation, 38 conversations** | | | **0.7530** |

Accuracy is effectively solved: 0.99 leaves five errors in 390, worth 0.006 of score. The remaining
gap is entirely in where the evidence span is cut. Decomposed over 178 annotated positives:

| | mean tIoU |
|---|---|
| as submitted | 0.617 |
| same passage, boundaries trimmed perfectly | 0.767 |
| right passage and right boundaries (oracle over the transcript's own words) | 0.890 |

So 55% of the remaining loss is boundary placement and 45% is choosing the wrong mention when a fact
is stated twice. The quote is the right *length* — 9 words median against 8 for the annotations — and
simply sits in a different place.

Nine approaches were measured against that gap: a second refinement pass, a second worked example, a
learned span ranker, a later-mention rule, extractive QA (DeBERTa-v3-large-squad2), NLI re-ranking,
MBR over several models, and two ways of enriching the worked example. None cleared the ±0.014 noise
floor, and the two that moved beyond it made things worse. `EXPERIMENTS.md` records each with numbers.

The reason is visible in the disagreements. Where the annotator and the model pick different mentions,
the model's citation matches the question's wording *better* — question-token coverage 0.35 against
0.20 — in 89% of cases. Matching the annotator would mean preferring the less on-point passage, which
no ranking signal in the transcript supports.

## Running it

```bash
pip install -r requirements.txt
python tools/download_models.py     # Qwen3-ASR-1.7B, Qwen3-ForcedAligner-0.6B, the answering GGUF
./serve.sh start                    # serves :9054, or: python api.py
python local_evaluator.py           # scores the 39 supplied conversations
```

All three models are public downloads; nothing custom is shipped.

| role | model |
|---|---|
| speech recognition | `Qwen/Qwen3-ASR-1.7B-hf` |
| word timestamps | `Qwen/Qwen3-ForcedAligner-0.6B-hf` |
| answering | `gemma-4-26B-A4B-it-qat` Q4_K_XL (GGUF, via llama.cpp) |

Roughly 19 GB on disk. The answering model needs a 16 GB card to itself; `serve.sh` puts the
recogniser on the second GPU and refuses to start rather than quietly serving a smaller model if the
memory is not there.

Every setting is an `MED_*` environment variable listed in `settings.py`. The ones that matter are
pinned in `serve.sh` so a restart reproduces what was submitted.

## Timing

Fitted over 85 requests: `reply = 12.8 s + 0.059 × audio_seconds`. The longest supplied conversation
(232 s) answers in 26.5 s against a 60 s budget; a ten-minute conversation would still fit. Generation
stops 50 s after the request arrives regardless, so an unusually long one degrades rather than times
out.

## Layout

| file | contents |
|---|---|
| `example.py` | the `/predict` entry point |
| `pipeline.py` | transcribe → generate → decide |
| `asr.py` | chunking, recognition, forced alignment |
| `transcript.py` | timed words and the numbered lines shown to the model |
| `answering.py` | prompt, output schema, tolerant parsing, lexical fallback |
| `fewshot.py` | the worked example, quotes taken from real annotations |
| `llm.py` | llama.cpp wrapper, chat template, the P(yes) logits probe |
| `spans.py` | quote → start and end second |
| `settings.py` | every knob, each overridable by an environment variable |
| `tools/` | offline scoring, span oracles, transcript caching, model download |
| `EXPERIMENTS.md` | what was measured, including what failed |
