# Medical appointment experiments

Score = 0.4 x accuracy + 0.6 x mean tIoU over the annotated yes questions. Gold spans are short (median 2.9 s,
middle half 1.9-4.0 s), so evidence must be phrase-level word timings, not whisper segments.

## Pipeline (example.py)

| stage | default | switch |
|---|---|---|
| speech recognition | Qwen3-ASR-1.7B (transformers) | `MED_ASR=qwen3\|whisper`, `MED_WHISPER_MODEL` |
| word timings | Qwen3-ForcedAligner-0.6B, 180 s chunks cut at silence | `MED_ALIGNER=qwen3\|none`, `MED_CHUNK_S` |
| answering | Qwen3.5-9B Q4_K_M via llama.cpp, one JSON-schema generation for all questions | `MED_LLM`, `MED_LLM_REASON=1` |
| decision | P(yes) from the logits at each answer, yes if >= 0.3 | `MED_YES_THRESHOLD` |
| evidence | verbatim quote fuzzy-matched onto the timed words, line span as fallback | `MED_PAD_START`, `MED_PAD_END`, `MED_MIN_QUOTE_RATIO` |
| budget | generation stops 50 s after the request arrived; unanswered questions fall back to lexical overlap | `MED_BUDGET_S` |

Why yes at 0.3: a missed yes loses its accuracy mark and its tIoU, a wrong yes only the accuracy mark, so yes
pays whenever P(yes) > 1 / (2 + 3 x tIoU), about 0.25-0.3 at tIoU 0.5-0.7.

Open question for the organizers: `local_evaluator.py` / `utils.mean_temporal_iou` credit a span returned with a
`false` answer, the README says a no scores 0 there. Kept to the README (`MED_SPAN_FOR_NO=0`).

## Workflow once a GPU is free

```bash
source /mnt/data/virtual_environments/nordicai-medical-appointment/bin/activate
export MED_DEVICE=cuda:1
python tools/transcribe_train.py                                  # Qwen3-ASR + aligner transcripts
MED_ASR=whisper MED_ALIGNER=none  python tools/transcribe_train.py
MED_ASR=whisper MED_ALIGNER=qwen3 python tools/transcribe_train.py
python tools/span_oracle.py --tag <tag>                           # best achievable tIoU per timing source + pads
python tools/eval_offline.py --tag <tag>                          # LLM on cached transcripts, threshold sweep
python tools/eval_offline.py --rescore <run>/raw.jsonl --pad-start 0.1 --pad-end 0.1
MEDICAL_PORT=9074 python api.py & python local_evaluator.py --url http://localhost:9074/predict   # end to end, timing
```

## Log

| date (UTC) | setup | acc | tIoU | score | notes |
|---|---|---|---|---|---|
| 2026-09-17 | baseline (always yes, no spans), local 39 | 0.500 | 0.000 | 0.200 | plumbing check |
| 2026-09-17 | CPU smoke: Qwen3-ASR-0.6B + aligner + Qwen3.5-0.8B, 30 s clip over HTTP | - | - | - | every stage runs, response valid; 20 s audio -> 13.7 s ASR on 4 CPU threads |
| 2026-09-17 13:50 | Qwen3-ASR-1.7B + aligner, 180 s chunks, sequential (P100 fp16) | - | - | - | ASR real-time factor 0.185, worst 43.1 s; span oracle: word window 0.892 (0.924 with pads +0.05/-0.15), whole lines 0.793 |
| 2026-09-17 14:05 | same, 30 s chunks batched | - | - | - | ASR real-time factor 0.071, worst 15.6 s at 232 s audio; word-window oracle unchanged 0.892 |
| 2026-09-17 14:25 | + Qwen3.5-9B Q4_K_M, JSON grammar, line range + quote, 29 of 39 conversations | 0.960 | 0.595 | 0.741 | line_quote spans, threshold 0.3; positive 0.968, hard_neg 0.944, off_topic 0.971; LLM 33-45 s per conversation (grammar: 12 tok/s) |
| 2026-09-17 14:32 | grammar off, assistant turn prefilled with the JSON opening | - | - | - | identical greedy output, 32 tok/s: LLM 13-16 s per conversation |
| 2026-09-17 14:35 | deployed on :9054 (GPU 1), pad_end -0.15 | - | - | - | public end-to-end: 232 s conversation answered in 30.8 s, 106 s one in 22.6 s |

Findings: 15% of annotated spans get zero overlap because the fact is said twice and the annotation marks the other
mention; overlapping spans are mostly too long (quotes are whole sentences, median end 0.18 s late).
| 2026-09-17 21:19 | validation attempt, Qwen3.5-9B + 1 worked example | - | - | **0.736** | 19 conversations, no errors, 7.5 min |
| 2026-09-17 21:42 | + span refinement pass (2nd LLM call over numbered words) | 0.976 | 0.578 | 0.737 | worse than one pass (0.636): the model marks a keyword where the annotation marks a clause. Kept behind MED_LLM_REFINE=1 |
| 2026-09-17 22:10 | 2 worked examples instead of 1 | 0.972 | 0.616 | 0.759 | worse than 1 example (0.641/0.774 on the same 36) |
| 2026-09-17 22:30 | Gemma-4-12B-it-qat Q4 (same prompt) | 0.986 | 0.640 | 0.778 | hard negatives 0.992; 19.8 s per conversation |
| 2026-09-17 22:40 | Gemma-4-26B-A4B-it-qat Q4 (MoE, 4B active) | **0.992** | 0.637 | **0.779** | fastest: 11.8 s mean, 16.4 worst; needs 15.4 GB, a whole P100 |
| 2026-09-17 22:43 | deployed Gemma-4-12B on :9054, n_ctx 4096 | - | - | - | 15.9 GB of 16 GB with the ASR models on the same GPU; 232 s conversation in 43.9 s |

Findings: tIoU sits at 0.64 for every model tried (9B to 26B) while accuracy reaches 0.99, so the remaining gap is
the annotation style of the evidence, not model strength. Deterministic span rules (sentence expansion, including a
preceding question line, pad sweeps) do not beat taking the quote's line start to the quote end.
Max prompt+generation seen: 3090 tokens, so n_ctx 4096 is enough.
| 2026-09-18 00:05 | local_evaluator against the live server (Gemma-4-12B, 39 conversations) | 0.985 | 0.627 | **0.770** | 0 failed, 0 timeouts; 27.9 s mean, 38.9 s worst (65% of the budget) |
| 2026-09-18 00:15 | span ranker (gradient boosting on 195 annotated spans, grouped CV) | - | 0.604 | - | worse than the quote spans (0.645); best candidate in the set 0.811 |
| 2026-09-18 00:40 | prompt rule "take the later statement that settles it" | 0.986 | 0.617 | 0.765 | worse than without (0.640); reverted |

Four attempts to lift tIoU all lost to simply mapping the LLM's quote onto the words (line start -> quote end,
pad_end -0.15): a second refinement pass, a second worked example, a learned span ranker, and a later-mention rule.
Accuracy is at 0.985-0.992 for every model, so the score is now bounded by how the evidence is cut, and the
annotators' choice among repeated mentions looks partly arbitrary from the transcript alone.

serve.sh now picks the model from free GPU memory: GPU 0 free (>= 15.5 GB) serves Gemma-4-26B-A4B there with
speech recognition on GPU 1; otherwise Gemma-4-12B shares GPU 1. MED_LLM overrides both.

## 2026-09-18 span diagnosis and three dead ends

Decomposition of the tIoU loss on `eval_20260918T001239` (36 conversations, 178 annotated yes):

| | mean tIoU |
|---|---|
| shipped (quote mapped onto words) | 0.617 |
| same passage, boundaries trimmed perfectly | 0.767 (+0.150) |
| full word-window oracle | 0.890 (+0.123 more) |

So 55% of the remaining loss is boundary placement and 45% is picking the wrong mention of a repeated fact
(27 of 178 positives score exactly zero; only 3 of those because we answered no). Accuracy is finished: 0.986
means 5 errors in 360, and perfect accuracy is worth +0.006 score. `yes_threshold` is dead — the score is flat
from 0.15 to 0.5 because P(yes) is saturated.

Timings are not the bottleneck: gold starts sit 0.04 s from the nearest word start and gold ends 0.18 s before
the nearest word end, which is what `pad_end = -0.15` already compensates. 74% of gold starts coincide with a
line start, 60% of gold ends with a line end. A better aligner buys nothing.

The LLM's quote is the right *length* (9 words median vs 8 gold) and matches the transcript almost verbatim
(median similarity 1.02, 2% below 0.6 ratio) — it is simply positioned differently from the annotation. Fixing
the start perfectly is worth +0.115 and the end +0.136, so there is no one-sided "stop earlier" instruction.

| date (UTC) | setup | acc | tIoU | score | notes |
|---|---|---|---|---|---|
| 2026-09-18 | deterministic span rules, 10 variants swept on saved generations | - | 0.552-0.623 | - | `quote` 0.623 best, `line_quote` 0.617, whole-line 0.552; nothing clears the noise floor |
| 2026-09-18 | DeBERTa-v3-large-squad2 extractive QA, best of 4 span mappings | - | **0.350** | - | dead: 33% zero-overlap, 13% no-answer; returns the minimal answer where the annotation marks a clause |
| 2026-09-18 | DeBERTa-v3-large-mnli mention re-selection over 1-3 line windows | - | **0.274** | - | dead: fixes 6 of the 27 zero-overlap cases, breaks 31 that were right; also 2.15 s per question |
| 2026-09-18 | MBR medoid over 3 model opinions (9B + 12B + 26B) | 0.989 | 0.663 | **0.793** | best measured, but see the noise floor below |
| 2026-09-18 | same, union / intersect / centroid | 0.989 | 0.591 / 0.658 / 0.607 | 0.750 / 0.790 / 0.760 | union and centroid are worse than any single model, as the metric predicts |

**The noise floor.** `eval_20260917T221143` and `eval_20260918T001239` are both Gemma-12B with one example and
differ only in `llm_ctx` 8192/`llm_batch` 1024 vs 4096/512 — performance-only knobs, and max prompt+generation is
3042 tokens so 4096 is functionally sufficient. They change llama.cpp's reduction order, so 17 of 36 greedy
generations came out different and the score moved 0.013 (0.778 vs 0.765) for no semantic reason.

Bootstrapping over conversations puts every idea tested inside that floor:

```
medoid - gemma12b     +0.015  95% CI [-0.004, +0.035]  P(>0)=0.94
medoid - gemma26b     +0.014  95% CI [-0.001, +0.030]  P(>0)=0.96
arbitrary-config gap  +0.014  95% CI [-0.002, +0.032]
```

The medoid gain is also not deployable: it comes from model-family diversity (9B+12B+26B = 0.793, ~29 GB and 3x
generation), not from resampling — medoid over two runs of the same model is 0.778, exactly the single-model
score, and the two-model 12B+26B medoid is 0.775, below 26B alone.

Consequence for method: 36 conversations cannot resolve an effect below about 0.03 score, and a 19-conversation
validation attempt is noisier still. Only the boundary headroom (+0.150) is large enough to measure here.

## 2026-09-18 the few-shot rebuild is a null

The boundary headroom (+0.150 tIoU) is the only lever big enough to clear the noise floor, and the worked example
is the only channel through which the annotators' convention reaches the model. Two changes were tested against it,
both behind default-off flags (`MED_FEWSHOT_CONV`, `MED_QUOTE_HINT`), all four variants scored on the same 37
conversations (sample_17 and sample_57 both excluded, since each is a prompt example in some variant):

| variant | example | hint | acc | tIoU | score | delta vs A (95% CI) |
|---|---|---|---|---|---|---|
| A | sample_17 (3 worked quotes) | no | 0.984 | 0.635 | 0.775 | baseline |
| B | sample_57 (7 worked quotes, corpus-matched lengths) | no | 0.981 | 0.633 | 0.773 | -0.002 [-0.020, +0.018] |
| C | sample_17 | yes | 0.984 | 0.637 | 0.776 | +0.001 [-0.016, +0.018] |
| D | sample_57 | yes | 0.984 | 0.641 | 0.778 | +0.003 [-0.017, +0.025] |

sample_57 was picked as the conversation whose annotations best match the corpus (median 8 words / 2.50 s against
8 / 2.88 overall) and it carries 7 worked quotes instead of 3, including two different sub-spans of one line. None
of it matters: every interval straddles zero, P(>0) is 0.41 / 0.55 / 0.61.

The instruction did change behaviour, which is what makes this a useful null rather than a non-event. Mean predicted
span length against a gold mean of 3.23 s:

```
A  3.20 s   zero-overlap 22        C  2.85 s   zero-overlap 26
B  3.52 s   zero-overlap 24        D  3.31 s   zero-overlap 27
```

The model obeys "5 to 12 words" and shortens its quote, and the zero-overlap count goes up rather than down. The
spans were already the right width; trimming them only moves them further from the annotation. This is the same
conclusion the earlier diagnosis reached from the other direction — the quote is the right length and in the wrong
place — and it says the remaining gap is not reachable by instructing the model.

Cost, which rules B and D out even at parity: the richer example takes max prompt+generation from 3025 to 3512
tokens against `n_ctx` 4096. That is 584 tokens of headroom on an evaluation set whose conversations may run longer
than the 232 s maximum in training.

**Shipped: variant A, unchanged.** Six ideas have now been measured against the boundary gap (refinement pass,
second example, learned ranker, later-mention rule, extractive QA, NLI re-selection) plus these three, and none has
cleared the +-0.014 noise floor.

## Operational

`watchdog.sh` + `systemd/medical-watchdog.{service,timer}` restart the endpoint if it stops answering: a check every
30 s, a restart after 2 consecutive misses (60 s, longer than any single request may take, so a busy server is never
killed), and a `maintenance` flag file in the storage directory to stand it down for experiments. Enabled, so it
survives a reboot.

`KillMode=process` in the service unit is load-bearing, not a nicety. `serve.sh` backgrounds the server and returns
as soon as the port answers; with the default `KillMode=control-group` systemd then tears down the oneshot's cgroup
and takes the server with it, seconds after it came up. The result is a restart loop that never keeps a server
alive, which is worse than having no watchdog at all. A `systemd-run --user` probe does *not* reproduce this - the
user manager cleans up differently - so the only test that settles it is killing the real server and checking the
process is still there a minute later.

Measured end to end: killed at 11:42:58, missed at 11:43:14 and 11:43:49, restarted, answering again at 11:44:14 and
still on the same pid 90 s later. **76 s from death to service**, so a mid-attempt crash costs one or two
conversations rather than the whole tail; five consecutive timeouts (300 s) is what ends an attempt.

Latency, fitted over 184 real requests (audio 74-232 s, replies 19.5-43.8 s), measured with the other project's
training job saturating GPU 0:

```
reply = 19.6 s + 0.061 x audio_seconds     ->  400 s audio = 43.9 s, 500 s audio = 50.0 s
```

The ~20 s constant is the LLM (ten questions regardless of length); the slope is speech recognition. `budget_s = 50`
truncates generation before the 60 s limit, so an over-long conversation degrades rather than timing out.

`MED_SPAN_FOR_NO` resolved: if the service scores like `local_evaluator.py` it is worth +0.003 score on the 26B and
0.000 on the deployed 12B, whose missed positives carry no usable span. Not a lever either way; left at the README's
reading.

## 2026-09-18 more worked quotes: the example's format matters more than its quantity

Only one worked example fits the prompt, so the 195 annotated spans reach the model through 3 of them. The obvious
fix is more examples, and the obvious obstacle is `n_ctx` 4096. Measured on the longest training conversation
(sample_79, 222 s), with 400 tokens reserved for the generation:

```
1 full example (shipped)     2576 + 400 = 2976    headroom +1120     3 worked quotes
2 full examples              3788 + 400 = 4188    OVERFLOW by  92   14 worked quotes
2 condensed examples         3273 + 400 = 3673    headroom  +423    14 worked quotes
3 condensed examples         4087 + 400 = 4487    OVERFLOW by 391   21 worked quotes
```

`build_example(..., condense=n)` keeps only the lines within n of an annotated span and renumbers from L0, which
roughly halves an example. That buys a second example; a third does not fit at any condensation.

A dose-response on the number of worked quotes, paired over the same 35 conversations:

| variant | examples | quotes | acc | tIoU | score | delta vs A (95% CI) | P(>0) | mean span |
|---|---|---|---|---|---|---|---|---|
| A | 1 full | 3 | 0.986 | 0.642 | 0.779 | baseline | - | 3.17 s |
| G | 1 condensed | 7 | 0.983 | 0.609 | 0.758 | -0.021 [-0.049, +0.007] | 0.07 | 3.91 s |
| E | 2 condensed | 14 | 0.983 | 0.616 | 0.763 | -0.016 [-0.040, +0.006] | 0.08 | 3.61 s |

Gold mean span over the same questions is 3.13 s. No truncation anywhere: all three finished on `stop` with every
question parsed, and E peaked at 3749 of 4096 tokens, so the regressions are real rather than an artifact.

This is the largest effect measured so far and it points the wrong way, which makes it the most useful result yet.
**Condensing the example inflates the quotes.** Strip a transcript to the lines around its annotated spans and
those spans become a much larger share of the visible text, so the model infers that a quote covers most of what it
can see: 3.17 s -> 3.91 s against a gold 3.13 s, and the zero-overlap count rises with it.

More quotes does help, but only within the condensed family, and only by undoing part of the damage: G to E adds
seven quotes, pulls the mean span back from 3.91 to 3.61 s and recovers +0.005. The hypothesis is not wrong; it is
that the format penalty is bigger than the supervision gain, and the budget offers no way to add quotes without
paying it. 2 full examples miss by 92 tokens, and `n_ctx` cannot grow because the server already holds 15.9 GB of a
16 GB card with speech recognition on the same GPU.

**Shipped: A, unchanged.** Eleven ideas measured against the boundary gap; the only ones outside the noise floor
are the two that made it worse.
| 2026-09-19 12:35 | deployed Gemma-4-26B-A4B (GPU 1) with speech recognition moved to GPU 0 | - | - | - | drone training freed GPU 0; 232 s conversation now 27.9 s (was 38.7 s with the 12B), spot checks 10/10 correct on two conversations |
