# Roadmap

Project build LLM stack from scratch in Rust, bottom-up: tokenizer, then inference pipeline, then coding harness on top. Each stage = layer next one stands on. This doc = plan: what each stage is, what it adds to pipeline, how verified, what still undecided.

Related docs: [CLAUDE.md](../CLAUDE.md) (project conventions, learning context), [RESEARCH.md](RESEARCH.md) (tokenizer research notes), [ARCHITECTURE.md](ARCHITECTURE.md) (how code laid out today).

For agents: read stage in progress plus [Cross-stage constraints](#cross-stage-constraints). Do not pull work forward from later stages unless constraint there asks.

## Status

Update in place as work lands. Never delete rows. Estimates are guesses, see [Time estimates](#time-estimates).

| Stage | What | Status | Estimate |
| --- | --- | --- | --- |
| 1 | Tokenizer: byte-level BPE | In progress (baseline tokenizers and `pre_tokenize` stub in `src/main.rs`) | 3-5 weeks |
| 2 | GPT-2 small on CPU | Not started | 6-10 weeks |
| 3 | Llama-style model + int8 | Not started | 6-8 weeks |
| 4 | Inference harness traits | Not started | 4-6 weeks |
| 5 | Coding harness | Not started | 4-8 weeks |
| 6 | Stretch: Metal or serving | Undecided, leaning Metal | Metal 6-10 weeks, serving 4-8 weeks |

## The stack at a glance

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5  Coding harness     agent loop · tools · context budget                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4  Inference harness  Session · LogitProcessor · Sampler · StopCondition    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3  Llama-style model  RoPE · GQA · RMSNorm · SwiGLU · int8                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2  GPT-2 small, CPU   safetensors · forward pass · KV cache · Backend trait │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1  Tokenizer          byte-level BPE: text <-> token ids                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

Every stage adds one layer. Once stage 4 exists, generating text = loop below. Stage 1 sits at both ends (text in, text out), model sits inside session, traits around it decide what happens between "logits" and "next id".

```text
           prompt text
                │
                ▼
┌───────────────────────────────┐
│ Tokenizer.encode              │
│ (stage 1)                     │
└───────────────┬───────────────┘
                │  token ids
                ▼
┌───────────────────────────────┐
│ Session.prefill / decode      │
│ (stage 4: owns KV cache)      │◀──┐
│ runs Model (stages 2-3)       │   │
└───────────────┬───────────────┘   │
                │  logits           │
                ▼                   │
┌───────────────────────────────┐   │
│ LogitProcessor chain          │   │
│ (stage 4)                     │   │
└───────────────┬───────────────┘   │
                │  logits           │ not stopped: id fed back
                ▼                   │
┌───────────────────────────────┐   │
│ Sampler                       │   │
│ (stage 4)                     │   │
└───────────────┬───────────────┘   │
                │  next id          │
                ▼                   │
┌───────────────────────────────┐   │
│ StopCondition                 │───┘
│ (stage 4)                     │
└───────────────┬───────────────┘
                │  stopped: all ids
                ▼
┌───────────────────────────────┐
│ Tokenizer.decode              │
│ (stage 1, streaming-safe)     │
└───────────────┬───────────────┘
                │
                ▼
         generated text
```

Diagram legend, used in every stage below:

```text
╔═══════╗
║  new  ║   built in this stage
╚═══════╝
┌───────┐
│ older │   built in an earlier stage
└───────┘
```

## Glossary

- **Token / id:** chunk of text and its integer index in vocabulary. Model only sees ids.
- **Logits:** one raw score per vocabulary entry for what next token should be. Picking from them (argmax, sampling) yields next id.
- **Prefill / decode:** prefill runs whole prompt through model in one pass. Decode then produces one token at a time. Prefill compute-bound (matrix times matrix), decode memory-bound (matrix times vector).
- **KV cache:** attention keys and values of every past token, kept per layer so decode does not recompute.
- **Roofline:** speed ceiling set by memory bandwidth. Decode reads every weight once per token, so tokens/sec at most memory bandwidth divided by bytes read per token. Roughly 0.5 GB f32 weights on 100 GB/s machine gives ceiling near 200 tok/s (illustrative numbers, real hardware still to be measured).

## Principles for every stage

- **Verify against reference before moving on.** Tokenizer ids against tiktoken, logits against Hugging Face layer by layer. Reference libraries live in tests or scripts only, never in code path being tested.
- **Correctness first, speed second.** Simple CPU path = reference every faster path checked against.
- **Measure, do not claim.** Report tok/s, share of roofline reached, comparisons against llama.cpp and mlx-lm.

---

## Stage 1: Tokenizer, byte-level BPE

**What it is.** Model cannot read text, only integers. Tokenizer maps text to ids and back. Byte-pair encoding (BPE) learns vocabulary by repeatedly merging most frequent adjacent pair of symbols. "Byte-level" = starting alphabet is 256 byte values, so any input (emoji, any language, arbitrary bytes) representable. No unknown token, round trip lossless.

Tiny training run, each merge becomes new vocabulary entry, its position in list is its rank:

```text
words:               low x5     lower x2     newest x6
start (chars):       l o w      l o w e r    n e w e s t
merge 1  (w,e) x8:   l o w      l o we r     n e we s t
merge 2  (l,o) x7:   lo w       lo we r      n e we s t
```

```text
train (offline)
            ╔═════════╗
  corpus ─▶ ║ Trainer ║ ─▶ vocab.json + merges.txt
            ╚═════════╝

encode
          ╔═══════════╗    ╔═══════════╗    ╔═══════════╗
          ║ Pre-split ║    ║ Bytes to  ║    ║ Merge by  ║
  text ─▶ ║ (regex)   ║ ─▶ ║ unicode   ║ ─▶ ║ rank      ║ ─▶ ids
          ╚═══════════╝    ╚═══════════╝    ╚═══════════╝

decode
         ╔═══════════╗    ╔════════════╗
         ║ Ids to    ║    ║ UTF-8 hold ║
  ids ─▶ ║ bytes     ║ ─▶ ║ buffer     ║ ─▶ text
         ╚═══════════╝    ╚════════════╝
```

**Scope**

- **Pre-tokenizer:** GPT-2 regex splits text into word-like chunks so merges never cross chunk boundaries. Pattern uses lookahead (`\s+(?!\S)`), which `regex` crate does not support, hence `fancy-regex`.
- **Bytes to unicode:** GPT-2 maps each of 256 byte values to visible character so its `vocab.json` and `merges.txt` printable. Needed to load published GPT-2 files.
- **Trainer:** count chunks after pre-splitting, then loop: find most frequent adjacent pair, record merge, update counts. Output = vocabulary plus ordered merge list.
- **Encoder:** per chunk, start from bytes, repeatedly apply lowest-rank merge present until none applies. Naive loop fine first.
- **Special tokens:** `<|endoftext|>` (id 50256) split out before pre-tokenization, never merged. GPT-2 vocabulary = 256 bytes + 50,000 merges + 1 special = 50,257.
- **Decoder:** ids to bytes to UTF-8. Streaming-safe: multi-byte character can straddle two tokens, so hold back incomplete tail until it completes.
- **Metrics:** compression ratio (bytes per token) and token counts against existing char, word, byte baselines.

**Done when**

- Ids match tiktoken's `gpt2` encoding on varied corpus: ASCII, code, emoji, CJK, repeated and trailing whitespace, contractions.
- `decode(encode(s)) == s` for any valid UTF-8, streaming decode never emits half a character.
- Trainer reproduces hand-computed merges on toy corpus like one above.

**Watch for**

- Whitespace handling in regex causes most mismatches against reference.
- At encode time, merges chosen by rank (position in merge list), not frequency.
- Treat vocabulary, merges, pre-tokenizer pattern as inputs, not constants. Stage 3 models ship their own.

**Feeds:** stage 2 (ids in and out), stage 4 (id-to-bytes table for grammar masks, streaming decode), stage 5 (token counting, chat-template special tokens).

---

## Stage 2: GPT-2 small on CPU

**What it is.** First model. Stage 1 turns text into ids. This stage turns ids into logits with hand-written transformer forward pass, then greedy loop turns logits back into text. GPT-2 small: 12 layers, 12 heads, width 768, MLP width 3072, context 1024, vocabulary 50,257, about 124M parameters (roughly 0.5 GB at f32). Uses learned position embeddings, LayerNorm, GELU, ties output projection to token embedding (logits = final hidden state times transposed `wte`).

```text
            prompt text
                 │
                 ▼
┌─────────────────────────────────┐
│ Tokenizer.encode                │
└────────────────┬────────────────┘
                 │  token ids
                 ▼
╔═════════════════════════════════╗
║ GPT-2 forward pass              ║◀──┐
║ + KV cache                      ║   │
╚════════════════╤════════════════╝   │
                 │  logits            │ append id, run again
                 ▼                    │
╔═════════════════════════════════╗   │
║ Greedy: argmax                  ║───┘
╚════════════════╤════════════════╝
                 │  next id
                 ▼
┌─────────────────────────────────┐
│ Tokenizer.decode                │
└────────────────┬────────────────┘
                 │
                 ▼
          generated text
```

Inside forward pass, each of 12 blocks = two residual steps. `(+)` adds block's input back to its output:

```text
        ╔═══════════════════╗             ╔═════════════╗
 x ──┬─▶║ ln_1 -> attention ║──▶ (+) ──┬─▶║ ln_2 -> MLP ║──▶ (+) ──▶ x'
     │  ╚═══════════════════╝     ▲    │  ╚═════════════╝     ▲
     └────────────────────────────┘    └──────────────────────┘
```

**Scope**

- **safetensors loader:** 8-byte little-endian header length, JSON header (tensor name, dtype, shape, byte offsets), then raw tensor bytes. Read or mmap, then slice.
- **Tensor ops behind `Backend` trait from day one:** matmul, LayerNorm, GELU, softmax, causal attention. CPU implementation = correctness reference (see [Cross-stage constraints](#cross-stage-constraints)).
- **Forward pass and greedy loop,** verified layer by layer against Hugging Face.
- **KV cache and prefill/decode split.** Prefill pushes whole prompt through in one matmul-heavy pass. Decode feeds one token, reuses cached keys and values. Per token cache costs 2 x 12 layers x 768 x 4 bytes, about 72 KiB at f32.
- **Speed:** SIMD matmul, rayon across cores, then benchmark and writeup.

**Done when**

- Logits match HF within about 1e-4 at every layer boundary for same ids.
- Greedy output equals HF `generate` with sampling off on test prompts.
- KV-cached decode gives same logits as recomputing everything.
- Benchmark reports prefill and decode tok/s and share of roofline reached.

**Watch for**

- HF stores GPT-2 weights as `Conv1D`, shape `[in, out]`. `nn.Linear` is `[out, in]`. Forgetting transpose = classic first bug.
- `c_attn` fuses Q, K, V into one `[768, 2304]` matrix, so split it.
- Attention scales by 1/sqrt(head_dim), applies causal mask before softmax.
- No separate `lm_head` tensor, it is tied `wte`.
- Decode is matvec reading every weight each token. Will not be compute-bound, that is the point of roofline.

**Needs:** stage 1. **Feeds:** stage 3 (same skeleton, new parts), stage 4 (loop split into traits).

---

## Stage 3: Llama-style model and int8

**What it is.** Modern architecture: Llama 3.2 1B or small Qwen coder model. Skeleton stays (embed, N blocks, final norm, logits) but four parts swapped, weights get quantized. Coder-capable model also what stage 5 needs.

```text
GPT-2 (stage 2)                      Llama-style (stage 3)
┌──────────────────────────────┐     ╔══════════════════════════════╗
│ wte + wpe                    │     ║ token embedding only         ║
│ (learned positions)          │     ║ (position lives in RoPE)     ║
└───────────────┬──────────────┘     ╚═══════════════╤══════════════╝
                │                                    │
                ▼                                    ▼
┌──────────────────────────────┐     ╔══════════════════════════════╗
│ LayerNorm                    │     ║ RMSNorm                      ║
└───────────────┬──────────────┘     ╚═══════════════╤══════════════╝
                │                                    │
                ▼                                    ▼
┌──────────────────────────────┐     ╔══════════════════════════════╗
│ Multi-head attention         │     ║ GQA + RoPE                   ║
│ (12 heads, no RoPE)          │     ║ (few KV heads)               ║
└───────────────┬──────────────┘     ╚═══════════════╤══════════════╝
                │                                    │
                ▼                                    ▼
┌──────────────────────────────┐     ╔══════════════════════════════╗
│ LayerNorm                    │     ║ RMSNorm                      ║
└───────────────┬──────────────┘     ╚═══════════════╤══════════════╝
                │                                    │
                ▼                                    ▼
┌──────────────────────────────┐     ╔══════════════════════════════╗
│ MLP: GELU                    │     ║ SwiGLU MLP                   ║
│ (2 matrices)                 │     ║ (3 matrices, gated)          ║
└──────────────────────────────┘     ╚══════════════════════════════╝
```

- **RMSNorm:** normalizes by root-mean-square only, no mean subtraction, no bias. Cheaper than LayerNorm.
- **RoPE:** rotates query and key vectors by position-dependent angle instead of adding learned position embeddings. Relative position falls out of dot product, context length not tied to learned table.
- **GQA (grouped-query attention):** several query heads share one key/value head. Llama 3.2 1B has 32 query heads and 8 KV heads, so KV cache 4x smaller than full multi-head attention.
- **SwiGLU:** gated MLP, `down(silu(gate(x)) * up(x))`, three matrices instead of two.

Quantization = separate pass over loaded weights. Weights stored as int8 with scales (per channel or per small group), activations stay f32, dequantization happens inside matvec. Decode bandwidth-bound, so 4x fewer bytes than f32 should give close to 4x faster decode.

```text
               ╔════════════╗    ╔═════════════════╗    ╔═════════════════╗
               ║ Quantize   ║    ║ int8 W          ║    ║ matvec: dequant ║
f32 weights ─▶ ║ (offline)  ║ ─▶ ║ + f32 scales    ║ ─▶ ║ inside the loop ║ ─▶ f32 out
               ╚════════════╝    ╚═════════════════╝    ╚═════════════════╝
```

**Scope**

- Loader for new checkpoint format and `config.json` parsing into model config.
- New `Backend` ops: `rmsnorm`, `rope`, SiLU/SwiGLU, GQA-aware attention.
- **Tokenizer generalization:** Llama 3 and Qwen ship own byte-level BPE vocabulary (about 128K entries for Llama 3), pre-tokenizer pattern, special tokens. Stage 1 code must load these as data.
- Weight-only int8 quantization, at load time or via offline converter.

**Done when**

- f32 logits match HF within tolerance, same layer-by-layer method as stage 2.
- Chat prompt gives coherent output.
- int8 versus f32: report top-1 agreement or perplexity change, plus tok/s gain. Quantization lossy, so 1e-4 bar does not apply.

**Watch for**

- RoPE pairing: HF Llama weights assume "rotate half" layout (element i pairs with i + d/2), not interleaved pairs. Mixing gives plausible but wrong text.
- Llama 3.x applies extra RoPE frequency scaling for long context. Read `rope_scaling` in `config.json`.
- GQA needs each KV head broadcast to its group of query heads.
- Read `config.json` for remaining deltas (tied embeddings, biases, norm epsilon) instead of assuming.

**Needs:** stage 2 (loader, `Backend`, KV cache), stage 1 generalized. **Feeds:** stage 4, stage 5 (model worth driving).

---

## Stage 4: Inference harness traits

**What it is.** Until now generation = hard-coded greedy loop. This stage splits it into small replaceable pieces so sampling, constraints, stopping, multiple users compose instead of tangled into forward pass.

```text
            prompt ids
                 │
                 ▼
╔═════════════════════════════════╗
║ Session                         ║
║ owns KV cache + position        ║◀──┐
║ prefill(prompt), decode(id)     ║   │
╚════════════════╤════════════════╝   │
                 │  logits            │
                 ▼                    │
╔═════════════════════════════════╗   │
║ LogitProcessor chain            ║   │
║ mask / penalise / constrain     ║   │
╚════════════════╤════════════════╝   │
                 │  logits            │ not stopped:
                 ▼                    │
╔═════════════════════════════════╗   │
║ Sampler                         ║   │
║ greedy · temperature · top-k/p  ║   │
╚════════════════╤════════════════╝   │
                 │  next id           │
                 ▼                    │
╔═════════════════════════════════╗   │
║ StopCondition                   ║   │
║ EOS · max tokens · stop text    ║───┘
╚════════════════╤════════════════╝
                 │  stopped
                 ▼
           generated ids
```

- **`Session`:** per-conversation state. Owns KV cache and position. `prefill` takes prompt, `decode` takes one id, both return logits.
- **`LogitProcessor`:** edits logits before sampling: repetition penalty, banned tokens, grammar masks.
- **`Sampler`:** picks next id: greedy, temperature, top-k, top-p, with seeded RNG so tests reproducible.
- **`StopCondition`:** end of sequence, max tokens, stop strings. Stop strings need decoded text, can span tokens.

Sessions borrow read-only weights, so many can exist at once. Cheap hook for serving layer later (see [Cross-stage constraints](#cross-stage-constraints)).

```text
╔══════════════════════════╗
║ Session A (KV A)         ║──┐
╚══════════════════════════╝  │   ┌──────────────────────┐
╔══════════════════════════╗  │   │ Model weights        │
║ Session B (KV B)         ║──┼──▶│ (read-only, shared)  │
╚══════════════════════════╝  │   └──────────────────────┘
╔══════════════════════════╗  │
║ Session C (KV C)         ║──┘
╚══════════════════════════╝
```

**One feature on top: grammar-constrained decoding.** `LogitProcessor` that sets logits of every token that would break grammar (for example valid JSON) to negative infinity. Needs stage 1's id-to-bytes table. Hard parts: computing allowed-token mask quickly at every step, and tokens that cross grammar boundaries. In stage 5 this makes tool calls from small local model reliable.

**Done when**

- Stage 2 and 3 greedy output unchanged when run through traits.
- Two interleaved sessions produce same output as same two run separately.
- Constrained output always parses.

**Watch for**

- State belongs to session (KV cache, processor state, RNG), never shared model.
- Design question to settle here: generics versus `dyn` for traits. Per-token virtual call negligible next to forward pass, so favour whichever reads simpler.

**Needs:** stages 1-3. **Feeds:** stage 5, and stage 6 if serving chosen.

---

## Stage 5: Coding harness

**What it is.** Agent loop around model: read task, call tool, see result, repeat until done. Sits behind `Model` trait so same loop can drive local inference (stages 2-4) or hosted API. Starts only after inference pipeline works.

```text
              user task
                  │
                  ▼
╔═══════════════════════════════════╗
║ Context builder                   ║
║ history + tool results,           ║◀──┐
║ trimmed to token budget           ║   │
╚═════════════════╤═════════════════╝   │
                  │  messages           │
                  ▼                     │
╔═══════════════════════════════════╗   │
║ Model trait                       ║   │
║ LocalModel (stages 2-4)           ║   │
║ | HostedModel (API)               ║   │
╚═════════════════╤═════════════════╝   │
                  │  completion         │ result appended
                  ▼                     │
╔═══════════════════════════════════╗   │
║ Parse response                    ║   │
║ tool call or final answer         ║   │
╚═════════════════╤═════════════════╝   │
                  │  tool call          │
                  ▼                     │
╔═══════════════════════════════════╗   │
║ Tool executor                     ║   │
║ read · edit · grep · run          ║───┘
╚═══════════════════════════════════╝
```

**Scope**

- **`Model` trait** with two implementations: local (this project's pipeline) and hosted API.
- **Context builder:** system prompt, history, tool results. Counts tokens with our tokenizer, trims or summarizes oldest tool output when over budget.
- **Chat template and special tokens** for local model. Format differs per model.
- **Tool-call parsing:** structured (JSON) calls from model output, with stage 4's grammar constraint on local models.
- **Tools:** read file, edit file, grep or glob, run command. Writes and commands ask permission.
- **Loop exit:** model answers without tool call, step cap hit, or budget runs out.

**Done when**

- Completes small real task (for example making failing test pass in toy repo) on both model implementations.
- Never overflows context window.
- Malformed tool call reported back to model instead of crashing loop.

**Watch for**

- Token counts must match model actually in use. For hosted API, budget from API's reported usage, since its tokenizer differs from ours.
- Small local models unreliable at tool-call syntax. Constrain output instead of prompting harder.

**Needs:** stages 1 and 4, and working local model. **Feeds:** nothing, top layer.

---

## Stage 6: Stretch decision (open)

Pick one after stage 5. Choice can wait, but it puts two design constraints on earlier stages (see [Cross-stage constraints](#cross-stage-constraints)).

### Option A: Metal backend

`Backend` implementation on Apple GPU: matvec kernel for decode, tiled matmul for prefill, fused attention, int8 dequant inside kernel. Weights mmap'd and shared through unified memory.

```text
┌────────────────────────────────────────────────┐
│ Model forward pass (stages 2-3)                │
└────────────────────────┬───────────────────────┘
                         │  calls
                         ▼
┌────────────────────────────────────────────────┐
│ trait Backend                                  │
│ matmul · rmsnorm · rope · softmax · attention  │
└───────────┬────────────────────────┬───────────┘
            ▼                        ▼
  ┌──────────────────┐     ╔══════════════════╗
  │ CpuBackend       │     ║ MetalBackend     ║
  │ (reference)      │ ◀─▶ ║ (GPU kernels)    ║
  └──────────────────┘     ╚══════════════════╝
    outputs compared op by op in tests
```

- **Hard parts:** kernel correctness against CPU reference, per-op dispatch overhead, unsafe FFI and buffer lifetimes.
- **Results to report:** tok/s CPU vs Metal, share of memory-bandwidth roofline, tok/s vs llama.cpp and mlx-lm.

### Option B: Serving layer

axum handler sends each request over channel to one engine loop that owns model and KV caches. Each loop iteration runs one batched decode step over all active sequences, admits new requests (prefill) between steps, retires finished ones. Tokens stream back over SSE.

```text
           clients (HTTP POST)
                    │
                    ▼
┌───────────────────────────────────────┐
│ axum handler                          │
│ one task per request                  │
└───────────────────┬───────────────────┘
                    │  mpsc channel: prompt + reply handle
                    ▼
╔═══════════════════════════════════════╗
║ engine loop (owns Model + KV caches)  ║
║ 1 admit new requests (prefill)        ║◀──┐
║ 2 one batched decode step over        ║   │ next iteration
║   all active sequences                ║   │
║ 3 retire finished sequences           ║───┘
╚═══════════════════╤═══════════════════╝
                    │  per-request token channel
                    ▼
┌───────────────────────────────────────┐
│ SSE stream                            │
│ tokens back to each client            │
└───────────────────────────────────────┘
```

- **Hard parts:** per-sequence KV management (paged KV = advanced version), prefill/decode scheduling, cancellation on client disconnect, streaming-safe decode (UTF-8 character can split across tokens, already planned in stage 1).
- **Results to report:** aggregate tok/s at 1, 4, 16 concurrent requests, time to first token, p99 inter-token latency, static vs continuous batching.

### Option C (optional third): speculative decoding

Small draft model proposes tokens, big model verifies them in one pass, roughly 1.5-2x faster with identical output. Needs two models sharing tokenizer, sits on top of `Sampler` and `Session`. Check available RAM first.

### Leaning: Metal

- Stated goal = measured work against llama.cpp and mlx-lm. mlx-lm runs on GPU, so CPU-only engine loses that comparison by wide margin.
- Stage 5 = single-user coding harness. Decode speed decides if usable. Batching only helps with many concurrent clients.
- North star = deep-systems Rust: unsafe FFI, manual buffer lifetimes, kernels verified against reference.
- Roofline framing (tok/s ~ memory bandwidth / weight bytes) gives concrete efficiency number.

Serving = more transferable skill (vLLM/TGI-style continuous batching), Metal = more interesting build. Continuous batching can come later as add-on instead of being stretch goal.

---

## Cross-stage constraints

| Constraint | Set in | Pays off in | Why |
| --- | --- | --- | --- |
| Tensor ops (matmul, RMSNorm, RoPE, softmax, attention) behind `Backend` trait. CPU path stays correctness reference (logits within ~1e-4 of HF). | Stage 2 | Stage 6 (Metal) | GPU backend must be swappable and checkable against CPU one. |
| `Session` supports many concurrent sessions from start. | Stage 4 | Stage 6 (serving) | Cheap now, rewrite later. |
| Tokenizer takes vocabulary, merges, pre-tokenizer pattern as data, not constants. | Stage 1 | Stage 3 | Llama 3 and Qwen ship own. |
| Tokenizer exposes id-to-bytes table and streaming-safe decode. | Stage 1 | Stages 4, 5, 6 | Grammar masks, stop strings, SSE streaming all work on partial text. |
| Reference tokenizers and models = test oracles only, never in code path under test. | All | All | Keeps from-scratch rule honest. |

## Time estimates

Guesses, not data. Assume 10-15 hours a week while still learning Rust; real number likely 1.5-2x higher.

- Stages 1-5: about 6-9 months (stage 1: 3-5 weeks, stage 2: 6-10, stage 3: 6-8, stage 4: 4-6, stage 5: 4-8).
- Metal: 6-10 weeks. Serving: 4-8 weeks, plus 3-4 for paged KV. Both: about 3-4.5 months.
- Everything: about 10-14 months, roughly August to December 2027.