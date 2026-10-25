# Roadmap

Project build LLM stack from scratch in Rust, bottom-up: tokenizer, then inference pipeline, then coding harness on top. Each stage = layer next one stands on. This doc = plan: what each stage is, what it adds to pipeline, how verified, what still undecided.

Related docs: [CLAUDE.md](../CLAUDE.md) (project conventions, learning context), [RESEARCH.md](RESEARCH.md) (tokenizer research notes), [ARCHITECTURE.md](ARCHITECTURE.md) (how code laid out today).

For adding agents and subagents to the harness read stage in progress plus [Cross-stage constraints](#cross-stage-constraints). Do not pull work forward from later stages unless constraint there asks.

## Status

Update in place as work lands. Never delete rows. Estimates are guesses, see [Time estimates](#time-estimates).

| Stage | What                      | Status                                                                                                                    | Estimate                            |
| ----- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| 1     | Tokenizer: byte-level BPE | In progress (trainer, encoder, decoder, special tokens, GPT-2 loader, tiktoken oracle and metrics done; streaming decode and faster encoder left) | 3-5 weeks                           |
| 2     | GPT-2 small on CPU        | Not started                                                                                                               | 6-10 weeks                          |
| 3     | Llama-style model + int8  | Not started                                                                                                               | 6-8 weeks                           |
| 4     | Inference harness traits  | Not started                                                                                                               | 4-6 weeks                           |
| 5     | Coding harness            | Not started                                                                                                               | 4-8 weeks                           |
| 6     | Stretch: Metal or serving | Undecided, leaning Metal                                                                                                  | Metal 6-10 weeks, serving 4-8 weeks |

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

- Ids match tiktoken's `gpt2` encoding on varied corpus: ASCII, code, emoji, CJK, repeated and trailing whitespace, contractions. _Met 2026-10-24: 39 oracle cases plus two benchmark texts (22,200 and 1,000 tokens) match exactly._
- `decode(encode(s)) == s` for any valid UTF-8, streaming decode never emits half a character. _Round trip met 2026-10-24. Streaming decode not started._
- Trainer reproduces hand-computed merges on toy corpus like one above. _Partly met: `first_merge_is_space_t` checks the first merge only. The `low`/`lower`/`newest` example above is not a test yet._

**Watch for**

- Whitespace handling in regex causes most mismatches against reference.
- At encode time, merges chosen by rank (position in merge list), not frequency.
- Treat vocabulary, merges, pre-tokenizer pattern as inputs, not constants. Stage 3 models ship their own.

**Feeds:** stage 2 (ids in and out), stage 4 (id-to-bytes table for grammar masks, streaming decode), stage 5 (token counting, chat-template special tokens).

**Progress.** Update in place, never delete. Design and mechanics of what exists: [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md#stage-1-byte-level-bpe-trainer).

| Item                                                                                         | State                                                                                                                                                                                                                                                                                                                                                                                 | Date       |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| Pre-tokenizer: GPT-2 regex via `fancy-regex`, `pre_tokenize`                                 | Done, 10 tests with expected splits checked against tiktoken's pattern                                                                                                                                                                                                                                                                                                                |            |
| Bytes to unicode: `bytes_to_unicode`, `[char; 256]` indexed by byte                          | Done, matches GPT-2's reference algorithm on all 256 bytes                                                                                                                                                                                                                                                                                                                            | 2026-10-16 |
| Trainer: pair counts, best-pair selection, merge, ordered `merges`                           | Done, deterministic, stops at `TOY_VOCAB_SIZE` or when no pair is left                                                                                                                                                                                                                                                                                                                    | 2026-10-16 |
| Trainer and byte-table tests                                                                 | Done, 8 trainer tests and 2 byte-table tests (20 total in `crates/tokenizer/src/tests.rs`)                                                                                                                                                                                                                                                                                                             | 2026-10-16 |
| Encoder: replay `merges` by rank per chunk                                                   | Done, `encode_chunk` plus `Tokenizer::encode`                                                                                                                                                                                                                                                                                                                                         | 2026-10-24 |
| Decoder: ids to bytes to UTF-8                                                               | Done for a whole id list, strict `from_utf8` (a cut codepoint is an `Err`)                                                                                                                                                                                                                                                                                                            | 2026-10-24 |
| Decoder: streaming hold-back of an incomplete UTF-8 tail                                     | Not started                                                                                                                                                                                                                                                                                                                                                                           |            |
| Special tokens: `<\|endoftext\|>` cut out before pre-tokenization, never merged              | Done, longest special wins, decode emits its text                                                                                                                                                                                                                                                                                                                                     | 2026-10-24 |
| Split train from encode: `Tokenizer { vocab, merges, pattern, specials }`                    | Done, `Tokenizer::train` and `Tokenizer::from_gpt2_merges` build one                                                                                                                                                                                                                                                                                                                  | 2026-10-24 |
| Load GPT-2 as data                                                                           | Done, `crates/tokenizer/data/gpt2/merges.txt` (OpenAI's `vocab.bpe`, sha256 matches tiktoken's pin). No `vocab.json`, ids follow from merges                                                                                                                                                                                                                                           | 2026-10-24 |
| Oracle test against tiktoken `gpt2`                                                          | Done, 39 cases (ASCII, code, emoji, CJK, whitespace, contractions, specials) from `crates/tokenizer/scripts/gen_oracle.py`, ids match exactly                                                                                                                                                                                                                                                          | 2026-10-24 |
| Metrics: compression ratio, token counts                                                     | Done in `main`: tokens and bytes per token for word, char, byte, trained BPE, GPT-2 BPE. Word row has unbounded vocab, not comparable. Rows state vocab size and training data: 280 entries from four sentences vs GPT-2's 50,257 is a data and vocab gap, not an algorithm gap                                     | 2026-10-24 |
| Speed benchmarks: `bench_encode_mixed_text`, `bench_encode_one_long_chunk` in `crates/tokenizer/src/tests.rs` | Done, `#[ignore]`d: `cargo test --release bench_ -- --ignored --nocapture`. Each checks token count and id hash against tiktoken. **Baseline** (`String`-based `encode_chunk`): 60 KB mixed text 10.4 ms (5.75 MB/s, 2.1M tokens/s); one 4000-byte chunk 137.9 ms (0.03 MB/s), ~200x slower per byte, cost grows with chunk length squared | 2026-10-24 |
| Faster `encode_chunk` (plan below)                                                           | Not started                                                                                                                                                                                                                                                                                                                                                                           |            |
| Tests overall                                                                                | 46 passing plus the 2 ignored benchmarks, clippy clean                                                                                                                                                                                                                                                                                                                                | 2026-10-24 |
| Cargo workspace: `crates/tokenizer` (lib) and `crates/tokenizer-cli` (bin) | Done, no behavior change: same 45 tests, CLI output, benchmark hashes and timings. Later crates: [Crate map](#crate-map). `Tokenizer.vocab` and `.merges` public for stage 4's id-to-bytes table. _Correction 2026-10-24:_ `vocab` holds GPT-2 symbol chars (`Ġthe`), not raw bytes, so not that table, see **Id-to-bytes table** below | 2026-10-24 |

**Open decisions in stage 1**

- **Id layout.** _Resolved 2026-10-24:_ id = index into `Tokenizer.vocab`, each constructor picks its layout. `from_gpt2_merges` builds GPT-2's (`Ġ` 220, `<|endoftext|>` 50256), so the oracle compares ids with no remap. Trained tokenizers keep the layout below. Original note: `vocab[0]` is `<|endoftext|>`, `vocab[1..=256]` the bytes in byte order, `vocab[257 + i]` merge `i`. GPT-2 puts the 188 printable bytes first, then the 68 shifted bytes, then merges, then `<|endoftext|>` at 50256 (`Ġ` is 220 there, 33 here). Matching tiktoken needs GPT-2's base order or a remap in the oracle test. Undecided.
- **Tie-break.** Equal-count pairs resolve to the smallest `(String, String)`. Deterministic, but not the HF course's or other reference trainers' rule, so merges after the first tie can differ. Matters only when comparing trained artifacts, not when encoding with GPT-2's merges.
- **Encoder speed.** Three steps, each benchmarked against the baseline above before the next. (1) Ids, not `String`s: `byte_ids` table and `(left id, right id) -> (rank, merged id)` map built once in `Tokenizer::new`, no per-merge allocation or string hashing. (2) Incremental ranks: rank per adjacent pair, recompute only the two pairs a merge touches. (3) Heap plus linked list, staleness check on pop, O(n log n). The long-chunk case needs step 3: `Vec::remove` and the per-round rescan keep steps 1 and 2 quadratic. Per-chunk caching (GPT-2's trick) needs interior mutability, belongs with stage 4 concurrency. Set a target number after step 1. Oracle tests and benchmark hashes guard correctness. The two direct `encode_chunk` unit tests need porting when its signature changes.
- **Special tokens in input.** Open. Specials in text always become their id. tiktoken refuses them unless `allowed_special` allows them, which matters once untrusted text reaches the tokenizer (stage 5, chat templates). Decide by then whether `encode` takes an allow-list.
- **Count type.** Pair counts are `u32`. Overflows past about 4.29e9 occurrences of one pair. Irrelevant at toy scale, revisit before training on a real corpus.
- **Id-to-bytes table.** Open, needed by streaming decode and stage 4. `Tokenizer.vocab` holds GPT-2 symbol chars, and converting them needs `char_to_byte` (`pub(crate)`). Needs a public way to get each id's raw bytes. `inference` takes the table as data for grammar masks and stop strings ([Crate map](#crate-map)).
- **Id type.** Open, decide before stage 2. Ids are `usize`. `u32` is what tiktoken and HF `tokenizers` use: half the size, fits a 128K vocab. Ids cross every crate boundary ([Crate map](#crate-map)), so a late change edits every crate. Option: `TokenId(u32)` newtype with a private field. Hides the representation (a change edits one crate) and makes swapped `prefill(ids, pos)` arguments a compile error. Conflicts with "no helper types until a tuple gets unwieldy", so weigh it at stage 2.

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
- `Backend` and `CpuBackend` start as an `inference` module using no model code, so they can become the `backend` crate at stage 6A. `Model<B: Backend>` generic: static dispatch in the hot path. Settle the id type first. See [Crate map](#crate-map).

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
- **`StopCondition`:** end of sequence, max tokens, stop strings. Stop strings can span tokens: match on a rolling buffer of recent token bytes. Needs only the id-to-bytes table, immune to characters cut between tokens.

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
- Design question to settle here: generics versus `dyn` for traits. Per-token virtual call negligible next to forward pass, so favour whichever reads simpler. Leaning: both, by role. Generics for `Backend` (`Model<B: Backend>`, hot path). `dyn` for processors: `Vec<Box<dyn LogitProcessor>>` chains different types. Keep `LogitProcessor` object-safe (no generic methods, no `self` by value); losing that breaks every `dyn` user.
- Where stage 4 pieces live (all in `inference`, or `Session` there and the rest in an `engine` crate). _Resolved 2026-10-24:_ all in `inference`, tokenizer passed in as data ([Crate map](#crate-map)). Raise this stage's [concerns](#concerns-to-raise-during-development) first: tokenizer code vs data (`todo!()` sketch), sampling as a separate call, `Session` return types and `Send`, continuous batching.

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
- **Chat template and special tokens** for local model. Format differs per model. Templates live in `tokenizer` (HF ships them in `tokenizer_config.json` with the specials).
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
- **First step:** move `Backend`, tensor type and `CpuBackend` into a `backend` crate, so `metal` never depends on model code ([Crate map](#crate-map)).
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
- **Crates:** the engine loop = batching scheduler, lives in `inference` (needs KV cache internals). `server` = axum and SSE around it ([Crate map](#crate-map)).
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

## Crate map

Workspace under `crates/`, split by side, not stage. Text side (`tokenizer`) and ids side (`inference`) meet only in crates using both. Model side (`inference`) and hardware side (`backend`, `metal`) meet only at the `Backend` trait. These boundaries sit where data representation changes, so they survive plan changes.

- New crate when dependencies, platform or artifact kind change, and only once a second consumer or implementation exists. Until then a module; promoting later is a `git mv`.
- Every boundary is a public API to maintain. Return types are the real commitment: arguments can loosen later (`Vec<Vec<u8>>` to `impl IntoIterator<Item = impl AsRef<[u8]>>`) without breaking callers, return types cannot.
- `inference` never depends on `tokenizer`. Stage 4 needs only tokenizer data (id-to-bytes table, end-of-sequence id), passed in at construction.

| Crate | Stage | Holds | Why a separate crate |
| --- | --- | --- | --- |
| `tokenizer` (exists) | 1, 3, 5 | Text side: trainer, encoder, decoder, specials, GPT-2 loader, baselines. Later: id-to-bytes table, streaming decoder, Llama and Qwen vocab loading, chat templates | Foundation. Only `fancy-regex`, plus `serde_json` for `tokenizer.json` |
| `tokenizer-cli` (exists) | 1 | Demo and comparison report | Binary, display only |
| `inference` | 2, 3, 4, 6B | Ids side: safetensors loader, GPT-2 and Llama forward passes, KV cache, int8, `Session`, `Sampler`, `LogitProcessor`, `StopCondition` (byte-matched stop strings), grammar masks, generate loop (ids in, ids out), later the batching scheduler. `Backend` and `CpuBackend` as a module until `backend` splits | No tokenizer dependency. Ids in and out is also what tok/s benchmarks measure |
| `backend` | 6A | `Backend` trait, tensor type, `CpuBackend`, split from `inference` when `metal` starts | `inference` and `metal` depend on it, not on each other: model edits do not rebuild `metal`, Metal tests compare against `CpuBackend` without model code |
| `harness` | 5 | `Model` trait (local, hosted API), context builder, tool-call parsing, tools, agent loop | Uses `tokenizer` and `inference`. Keeps HTTP and TLS out of the inference path |
| `metal` | 6A | `MetalBackend` | `unsafe`, Metal bindings, shaders, macOS only. Depends on `backend` only |
| `server` | 6B | axum handler, request channel, SSE | Uses `tokenizer` and `inference`. Async runtime. Batching scheduler stays in `inference` (needs KV internals) |
| Grammar (unnamed) | 4, only if it grows | Grammar compiler, token-mask engine | Needs neither model nor tokenizer, like llguidance's separate [`toktrie`](https://docs.rs/toktrie) |
| Python bindings (unnamed) | any | PyO3 wrapper for benchmarking (see README) | `cdylib` and pyo3 |

```text
            backend   (Backend trait, tensor type, CpuBackend)
             ▲     ▲
             │     │
     inference     metal   (stage 6, option A)
             ▲
             │
tokenizer-cli · harness · server   (encode, run inference, streaming decode)
             │
             ▼
         tokenizer
```

Arrows point at the dependency. `inference` and `tokenizer` never point at each other.

Stage 2 and 3 demos and benchmarks go in `inference/examples/` with `tokenizer` as a dev-dependency: library stays tokenizer-free, bacon's `ex` job runs them.

Not planned: `engine` crate (shared text layer between `inference` and consumers). Create only if text wiring repeated in `harness` and `server` grows past a few calls. Streaming decode and chat templates live in `tokenizer`, so it should not.

**Decision: where stage 4 splits.** *Resolved 2026-10-24: all in `inference`, tokenizer passed in as data (option F).* Stage 4 needs the tokenizer's data, not its code. Grammar masks come from token bytes (llguidance builds masks from a trie over vocab bytes). Stop strings match on bytes. End-of-sequence id comes from model config. Only prompt encoding and display decoding need tokenizer code, done by the consumer, as in [vLLM V1](https://docs.vllm.ai/en/latest/design/arch_overview/): engine core returns ids, API server process tokenizes and detokenizes.

- **A, B, C:** original note below.
- **D:** A, but tokenizer-free (plain data, not `&Tokenizer`). Grew into F.
- **E:** split by dependency. Tokenizer-free pieces in `inference`; grammar masks, stop strings, generate loop in `engine`. Unneeded once those needed only data.
- **F (chosen):** ids-only core, plus `backend` crate, text wiring in `tokenizer`, splits only on a second consumer.
- **Rejected:** Cargo feature gating `tokenizer` (features are additive across the build, `metal` cannot rely on it off). `trait Vocab` in `inference` (orphan rule blocks `engine` implementing it for `tokenizer::Tokenizer`; implementing it in `tokenizer` reverses the dependency). Separate sampling crate (`Sampler` has no dependencies).

Rust's crate rules (compilation unit, coherence boundary, public API) rank options, don't pick. C: widest API. A: tokenizer edits rebuild `inference` and `metal`. B, E: `engine` before a second consumer. The data-not-code finding is a domain fact; if wrong, the choice changes. See [Concerns to raise](#concerns-to-raise-during-development).

Original note: Stage 4 adds `Session`, `Sampler`, `LogitProcessor`, `StopCondition`, grammar masks. `Session` owns the KV cache and borrows weights, so it is tied to model internals. `Sampler` only turns logits into an id. Grammar masks need the id-to-bytes table and stop strings need decoded text, so both need the tokenizer. Question: does `inference` stay tokenizer-free?

- **A: all in `inference`.** One crate fewer, but `inference` (and `metal`, through it) depends on `tokenizer`, and sampling compiles with the model.
- **B: `Session` in `inference`, rest in `engine`.** `inference` stays ids in, logits out; `engine` is where tokenizer and model meet.
- **C: all in `engine`, `Session` included.** `inference` would have to make KV cache layout and forward pass public. Wider API, no gain.

What would change it: if `Sampler`, `LogitProcessor`, `StopCondition` work on plain slices (`&[f32]`, `&[u32]`) and `&Tokenizer`, B is cheap. If `Session` keeps needing `pub(crate)` model internals, it stays in `inference`, ruling out C. If every consumer needs both crates anyway, A is simpler.

### Concerns to raise during development

Where the crate map rests on a prediction. Raise each when its stage starts, before building on it. Resolve in place, never delete.

| Stage | Concern | If it goes the other way |
| --- | --- | --- |
| 1, before 2 | Id type (stage 1 open decisions) | Ids are in every crate's API; a late change edits every crate |
| 1 | Id-to-bytes table missing (stage 1 open decisions) | Streaming decode, grammar masks, stop strings wait on it |
| 4, at start | Does any stage 4 piece need tokenizer code, not just data? Sketch structs and signatures with `todo!()` bodies first; `cargo check` shows a missing dependency before logic exists | F's premise breaks; revisit B or E |
| 4 | Sampling stays a separate call after the forward pass, not inside a `Session` step | Folding it in blocks the stage 6A overlap: [XGrammar](https://arxiv.org/abs/2411.15100) builds the mask on CPU during the GPU forward pass, waiting only at sampling |
| 4 | `Session` return types are the commitment. Logits as borrowed `&[f32]`; caller can `.to_vec()` | Owned-to-borrowed later breaks every caller |
| 4 | `Session` stays `Send`: compile-only `fn assert_send<T: Send>() {}` test | A private `Rc` or `RefCell` drops `Send` silently; `server` breaks at stage 6 |
| 4, 6B | Does `Session`'s API survive continuous batching? Stepping many sessions in one forward pass needs KV internals | Scheduler goes in `inference`. If `Session` cannot express batching, rework before stage 5 builds on it |
| 4 | Grammar code size | Own crate if large |
| 5, 6B | Text wiring repeated in `harness` and `server` | `engine` if past a few calls |
| 6A, at start | `backend` split | Move `Backend`, tensor type, `CpuBackend` out of `inference` before `metal`, so `metal` never depends on model code |

---

## Cross-stage constraints

| Constraint                                                                                                                                       | Set in  | Pays off in       | Why                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ------- | ----------------- | -------------------------------------------------------------------- |
| Tensor ops (matmul, RMSNorm, RoPE, softmax, attention) behind `Backend` trait. CPU path stays correctness reference (logits within ~1e-4 of HF). | Stage 2 | Stage 6 (Metal)   | GPU backend must be swappable and checkable against CPU one.         |
| `Session` supports many concurrent sessions from start.                                                                                          | Stage 4 | Stage 6 (serving) | Cheap now, rewrite later.                                            |
| Tokenizer takes vocabulary, merges, pre-tokenizer pattern as data, not constants.                                                                | Stage 1 | Stage 3           | Llama 3 and Qwen ship own.                                           |
| Tokenizer exposes id-to-bytes table and streaming-safe decode.                                                                                   | Stage 1 | Stages 4, 5, 6    | Grammar masks, stop strings, SSE streaming all work on partial text. |
| Reference tokenizers and models = test oracles only, never in code path under test.                                                              | All     | All               | Keeps from-scratch rule honest.                                      |
| `inference` never depends on `tokenizer`; takes id-to-bytes table and end-of-sequence id as data. | Stage 2 | Stages 4, 6A | Tokenizer edits never rebuild model or `metal`. |
| Sampling is a separate call after the forward pass. | Stage 4 | Stage 6A | CPU builds grammar mask during GPU forward pass. |
| Id type settled before ids cross crates. | Stage 1 | All | Ids are in every crate's API. |

## Time estimates

Guesses, not data. Assume 10-15 hours a week while still learning Rust; real number likely 1.5-2x higher.

- Stages 1-5: about 6-9 months (stage 1: 3-5 weeks, stage 2: 6-10, stage 3: 6-8, stage 4: 4-6, stage 5: 4-8).
- Metal: 6-10 weeks. Serving: 4-8 weeks, plus 3-4 for paged KV. Both: about 3-4.5 months.
- Everything: about 10-14 months, roughly August to December 2027.
