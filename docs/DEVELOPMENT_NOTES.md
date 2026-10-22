# Development Notes

Related docs: [CLAUDE.md](../CLAUDE.md) (conventions, learning context), [ROADMAP.md](ROADMAP.md) (stages, progress, open decisions), [ARCHITECTURE.md](ARCHITECTURE.md) (tokenizer pipeline overview), [RESEARCH.md](RESEARCH.md) (tokenizer research notes), [RUST_NOTES.md](RUST_NOTES.md) (Rust mechanics met along the way).

Technical write-ups of what has been built, per stage, with the reasoning and the open questions. Everything lives in `src/main.rs`, tests in `src/tests.rs`.

## Stage 1: byte-level BPE trainer

Status: trainer done, encoder and decoder not started. Progress and open decisions are tracked in [ROADMAP.md](ROADMAP.md#stage-1-tokenizer-byte-level-bpe).

### Data flow

```text
corpus: Vec<&str>
   │  pre_tokenize (GPT-2 regex, runs on the raw text)
   ▼
word_freqs: HashMap<&str, u32>        chunk -> occurrences in corpus
   │  chunk.bytes() -> bytes_to_unicode table
   ▼
splits: HashMap<&str, Vec<String>>    chunk -> its current symbols
   │
   │  loop until vocab.len() == vocab_size or no pair is left
   │    compute_pair_freqs(&splits, &word_freqs) -> HashMap<(String, String), u32>
   │    best pair = max by (count, then smallest pair)
   │    merge_pairs(a, b, &mut splits)            rewrites every chunk in place
   │    merges.push(best_pair); vocab.push(a + b)
   ▼
(vocab: Vec<String>, merges: Vec<(String, String)>)
```

### Output contract

`tokenize_bpe(corpus: &[&str], vocab_size: usize)` returns `(vocab, merges)`.

| Index                               | Content                                                     |
| ----------------------------------- | ----------------------------------------------------------- |
| `vocab[0]`                          | `<\|endoftext\|>`, special token, never produced by a merge |
| `vocab[1 + b]` for `b` in `0..=255` | the symbol for byte `b`, which is `bytes_to_unicode()[b]`   |
| `vocab[257 + i]`                    | `merges[i].0 + merges[i].1`, the token merge `i` created    |

- A token's id is its index in `vocab`.
- A merge's rank is its index in `merges`. Rank order is the order the merges were learned, and the encoder must replay them in that order, so the order is part of the trained artifact. That is why `merges` is a `Vec` and not a `HashMap`: training never looks a merge up, and a map would discard the order and iterate randomly. The encoder will build `pair -> rank` and `token -> id` maps from these two vectors when it loads them.
- `vocab[257 + i] == format!("{a}{b}")` for `merges[i] == (a, b)`. `vocab_and_merges_stay_aligned` guards this.
- Starting size is 257, so with `VOCAB_SIZE = 280` the toy run learns 23 merges.

### Pre-tokenizer

`GPT2_PATTERN` is GPT-2's split regex: contractions, optional-space-prefixed letter runs, digit runs, and punctuation runs, then whitespace. The trailing alternative `\s+(?!\S)` is a negative lookahead, which the `regex` crate cannot express, so `fancy-regex` is used. `pre_tokenize<'a>(re: &Regex, text: &'a str) -> Vec<&'a str>` returns slices of `text`. The explicit `'a` is required because two reference parameters defeat lifetime elision, see [RUST_NOTES.md](RUST_NOTES.md#lifetimes).

Two consequences for training:

- A leading space stays attached to its word (`" the"`), so the space is part of the learned tokens. That is where `Ġt` comes from.
- Pairs are counted with `windows(2)` inside one chunk, so no merge ever spans two chunks.

### Bytes to unicode

`bytes_to_unicode() -> [char; 256]` is GPT-2's table, indexed by byte value.

- 188 bytes already print cleanly and map to themselves: `0x21..=0x7E` (`!` to `~`), `0xA1..=0xAC`, `0xAE..=0xFF`.
- The other 68 bytes are the control characters, space, `0x7F..=0xA0`, and `0xAD`. They are assigned `U+0100`, `U+0101`, ... in increasing byte order, which ends at `U+0143`. So `0x00` is `Ā` (`U+0100`), newline is `Ċ` (`U+010A`), space is `Ġ` (`U+0120`), and `0xAD` is `Ń` (`U+0143`).
- The `shifted` counter advancing in byte order is what makes the table identical to GPT-2's. Any other order is still a valid bijection but would no longer line up with GPT-2's `vocab.json` and `merges.txt`.

Why this representation and not something simpler:

- **One byte, one `char`.** The alphabet is closed at 256 symbols, so any input is representable and there is no unknown token. A `char` has about 1.1 million values, which makes the base alphabet corpus-dependent and open-ended.
- **Not decimal strings.** Mapping each byte to its decimal text makes merges ambiguous. Merging `"1"` and `"16"` yields `"116"`, which is also the symbol for byte 116 (`t`). With one char per byte, concatenation of symbols is unambiguous.
- **Printable and no whitespace.** Every symbol is visible, none is a control character or whitespace, so tokens can be stored and printed as plain text. `byte_table_is_distinct_and_printable` checks this, and also that all 256 chars are distinct, which is what makes the table reversible for decoding.
- **Pre-tokenize first, map second.** All 68 shifted symbols are in Latin Extended-A and are letters (`\p{L}`). If the table were applied before the regex, spaces would become letters, `\s` would never match, and the whole text would collapse into one letter run. The regex needs the real characters, so the mapping happens afterwards, per byte of each chunk.
- **Always the full alphabet.** All 256 symbols enter `vocab` regardless of corpus. The first 257 entries are identical for every training text, and a prompt containing a byte the corpus never showed still encodes.

Because the table is fixed, the base alphabet no longer comes from iterating a `HashMap`, so it is deterministic by construction.

Multi-byte characters become several symbols. `é` is two UTF-8 bytes, so it starts as `Ã` and `©`, and an emoji starts as four symbols. A merged token is a sequence of bytes and can end in the middle of a UTF-8 sequence, which is why the decoder needs a hold-back buffer (see [ROADMAP.md](ROADMAP.md#stage-1-tokenizer-byte-level-bpe)).

### Training loop

State: `word_freqs` (fixed), `splits` (rewritten each merge), `vocab`, `merges`.

Each iteration:

1. **Count.** `compute_pair_freqs` walks `word_freqs`, takes each chunk's current symbols, and adds the chunk's frequency to every adjacent pair. A pair's count is the number of times it occurs in the whole corpus, not the number of chunks that contain it.
2. **Stop check.** An empty `pair_freqs` means every chunk is a single symbol. The loop breaks. Without this guard the loop would pick nothing and never grow `vocab`.
3. **Select.** One pass over `pair_freqs` keeps the pair with the highest count, breaking ties to the smallest `(String, String)`.
4. **Merge.** `merge_pairs` rewrites every chunk's symbol list in place.
5. **Record.** Push the pair onto `merges` and the concatenation onto `vocab`.

`while vocab.len() < vocab_size` ends the loop at the target size.

**Selection is order-independent.** `HashMap` uses a per-instance random seed, so two maps with the same contents iterate in different orders, and so does the same code on two runs. A rule like "first pair seen with the max count" therefore picks different merges between runs. The rule here, maximum under `(count, smaller pair)`, is a total order, so the winner is the same whatever the iteration order. `training_is_deterministic` trains twice and compares. HF's Python course code is deterministic only because Python dicts keep insertion order. For that reason this tie-break differs from the course: on the toy corpus, `("e", "r")`, `("i", "s")` and `("Ġ", "a")` all occur 5 times. This repo takes them in byte order of the pair (`er`, `is`, `Ġa`), where the course takes `is` before `er`. Only the first merge, `("Ġ", "t")` with 7 occurrences (` the`, ` tokenization`, ` tokenizer`, ` to`, ` they`, ` trained`, ` tokens`), has no tie.

**Merge semantics.** `merge_pairs` scans each chunk left to right. On a match at `i` it overwrites `split[i]` with the concatenation and removes `split[i + 1]`, and does not advance `i`, so the new symbol is compared with its right neighbour next. Matches never overlap, and `["a","a","a"]` under `("a","a")` becomes `["aa","a"]`, while four `a`s become `["aa","aa"]`. This matches the left-to-right greedy order the encoder has to use, otherwise trained merges and encode-time merges would disagree.

### Cost

Every iteration recounts all pairs from scratch.

- `compute_pair_freqs` is linear in the total number of symbols across unique chunks. It also clones each chunk's `Vec<String>` and both strings of every window, which is allocation heavy.
- `merge_pairs` is linear in the same total, and quadratic in chunk length in the worst case because `Vec::remove` shifts the tail.
- Total is about `merges × symbols`. Fine for the toy corpus, wrong shape for a real corpus. The standard fix is to keep pair counts incrementally, updating only the chunks touched by a merge, with a heap for the argmax. Not done: correctness first, per [CLAUDE.md](../CLAUDE.md).
- Pair counts are `u32` and overflow past about 4.29e9 occurrences of one pair.

### Known limitations and open decisions

Resolved items stay in place, marked resolved. Never delete.

- **Resolved (2026-10-22): corpus selection.** `tokenize_bpe(prompt)` used to train on the prompt if it was over 100 bytes, otherwise on four fixed sentences from the HF course, with the target size fixed at the `VOCAB_SIZE` constant. The corpus and `vocab_size` are now arguments. `main` still applies the 100-byte rule and passes `TOY_CORPUS` or the prompt, and `VOCAB_SIZE` stays as the default it passes. The `Tokenizer { vocab, merges, pattern }` struct is still to come.
- **Open: id layout differs from GPT-2.** Here the bytes sit in byte order after `<|endoftext|>`. GPT-2 puts the 188 printable bytes first, then the 68 shifted ones, then merges, then `<|endoftext|>` at 50256. In GPT-2 `Ġ` is id 220 and the first merge `Ġt` is id 256. Here `Ġ` is id 33. Matching tiktoken's ids needs either GPT-2's base order or a remap inside the oracle test.
- **Open: tie-break differs from reference trainers.** See above. It affects trained artifacts only, not encoding with published GPT-2 merges.
- **Open: no encoder or decoder.** Merges are learned but never replayed on new text.
- **Open: table verified against a Python port, not in-repo.** The one-time check compared all 256 entries with a Python port of GPT-2's `bytes_to_unicode`. The test suite checks five known entries plus bijectivity and printability. The tiktoken oracle test will cover the rest.
- **Resolved: nondeterministic merges.** Caused by `HashMap` iteration with a strict `>` and no tie-break, and by an alphabet collected from a `HashMap`. Fixed by the `(count, pair)` total order and the fixed byte table.
- **Resolved: decimal-string byte symbols.** Rejected for the merge ambiguity above, replaced by the GPT-2 table.

### Tests

20 tests in `src/tests.rs`. The first 10 cover the pre-tokenizer (splits checked against tiktoken's pattern). The rest cover the byte table and trainer. Most trainer tests go through `train()`, which passes `TOY_CORPUS` and `VOCAB_SIZE` to `tokenize_bpe`. The last two pass their own text as a one-element corpus.

| Test                                      | What it guards                                                                                          |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `byte_table_matches_gpt2`                 | space is `Ġ`, newline is `Ċ`, `a` maps to itself, byte 0 is `Ā`, byte 255 is `ÿ`                        |
| `byte_table_is_distinct_and_printable`    | 256 distinct chars, none control or whitespace, so the table is reversible                              |
| `first_merge_is_space_t`                  | first merge is `("Ġ", "t")`, the only pair with count 7, so no tie-break dependence                     |
| `base_alphabet_is_all_256_bytes`          | `vocab[0]` is the special token and `vocab[1 + b]` is the table entry for every byte                    |
| `trains_to_target_vocab_size`             | `vocab.len() == VOCAB_SIZE` and `merges.len() == VOCAB_SIZE - 257`                                      |
| `no_empty_tokens`                         | no empty token or empty half of a merge                                                                 |
| `vocab_and_merges_stay_aligned`           | `vocab[257 + i]` is the concatenation of `merges[i]`                                                    |
| `training_is_deterministic`               | two runs return identical `(vocab, merges)`                                                             |
| `non_ascii_corpus_uses_only_byte_symbols` | with `é` and an emoji in the corpus, every char in every token (past the special token) is a table char |
| `stops_when_no_pairs_remain`              | 101 identical letters train down to one token and stop before `VOCAB_SIZE`                              |
