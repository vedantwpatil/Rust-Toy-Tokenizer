# Development Notes

Related docs: [CLAUDE.md](../CLAUDE.md) (conventions, learning context), [ROADMAP.md](ROADMAP.md) (stages, progress, open decisions), [ARCHITECTURE.md](ARCHITECTURE.md) (tokenizer pipeline overview), [RESEARCH.md](RESEARCH.md) (tokenizer research notes), [RUST_NOTES.md](RUST_NOTES.md) (Rust mechanics met along the way).

Technical write-ups of what has been built, per stage, with reasoning and open questions. Library: `crates/tokenizer` (modules `baseline`, `byte_level`, `pretokenize`, `train`, `encode`, `tokenizer`, `metrics`, tests in `src/tests.rs`). Demo binary: `crates/tokenizer-cli`. Before 2026-10-24 everything lived in `src/main.rs`; the split did not change the behavior described below.

## Stage 1: byte-level BPE trainer

Status: trainer done, encoder and decoder not started. Progress and open decisions: [ROADMAP.md](ROADMAP.md#stage-1-tokenizer-byte-level-bpe).

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
- A merge's rank is its index in `merges`, the order merges were learned. The encoder replays them in that order, so the order is part of the trained artifact. Hence `Vec`, not `HashMap`: training never looks a merge up, and a map would lose the order. The encoder builds `pair -> rank` and `token -> id` maps from the two vectors at load.
- `vocab[257 + i] == format!("{a}{b}")` for `merges[i] == (a, b)`, guarded by `vocab_and_merges_stay_aligned`.
- Starting size is 257, so `TOY_VOCAB_SIZE = 280` learns 23 merges.

### Pre-tokenizer

`GPT2_PATTERN` is GPT-2's split regex: contractions, optional-space-prefixed letter, digit and punctuation runs, then whitespace. The trailing `\s+(?!\S)` is a negative lookahead, which the `regex` crate cannot express, hence `fancy-regex`. `pre_tokenize<'a>(re: &Regex, text: &'a str) -> Vec<&'a str>` returns slices of `text`. The explicit `'a` is needed because two reference parameters defeat lifetime elision, see [RUST_NOTES.md](RUST_NOTES.md#lifetimes).

Consequences for training:

- A leading space stays attached to its word (`" the"`), so spaces are part of learned tokens. Hence `Ġt`.
- Pairs are counted with `windows(2)` inside one chunk, so no merge spans two chunks.

### Bytes to unicode

`bytes_to_unicode() -> [char; 256]` is GPT-2's table, indexed by byte.

- 188 bytes print cleanly and map to themselves: `0x21..=0x7E` (`!` to `~`), `0xA1..=0xAC`, `0xAE..=0xFF`.
- The other 68 (control characters, space, `0x7F..=0xA0`, `0xAD`) get `U+0100`, `U+0101`, ... in byte order, ending at `U+0143`: `0x00` is `Ā` (`U+0100`), newline `Ċ` (`U+010A`), space `Ġ` (`U+0120`), `0xAD` `Ń` (`U+0143`).
- The `shifted` counter advancing in byte order makes the table identical to GPT-2's. Any other order is a valid bijection but would not match GPT-2's `vocab.json` and `merges.txt`.

Why this representation:

- **One byte, one `char`.** Closed 256-symbol alphabet: any input representable, no unknown token. Starting from `char`s (about 1.1 million values) makes the base alphabet corpus-dependent and open-ended.
- **Not decimal strings.** Decimal text per byte makes merges ambiguous: `"1"` + `"16"` = `"116"`, also the symbol for byte 116 (`t`). One char per byte keeps concatenation unambiguous.
- **Printable, no whitespace.** Tokens store and print as plain text. `byte_table_is_distinct_and_printable` checks this and that all 256 chars are distinct, which makes the table reversible for decoding.
- **Pre-tokenize first, map second.** All 68 shifted symbols are Latin Extended-A letters (`\p{L}`). Mapping before the regex turns spaces into letters, `\s` never matches, and the text collapses into one letter run. So the mapping runs afterwards, per byte of each chunk.
- **Always the full alphabet.** All 256 symbols enter `vocab` whatever the corpus. The first 257 entries are identical for every training text, and a byte the corpus never showed still encodes.

The fixed table also means the base alphabet no longer comes from iterating a `HashMap`, so it is deterministic.

Multi-byte characters become several symbols: `é` (two UTF-8 bytes) starts as `Ã` and `©`, an emoji as four symbols. A merged token is a byte sequence and can end mid-character, hence the decoder's hold-back buffer (see [ROADMAP.md](ROADMAP.md#stage-1-tokenizer-byte-level-bpe)).

### Training loop

State: `word_freqs` (fixed), `splits` (rewritten each merge), `vocab`, `merges`. Each iteration:

1. **Count.** `compute_pair_freqs` adds each chunk's frequency to every adjacent pair of its current symbols. A pair's count is its occurrences in the whole corpus, not the number of chunks containing it.
2. **Stop check.** Empty `pair_freqs` means every chunk is one symbol: break. Without it the loop picks nothing and never grows `vocab`.
3. **Select.** One pass keeps the highest count, ties to the smallest `(String, String)`.
4. **Merge.** `merge_pairs` rewrites every chunk's symbols in place.
5. **Record.** Push the pair onto `merges`, the concatenation onto `vocab`.

`while vocab.len() < vocab_size` stops at the target size.

**Selection is order-independent.** `HashMap` has a per-instance random seed, so iteration order differs between maps and between runs, and "first pair seen with the max count" picks different merges per run. Max under `(count, smaller pair)` is a total order, so the winner does not depend on iteration order. `training_is_deterministic` trains twice and compares. HF's Python course is deterministic only because dicts keep insertion order, so its tie-break differs: on the toy corpus `("e", "r")`, `("i", "s")` and `("Ġ", "a")` all occur 5 times. This repo takes them in pair byte order (`er`, `is`, `Ġa`), the course takes `is` before `er`. Only the first merge, `("Ġ", "t")` with 7 occurrences (` the`, ` tokenization`, ` tokenizer`, ` to`, ` they`, ` trained`, ` tokens`), has no tie.

**Merge semantics.** `merge_pairs` scans each chunk left to right. On a match at `i` it writes the concatenation to `split[i]`, removes `split[i + 1]`, and does not advance `i`, so the new symbol is compared with its right neighbour next. Matches never overlap: `["a","a","a"]` under `("a","a")` becomes `["aa","a"]`, four `a`s become `["aa","aa"]`. The encoder must use the same left-to-right greedy order, or trained and encode-time merges disagree.

### Cost

Every iteration recounts all pairs from scratch.

- `compute_pair_freqs` is linear in total symbols across unique chunks, but clones each chunk's `Vec<String>` and both strings of every window, so it is allocation heavy.
- `merge_pairs` is linear in the same total, quadratic in chunk length worst case (`Vec::remove` shifts the tail).
- Total is about `merges × symbols`. Fine for the toy corpus, wrong shape for a real one. Standard fix: incremental pair counts, updating only chunks a merge touched, with a heap for the argmax. Not done: correctness first, per [CLAUDE.md](../CLAUDE.md).
- Pair counts are `u32` and overflow past about 4.29e9 occurrences of one pair.

### Known limitations and open decisions

Resolved items stay in place, marked resolved. Never delete.

- **Resolved (2026-10-22): corpus selection.** `tokenize_bpe(prompt)` trained on the prompt if over 100 bytes, else on four HF course sentences, at the fixed `VOCAB_SIZE`. Corpus and `vocab_size` are now arguments. `main` still applies the 100-byte rule, passing `TOY_CORPUS` or the prompt, with `VOCAB_SIZE` as its default. The `Tokenizer { vocab, merges, pattern }` struct is still to come.
- **Open: id layout differs from GPT-2.** Bytes sit in byte order after `<|endoftext|>`. GPT-2 puts the 188 printable bytes first, then the 68 shifted ones, then merges, then `<|endoftext|>` at 50256. There `Ġ` is 220 and the first merge `Ġt` 256; here `Ġ` is 33. Matching tiktoken needs GPT-2's base order or a remap in the oracle test.
- **Open: tie-break differs from reference trainers.** See above. Affects trained artifacts only, not encoding with published GPT-2 merges.
- **Open: no encoder or decoder.** Merges are learned but never replayed on new text.
- **Open: table verified against a Python port, not in-repo.** A one-time check compared all 256 entries with a Python port of GPT-2's `bytes_to_unicode`. Tests check five known entries plus bijectivity and printability. The tiktoken oracle test will cover the rest.
- **Resolved: nondeterministic merges.** Caused by `HashMap` iteration with a strict `>` and no tie-break, and by an alphabet collected from a `HashMap`. Fixed by the `(count, pair)` total order and the fixed byte table.
- **Resolved: decimal-string byte symbols.** Rejected for the merge ambiguity above, replaced by the GPT-2 table.

### Tests

20 tests in `crates/tokenizer/src/tests.rs`. The first 10 cover the pre-tokenizer (splits checked against tiktoken's pattern), the rest the byte table and trainer. Most trainer tests use `train()`, which trains `TOY_CORPUS` at `TOY_VOCAB_SIZE` via `Tokenizer::train`. The last two pass their own text as a one-element corpus.

| Test                                      | What it guards                                                                                          |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `byte_table_matches_gpt2`                 | space is `Ġ`, newline is `Ċ`, `a` maps to itself, byte 0 is `Ā`, byte 255 is `ÿ`                        |
| `byte_table_is_distinct_and_printable`    | 256 distinct chars, none control or whitespace, so the table is reversible                              |
| `first_merge_is_space_t`                  | first merge is `("Ġ", "t")`, the only pair with count 7, so no tie-break dependence                     |
| `base_alphabet_is_all_256_bytes`          | `vocab[0]` is the special token and `vocab[1 + b]` is the table entry for every byte                    |
| `trains_to_target_vocab_size`             | `vocab.len() == TOY_VOCAB_SIZE` and `merges.len() == TOY_VOCAB_SIZE - 257`                                      |
| `no_empty_tokens`                         | no empty token or empty half of a merge                                                                 |
| `vocab_and_merges_stay_aligned`           | `vocab[257 + i]` is the concatenation of `merges[i]`                                                    |
| `training_is_deterministic`               | two runs return identical `(vocab, merges)`                                                             |
| `non_ascii_corpus_uses_only_byte_symbols` | with `é` and an emoji in the corpus, every char in every token (past the special token) is a table char |
| `stops_when_no_pairs_remain`              | 101 identical letters train down to one token and stop before `TOY_VOCAB_SIZE`                              |
