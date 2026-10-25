# Chars vs bytes as BPE base symbols

Related docs: [RESEARCH.md](RESEARCH.md#chars-vs-bytes) (short summary), [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md#bytes-to-unicode) (trainer write-up), [CLAUDE.md](../CLAUDE.md) (conventions).

The Hugging Face course starts BPE from the characters in the corpus. This project starts from the 256 byte values: `tokenize_bpe` splits each chunk with `.bytes()`, not `.chars()`, and maps each byte to a symbol through the fixed `bytes_to_unicode` table. All 256 symbols are always in the base vocabulary.

## The code change

```rust
// Before: the base vocab is the distinct chars found in the corpus
let mut alphabet: Vec<char> = Vec::new();
for word in word_freqs.keys() {
    for letter in word.chars() {
        if !alphabet.contains(&letter) {
            alphabet.push(letter);
        }
    }
}
let mut vocab: Vec<String> = vec!["<|endoftext|>".to_string()];
vocab.extend(alphabet.iter().map(std::string::ToString::to_string));

let mut splits: HashMap<&str, Vec<String>> = HashMap::new();
for word in word_freqs.keys() {
    let mut vec_of_chars: Vec<String> = Vec::new();
    for char in word.chars() {
        vec_of_chars.push(char.to_string());
    }
    splits.insert(word, vec_of_chars);
}
```

```rust
// After: the base vocab is all 256 byte symbols, whatever the corpus holds
let byte_symbols = bytes_to_unicode();
let mut vocab: Vec<String> = vec!["<|endoftext|>".to_string()];
vocab.extend(byte_symbols.iter().map(std::string::ToString::to_string));

let mut splits: HashMap<&str, Vec<String>> = HashMap::new();
for word in word_freqs.keys() {
    let vec_of_chars: Vec<String> = word
        .bytes()
        .map(|b| byte_symbols[usize::from(b)].to_string())
        .collect();
    splits.insert(word, vec_of_chars);
}
```

Same chunks, different starting symbols:

| Text     | chars                | bytes              |
| -------- | -------------------- | ------------------ |
| `" the"` | space, `t`, `h`, `e` | `Ġ`, `t`, `h`, `e` |
| `é`      | `é`                  | `Ã`, `©`           |
| `😀`     | `😀`                 | `ð`, `Ł`, `ĺ`, `Ģ` |

## What it means

Every input is bytes, so it always splits into known symbols and needs no unknown token. A character common in the corpus, like `é`, merges into one token during training. A rare one stays several byte tokens, so scripts rare in the corpus cost more tokens.

Tokens are byte sequences, not text. One can end mid-character, so the decoder converts ids to bytes first and holds back an incomplete character until its remaining bytes arrive.

## Why a mapping table

The table maps each byte to one `char`. Decimal text per byte would make merges ambiguous: `"1"` + `"16"` = `"116"`, also the symbol for byte 116. 188 bytes map to themselves, the other 68 to `U+0100` and up, so space becomes `Ġ` and newline `Ċ`. See [Bytes to unicode](DEVELOPMENT_NOTES.md#bytes-to-unicode).

```rust
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut shifted = 0;
    for b in 0..=u8::MAX {
        table[usize::from(b)] = if matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF) {
            char::from(b)
        } else {
            let c = char::from_u32(256 + shifted).expect("below the surrogate range");
            shifted += 1;
            c
        };
    }
    table
}
```

## Order of operations

The pre-tokenizer regex runs on the original text, the table afterwards. The 68 shifted symbols are letters, so mapping first would make spaces letters and the regex would stop splitting words.

## Determinism

The fixed table also makes training deterministic. The old alphabet came from a `HashMap`, whose iteration order is random, so two runs could differ.
