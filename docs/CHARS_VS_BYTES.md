# Chars vs bytes as BPE base symbols

Related docs: [RESEARCH.md](RESEARCH.md#chars-vs-bytes) (short summary), [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md#bytes-to-unicode) (trainer write-up), [CLAUDE.md](../CLAUDE.md) (conventions).

The Hugging Face course starts BPE from the characters found in the corpus. This project starts from the 256 byte values instead. `tokenize_bpe` splits each chunk with `.bytes()` instead of `.chars()` and maps each byte to a symbol through `bytes_to_unicode`, a fixed table. All 256 symbols are always in the base vocabulary, whatever the corpus contains.

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

The same chunks now start as different symbols:

| Text     | chars                | bytes              |
| -------- | -------------------- | ------------------ |
| `" the"` | space, `t`, `h`, `e` | `Ġ`, `t`, `h`, `e` |
| `é`      | `é`                  | `Ã`, `©`           |
| `😀`     | `😀`                 | `ð`, `Ł`, `ĺ`, `Ģ` |

## What it means

Every input is made of bytes, so every input can be split into known symbols and the tokenizer needs no unknown token. A character that is common in the corpus, such as `é`, is merged into a single token during training. A character that is rare stays as several byte tokens, so text in scripts that are rare in the corpus produces more tokens.

Tokens are byte sequences, not text. A token can end in the middle of a UTF-8 character, so the decoder converts ids to bytes first and holds back an incomplete character until its remaining bytes arrive.

## Why a mapping table

The table maps each byte to one `char`. A simpler choice, using the decimal text of each byte, makes merges ambiguous: `"1"` + `"16"` gives `"116"`, which is also the symbol for byte 116. In the table, 188 bytes map to themselves and the other 68 map to `U+0100` and up, so a space becomes `Ġ` and a newline becomes `Ċ`. See [Bytes to unicode](DEVELOPMENT_NOTES.md#bytes-to-unicode).

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

The pre-tokenizer regex runs on the original text and the table is applied afterward. The 68 shifted symbols are letters, so applying the table first would make spaces count as letters and the regex would stop splitting words.

## Determinism

A fixed table also makes training deterministic. The earlier alphabet was collected from a `HashMap`, whose iteration order is random, so two runs could produce different results.
