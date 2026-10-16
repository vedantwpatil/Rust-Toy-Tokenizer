use super::*;

// Expected splits were checked against tiktoken's gpt2 pattern, run through
// the python `regex` module.

// `text` is the only borrowed input, so elision ties the output to it. The
// Regex is a temporary that dies here, which is fine because pre_tokenize's
// output only borrows from `text`.
fn split(text: &str) -> Vec<&str> {
    let re = Regex::new(GPT2_PATTERN).unwrap();
    pre_tokenize(&re, text)
}

#[test]
fn words_and_punctuation() {
    assert_eq!(split("Hello, world!"), ["Hello", ",", " world", "!"]);
}

#[test]
fn leading_space_stays_with_the_word() {
    assert_eq!(
        split("This is the hugging face course"),
        ["This", " is", " the", " hugging", " face", " course"]
    );
    assert_eq!(split(" hello"), [" hello"]);
}

#[test]
fn contractions_split_off() {
    assert_eq!(split("I've"), ["I", "'ve"]);
    assert_eq!(split("don't"), ["don", "'t"]);
}

#[test]
fn uppercase_contractions_are_not_special() {
    // The pattern is case sensitive, so "'VE" falls through to the
    // punctuation and letter alternatives.
    assert_eq!(split("I'VE"), ["I", "'", "VE"]);
}

#[test]
fn digits_split_from_letters() {
    assert_eq!(split("abc123"), ["abc", "123"]);
    assert_eq!(split(" 42"), [" 42"]);
}

#[test]
fn punctuation_runs_group_together() {
    assert_eq!(split("Hi!!! ok"), ["Hi", "!!!", " ok"]);
}

// `\s+(?!\S)` takes a whitespace run minus its last char when a non-space
// follows, so the last space can attach to the next word.
#[test]
fn whitespace_runs() {
    assert_eq!(split("a  b"), ["a", " ", " b"]);
    assert_eq!(split("a   b"), ["a", "  ", " b"]);
    assert_eq!(split("hi "), ["hi", " "]);
    assert_eq!(split("a\nb"), ["a", "\n", "b"]);
    assert_eq!(split("  \t\n  x  "), ["  \t\n ", " x", "  "]);
}

#[test]
fn non_ascii_text() {
    assert_eq!(split("caf\u{e9}"), ["caf\u{e9}"]);
    assert_eq!(split("hi \u{1F600}"), ["hi", " \u{1F600}"]);
    assert_eq!(split("\u{4f60}\u{597d}"), ["\u{4f60}\u{597d}"]);
}

#[test]
fn empty_input_has_no_chunks() {
    assert!(split("").is_empty());
}

// Every alternative together covers all characters, so the chunks must
// join back into the exact input.
#[test]
fn chunks_join_back_to_input() {
    let samples = [
        "",
        "Hello, world!",
        "a  b",
        "  \t\n  x  ",
        "I've seen 42 things... don't you?",
        "caf\u{e9} \u{1F600} \u{4f60}\u{597d}",
        "This is the hugging face course",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ];
    for s in samples {
        assert_eq!(split(s).concat(), s);
    }
}

// A prompt of 100 bytes or less makes tokenize_bpe train on the toy corpus.
fn train() -> (Vec<String>, Vec<(String, String)>) {
    tokenize_bpe("hi")
}

// 256 byte symbols plus <|endoftext|>.
const START_VOCAB: usize = 257;

#[test]
fn byte_table_matches_gpt2() {
    let table = bytes_to_unicode();
    assert_eq!(table[usize::from(b' ')], '\u{120}'); // 'Ġ'
    assert_eq!(table[usize::from(b'\n')], '\u{10a}'); // 'Ċ'
    assert_eq!(table[usize::from(b'a')], 'a');
    assert_eq!(table[0], '\u{100}'); // 'Ā', first shifted byte
    assert_eq!(table[255], '\u{ff}'); // 'ÿ' maps to itself
}

// Distinct chars make the table reversible, and printable ones mean a token
// can be shown and stored as ordinary text.
#[test]
fn byte_table_is_distinct_and_printable() {
    let table = bytes_to_unicode();
    let distinct: std::collections::HashSet<char> = table.iter().copied().collect();
    assert_eq!(distinct.len(), 256);
    assert!(table.iter().all(|c| !c.is_control() && !c.is_whitespace()));
}

// ("Ġ", "t") is a space followed by t. It occurs 7 times and no other pair
// ties it, so this holds whatever the tie-break rule is.
#[test]
fn first_merge_is_space_t() {
    let (_, merges) = train();
    assert_eq!(merges[0], ("\u{120}".to_string(), "t".to_string()));
}

// The base alphabet is all 256 bytes in byte order, even those the corpus
// never uses, so the front of the vocab does not depend on the training text.
#[test]
fn base_alphabet_is_all_256_bytes() {
    let (vocab, _) = train();
    assert_eq!(vocab[0], "<|endoftext|>");
    for (b, c) in bytes_to_unicode().iter().enumerate() {
        assert_eq!(vocab[1 + b], c.to_string());
    }
}

#[test]
fn trains_to_target_vocab_size() {
    let (vocab, merges) = train();
    assert_eq!(vocab.len(), VOCAB_SIZE);
    assert_eq!(merges.len(), VOCAB_SIZE - START_VOCAB);
}

#[test]
fn no_empty_tokens() {
    let (vocab, merges) = train();
    assert!(vocab.iter().all(|t| !t.is_empty()));
    assert!(merges.iter().all(|(a, b)| !a.is_empty() && !b.is_empty()));
}

// vocab[START_VOCAB + i] is the token that merges[i] produced.
#[test]
fn vocab_and_merges_stay_aligned() {
    let (vocab, merges) = train();
    for (i, (a, b)) in merges.iter().enumerate() {
        assert_eq!(vocab[START_VOCAB + i], format!("{a}{b}"));
    }
}

// Each HashMap gets its own random hash seed, so iteration order differs
// between calls. Ties between equal-count pairs must be broken by the pair
// itself, not by which one the loop saw first.
#[test]
fn training_is_deterministic() {
    assert_eq!(train(), train());
}

// "é" is two UTF-8 bytes and the emoji is four. Every token must still be
// built only from byte symbols, since the pre-tokenizer cuts on characters
// but training works on bytes.
#[test]
fn non_ascii_corpus_uses_only_byte_symbols() {
    let text = "caf\u{e9} \u{1F600} ".repeat(10);
    assert!(text.len() > 100);
    let (vocab, _) = tokenize_bpe(&text);
    let table = bytes_to_unicode();
    assert!(vocab[1..].iter().all(|t| t.chars().all(|c| table.contains(&c))));
}

// One chunk of 101 identical letters is a single word. Training must stop
// once it is one token, before reaching the vocab target.
#[test]
fn stops_when_no_pairs_remain() {
    let text = "a".repeat(101);
    let (vocab, merges) = tokenize_bpe(&text);
    assert!(vocab.len() < VOCAB_SIZE);
    assert_eq!(vocab.len(), START_VOCAB + merges.len());
    assert!(vocab.iter().all(|t| !t.is_empty()));
    assert!(vocab.contains(&text));
}
