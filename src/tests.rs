use super::*;
use std::time::{Duration, Instant};

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

// Trains on the toy corpus at the default vocab size.
fn train() -> (Vec<String>, Vec<(String, String)>) {
    let re = Regex::new(GPT2_PATTERN).unwrap();
    tokenize_bpe(&TOY_CORPUS, VOCAB_SIZE, &re)
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
    let re = Regex::new(GPT2_PATTERN).unwrap();
    let (vocab, _) = tokenize_bpe(&[text.as_str()], VOCAB_SIZE, &re);
    let table = bytes_to_unicode();
    assert!(
        vocab[1..]
            .iter()
            .all(|t| t.chars().all(|c| table.contains(&c)))
    );
}

// One chunk of 101 identical letters is a single word. Training must stop
// once it is one token, before reaching the vocab target.
#[test]
fn stops_when_no_pairs_remain() {
    let text = "a".repeat(101);
    let re = Regex::new(GPT2_PATTERN).unwrap();
    let (vocab, merges) = tokenize_bpe(&[text.as_str()], VOCAB_SIZE, &re);
    assert!(vocab.len() < VOCAB_SIZE);
    assert_eq!(vocab.len(), START_VOCAB + merges.len());
    assert!(vocab.iter().all(|t| !t.is_empty()));
    assert!(vocab.contains(&text));
}

// Decode in miniature: looking every id up in vocab and joining the tokens
// must give back the byte-symbol form of the input. Pre-tokenizer chunks
// cover the whole text, so this holds for the text as a whole.
#[test]
fn encode_ids_join_back_to_symbols() {
    let tok = toy_tokenizer();
    for text in ["the tokenizer", " caf\u{e9} \u{1F600}", "x", ""] {
        let ids = tok.encode(text);
        let joined: String = ids.iter().map(|&id| tok.vocab[id].as_str()).collect();
        assert_eq!(joined, to_symbols(&tok.byte_symbols, text).concat());
    }
}

fn ranks_from(merges: &[(&str, &str)]) -> HashMap<(String, String), usize> {
    let owned: Vec<(String, String)> = merges
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    build_ranks(&owned)
}

// "lower" has (l,o) rank 0, (w,e) rank 1 and, once lo exists, (lo,w) rank 2.
// Lowest rank first gives lo, we, r. Merging the leftmost pair that has any
// rank would take (lo,w) before (w,e) and give low, e, r.
#[test]
fn encode_lowest_rank_wins_over_leftmost() {
    let ranks = ranks_from(&[("l", "o"), ("w", "e"), ("lo", "w")]);
    let pieces = encode_chunk("lower", &ranks, &bytes_to_unicode());
    assert_eq!(pieces, ["lo", "we", "r"]);
}

// Equal ranks tie to the leftmost pair: aaa is (aa)(a), not (a)(aa).
#[test]
fn encode_repeated_pair_merges_leftmost_first() {
    let ranks = ranks_from(&[("a", "a")]);
    let pieces = encode_chunk("aaa", &ranks, &bytes_to_unicode());
    assert_eq!(pieces, ["aa", "a"]);
}

// Merge 0 of the toy corpus is (Ġ, t), so " t" is a single token. Joining the
// symbols back would pass with no merges at all, this does not.
#[test]
fn encode_applies_trained_merges() {
    let tok = toy_tokenizer();
    assert_eq!(tok.encode(" t"), [tok.vocab_ids["\u{120}t"]]);
}

// Two merges can spell the same token, e.g. (ab,c) and (a,bc). vocab_ids
// would then keep only the later id. Train until no pair is left to see it.
#[test]
fn vocab_has_no_duplicate_tokens() {
    let re = Regex::new(GPT2_PATTERN).unwrap();
    let (vocab, _) = tokenize_bpe(&TOY_CORPUS, 2000, &re);
    assert_eq!(build_vocab_ids(&vocab).len(), vocab.len());
}

// Decode takes the inverse of the byte table, so every symbol char must map
// back to the byte that produced it.
#[test]
fn char_to_byte_inverts_the_table() {
    let table = bytes_to_unicode();
    let inverse = build_char_to_byte(&table);
    assert_eq!(inverse.len(), 256);
    for b in 0..=u8::MAX {
        assert_eq!(inverse[&table[usize::from(b)]], b);
    }
}

fn toy_tokenizer() -> Tokenizer {
    Tokenizer::train(&TOY_CORPUS, VOCAB_SIZE)
}

#[test]
fn decode_of_no_ids_is_empty() {
    assert_eq!(toy_tokenizer().decode(&[]).unwrap(), "");
}

// Byte token ids are 1 + the byte (id 0 is <|endoftext|>) and merged tokens
// come after them, so these ids are " t" followed by "o".
#[test]
fn decode_mixes_merged_and_byte_tokens() {
    let tok = toy_tokenizer();
    let ids = [tok.vocab_ids["\u{120}t"], 1 + usize::from(b'o')];
    assert_eq!(tok.decode(&ids).unwrap(), " to");
}

// <|endoftext|> is a special, so id 0 decodes to its own text.
#[test]
fn decode_id_zero_is_the_literal_endoftext() {
    assert_eq!(toy_tokenizer().decode(&[0]).unwrap(), "<|endoftext|>");
}

// "é" is the bytes C3 A9. Neither byte is valid UTF-8 alone, so decode must
// collect bytes across tokens and only then build the string. Half a
// codepoint is reported as an error, not patched over.
#[test]
fn decode_rejects_a_cut_codepoint() {
    let tok = toy_tokenizer();
    let ids = [1 + 0xC3, 1 + 0xA9];
    assert!(tok.decode(&ids[..1]).is_err());
    assert_eq!(tok.decode(&ids).unwrap(), "\u{e9}");
}

// Trains on `corpus`, then encodes and decodes `text`. Returns the ids too,
// so tests can look at how the text was split.
fn round_trip(corpus: &[&str], vocab_size: usize, text: &str) -> (Vec<usize>, String) {
    let tok = Tokenizer::train(corpus, vocab_size);
    let ids = tok.encode(text);
    let decoded = tok.decode(&ids).unwrap();
    (ids, decoded)
}

fn toy_round_trip(text: &str) -> String {
    round_trip(&TOY_CORPUS, VOCAB_SIZE, text).1
}

#[test]
fn round_trip_ascii() {
    for text in [
        "x",
        "the tokenizer",
        "This is the hugging face course",
        "Hello, world!",
        "I've been to 3 places",
    ] {
        assert_eq!(toy_round_trip(text), text);
    }
}

#[test]
fn round_trip_whitespace() {
    for text in [
        " ",
        "   ",
        "a  b",
        "  two  spaces",
        "trailing ",
        "line\nbreak",
        "\n\n\n",
        "\r\n",
        "tab\there",
    ] {
        assert_eq!(toy_round_trip(text), text);
    }
}

#[test]
fn round_trip_empty_string() {
    assert_eq!(toy_round_trip(""), "");
}

// Accents, an emoji and CJK (\u{65e5}\u{672c}\u{8a9e}) that the ASCII toy
// corpus never learns a merge for, so they decode from raw byte tokens.
#[test]
fn round_trip_non_ascii() {
    for text in [
        "caf\u{e9}",
        "\u{1F642}",
        "h\u{e9}llo \u{1F642} w\u{f6}rld",
        "\u{65e5}\u{672c}\u{8a9e}",
    ] {
        assert_eq!(toy_round_trip(text), text);
    }
}

// Trained until no pair is left, every corpus chunk is one long merged token.
#[test]
fn round_trip_with_long_merged_tokens() {
    for line in TOY_CORPUS {
        let (ids, decoded) = round_trip(&TOY_CORPUS, 2000, line);
        assert_eq!(decoded, line);
        assert_eq!(ids.len(), split(line).len());
    }
}

// main trains on the prompt itself when it is long, so merges can cover the
// bytes of non-ASCII characters.
#[test]
fn round_trip_when_trained_on_the_text() {
    let text = "caf\u{e9} \u{1F642} au lait, na\u{ef}ve r\u{e9}sum\u{e9} \u{1F642}\n".repeat(5);
    let (ids, decoded) = round_trip(&[text.as_str()], VOCAB_SIZE, &text);
    assert_eq!(decoded, text);
    assert!(ids.len() < text.len());
}

// Specials are cut out of the text before pre-tokenization and emitted as one
// id (0 here), whatever surrounds them. 'a' is byte 97, so id 98.
#[test]
fn endoftext_in_text_is_one_special_id() {
    let tok = toy_tokenizer();
    assert_eq!(tok.encode("a<|endoftext|>b"), [98, 0, 99]);
    assert_eq!(tok.encode("<|endoftext|>"), [0]);
    assert_eq!(tok.encode("<|endoftext|><|endoftext|>"), [0, 0]);
}

// A cut-off special is ordinary text.
#[test]
fn partial_special_is_plain_text() {
    let tok = toy_tokenizer();
    let ids = tok.encode("<|endoftext|");
    assert!(!ids.contains(&0));
    assert_eq!(tok.decode(&ids).unwrap(), "<|endoftext|");
}

#[test]
fn round_trip_endoftext() {
    for text in [
        "a<|endoftext|>b",
        "<|endoftext|>",
        "x <|endoftext|>\n<|endoftext|> y",
    ] {
        assert_eq!(toy_round_trip(text), text);
    }
}

// Of two specials starting at the same spot the longer one wins.
#[test]
fn longest_special_wins() {
    let base = toy_tokenizer();
    let mut vocab = base.vocab[..START_VOCAB].to_vec();
    vocab.extend(["<a>".to_string(), "<a>b".to_string()]);
    let specials = vec![("<a>".to_string(), 257), ("<a>b".to_string(), 258)];
    let tok = Tokenizer::new(vocab, Vec::new(), GPT2_PATTERN, specials);
    assert_eq!(tok.encode("<a>b"), [258]);
    assert_eq!(tok.encode("<a>c"), [257, 1 + 99]);
}

fn gpt2() -> Tokenizer {
    Tokenizer::from_gpt2_merges(include_str!("../data/gpt2/merges.txt"))
}

// Expected ids come from tiktoken's gpt2 encoding, see scripts/gen_oracle.py.
const ORACLE: &[(&str, &[usize])] = include!("../data/gpt2/oracle_cases.in");

// Built from merges.txt alone, the ids must land where GPT-2 puts them.
#[test]
fn gpt2_layout_matches_published_ids() {
    let tok = gpt2();
    assert_eq!(tok.vocab.len(), 50257);
    assert_eq!(tok.vocab_ids["!"], 0);
    assert_eq!(tok.vocab_ids["\u{120}"], 220);
    assert_eq!(tok.vocab_ids[ENDOFTEXT], 50256);
}

#[test]
fn gpt2_encode_matches_tiktoken() {
    let tok = gpt2();
    for &(text, ids) in ORACLE {
        assert_eq!(tok.encode(text), ids, "encode mismatch for {text:?}");
    }
}

#[test]
fn gpt2_decode_matches_tiktoken() {
    let tok = gpt2();
    for &(text, ids) in ORACLE {
        assert_eq!(
            tok.decode(ids).unwrap(),
            text,
            "decode mismatch for {text:?}"
        );
    }
}

#[test]
fn bytes_per_token_is_bytes_over_tokens() {
    assert_eq!(bytes_per_token("abcd", 2), 2.0);
    assert_eq!(bytes_per_token("abcd", 4), 1.0);
    assert_eq!(bytes_per_token("", 0), 0.0);
}

// The byte baseline is 1.0 byte per token by definition. Trained BPE beats it
// on its own corpus and GPT-2's 50k merges beat the toy vocab.
#[test]
fn bpe_compresses_better_than_bytes() {
    let text = "This is the hugging face course";
    let trained = toy_tokenizer().encode(text).len();
    let published = gpt2().encode(text).len();
    assert!(trained < text.len());
    assert!(published < trained);
    assert!(bytes_per_token(text, published) > 3.0);
}

// Speed benchmarks. Ignored by default: timings only mean something in a
// release build and are too noisy to pass or fail on. Run with
//   cargo test --release bench_ -- --ignored --nocapture
// Each one also checks the ids against tiktoken's (count and hash), so a faster
// encoder cannot get quicker by getting wrong.
const BENCH_REPEATS: usize = 50;
const BENCH_EXPECTED: [(usize, u64); 2] = include!("../data/gpt2/bench_expected.in");

// FNV-1a over the ids, the same rule as fnv() in scripts/gen_oracle.py.
fn fnv(ids: &[usize]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &id in ids {
        h = (h ^ id as u64).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

// Best of 3, so one noisy run does not set the number.
fn bench_encode(label: &str, tok: &Tokenizer, text: &str, expected: (usize, u64)) {
    let mut best = Duration::MAX;
    let mut ids = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        ids = tok.encode(text);
        best = best.min(start.elapsed());
    }
    assert_eq!(
        (ids.len(), fnv(&ids)),
        expected,
        "{label}: ids differ from tiktoken"
    );
    let secs = best.as_secs_f64();
    println!(
        "{label}: {} bytes -> {} tokens in {:.1} ms ({:.2} MB/s, {:.0} tokens/s)",
        text.len(),
        ids.len(),
        secs * 1000.0,
        text.len() as f64 / secs / 1e6,
        ids.len() as f64 / secs,
    );
}

// Every oracle text back to back, repeated: mostly short chunks, the usual
// shape of real text.
#[test]
#[ignore]
fn bench_encode_mixed_text() {
    let texts: Vec<&str> = ORACLE.iter().map(|&(text, _)| text).collect();
    let text = texts.join("\n").repeat(BENCH_REPEATS);
    bench_encode("mixed text", &gpt2(), &text, BENCH_EXPECTED[0]);
}

// One 4000-byte chunk with no boundary to cut it: the worst case for an encoder
// whose cost grows with the square of the chunk length.
#[test]
#[ignore]
fn bench_encode_one_long_chunk() {
    bench_encode("long chunk", &gpt2(), &"a".repeat(4000), BENCH_EXPECTED[1]);
}
