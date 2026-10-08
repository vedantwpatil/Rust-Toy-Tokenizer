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
