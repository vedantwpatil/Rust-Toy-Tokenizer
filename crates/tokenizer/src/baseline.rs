// Naive baselines, kept simple so the BPE tokenizer has something to compare
// against. They split into units, not ids, and have no decode.
pub fn tokenize_by_char(prompt: &str) -> Vec<char> {
    prompt.chars().collect()
}

pub fn tokenize_by_word(prompt: &str) -> Vec<String> {
    prompt
        .split_whitespace()
        .map(std::string::ToString::to_string)
        .collect()
}

pub fn tokenize_by_byte(prompt: &str) -> Vec<u8> {
    prompt.bytes().collect()
}
