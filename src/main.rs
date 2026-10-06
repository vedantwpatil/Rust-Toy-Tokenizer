use fancy_regex::Regex;
use std::{collections::HashMap, env};

fn main() -> Result<(), String> {
    println!("Hello, rust tokenizer!\n");

    // We need to take in input "prompt"
    let prompt = take_input()?;

    // Tokenize the prompt

    // Could have all of the different types of tokenizers here to see how they change for the same
    // prompt

    let word_by_word = tokenize_by_word(&prompt);
    let char_by_char = tokenize_by_char(&prompt);
    let byte_by_byte = tokenize_by_byte(&prompt);
    let bpe = tokenize_bpe(&prompt);

    // DISPLAY
    println!("Prompt: {prompt:?}\n");

    // Should consider running this into a llm or having some visual/demonstration of how each
    // tokenizer works and the performance of it. That might be something to do after the tokenizers
    // themselves as it would first require creating my own LLM to be able to feed a tokenized prompt
    // into which would be it's own project
    //
    // Some ideas on displays with just the tokenizer itself were to figure out the compresion ratio
    // it was able to determine and printing out the merged tokens to see the patterns it learned
    println!("Tokenizing word by word: {word_by_word:?}\n");
    println!("Tokenizing char by char: {char_by_char:?}\n");
    println!("Tokenizing byte by byte: {byte_by_byte:?}\n");
    Ok(())
}

fn take_input() -> Result<String, String> {
    env::args()
        .nth(1)
        .ok_or_else(|| "No prompt provided".to_string())
}

// TOKENIZERS

fn tokenize_by_char(prompt: &str) -> Vec<char> {
    prompt.chars().collect()
}

fn tokenize_by_word(prompt: &str) -> Vec<String> {
    prompt
        .split_whitespace()
        .map(std::string::ToString::to_string)
        .collect()
}

fn tokenize_by_byte(prompt: &str) -> Vec<u8> {
    prompt.bytes().collect()
}

fn tokenize_bpe(prompt: &str) -> Vec<String> {
    let corpus: Vec<&str> = if prompt.len() > 100 {
        vec![prompt]
    } else {
        vec![
            "This is the hugging face course",
            "This chapter is about tokenization",
            "This section shows several tokenizer algorithms",
            "Hopefully, you will be able to understand how they are trained and generate tokens.",
        ]
    };

    // Pre tokenize
    let mut word_freqs: HashMap<&str, i32> = HashMap::new();

    let re =
        Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
            .unwrap();

    // pre_tokenize step?
    for text in corpus {
        let tokenize = pre_tokenize(&re, text);
    }
    todo!()
}

// Regex match pattern
// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

fn pre_tokenize<'a>(re: &Regex, text: &'a str) -> Vec<&'a str> {
    //     re.find_iter(text).map(|m| ).collect()
    let mut pieces = Vec::new();
    for m in re.find_iter(text) {
        // println!("{:?}", m);
        let temp = m.unwrap();
        pieces.push(temp.as_str());
    }
    // println!("{:?}", pieces);
    pieces
}
