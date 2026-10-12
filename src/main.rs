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
    // println!("Tokenize with BPE: {bpe}\n");
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

    println!("Corpus: {corpus:?}\n");

    // Pre tokenize
    let mut word_freqs: HashMap<&str, i32> = HashMap::new();

    let re = Regex::new(GPT2_PATTERN).unwrap();

    // pre_tokenize step?
    for text in corpus {
        let tokenize = pre_tokenize(&re, text);
        // Next step is the compute the frequency of each word in the tokenized corpus
        for word in tokenize {
            *word_freqs.entry(word).or_insert(0) += 1;
        }
    }

    // Sorting for printing
    descending_sort(&word_freqs);

    let mut alphabet: Vec<char> = Vec::new();
    for word in word_freqs.keys() {
        for letter in word.chars() {
            if !alphabet.contains(&letter) {
                alphabet.push(letter);
            }
        }
    }

    alphabet.sort_unstable();
    println!("Alphabet: {alphabet:?}\n");

    let mut vocab: Vec<String> = vec!["<|endoftext|>".to_string()];
    vocab.extend(alphabet.iter().map(std::string::ToString::to_string));
    println!("Extended Alphabet: {vocab:?}\n");

    // Split alphabet character to character to be able to start training
    // We split alphabet and not vocab as special characters are not standardized
    let mut splits: HashMap<&str, Vec<String>> = HashMap::new();
    for word in word_freqs.keys() {
        let mut vec_of_chars: Vec<String> = Vec::new();
        for char in word.chars() {
            vec_of_chars.push(char.to_string());
        }
        splits.insert(word, vec_of_chars);
    }
    println!("Word[char] Dictionary: {splits:?}\n");

    let pair_freqs = compute_pair_freqs(splits, &word_freqs);

    // Sorted for printing
    descending_pair_sort(&pair_freqs);
    todo!()
}

fn compute_pair_freqs(
    splits: HashMap<&str, Vec<String>>,
    word_freqs: &HashMap<&str, i32>,
) -> HashMap<(String, String), i32> {
    let mut pair_freqs: HashMap<(String, String), i32> = HashMap::new();

    for (word, freq) in word_freqs {
        let split = splits[word].clone();
        for window in split.windows(2) {
            let pair = (window[0].clone(), window[1].clone());
            *pair_freqs.entry(pair).or_insert(0) += freq;
        }
    }
    pair_freqs
}

// Regex match pattern
const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

fn pre_tokenize<'a>(re: &Regex, text: &'a str) -> Vec<&'a str> {
    re.find_iter(text).map(|m| m.unwrap().as_str()).collect()
}

fn descending_sort(word_freqs: &HashMap<&str, i32>) {
    println!("Word Frequency Dictionary:");
    let mut sorted: Vec<_> = word_freqs.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    for (word, count) in sorted {
        println!("Count: {count:>3}, \tWord: {word:?}");
    }
    println!();
}

fn descending_pair_sort(pair_freqs: &HashMap<(String, String), i32>) {
    println!("BPE Dictionary:");
    let mut sorted: Vec<_> = pair_freqs.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    for (pair, count) in sorted {
        println!("Count: {count:>3}, \tPair: {pair:?}");
    }
    println!();
}

#[cfg(test)]
mod tests;
