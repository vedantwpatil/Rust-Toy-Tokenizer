use fancy_regex::Regex;
use std::{cmp::Reverse, collections::HashMap, env};

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
    // TODO: Load a larger corpus
    let corpus: Vec<&str> = if prompt.len() > 100 {
        vec![prompt.as_str()]
    } else {
        TOY_CORPUS.to_vec()
    };
    let (vocab, merges) = tokenize_bpe(&corpus, VOCAB_SIZE);

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
    println!("Tokenize with BPE\nVocab:{vocab:?}\n\nMerges{merges:?}\n\n");

    // ranks and vocab_ids are derived once from merges and vocab, then reused by encode
    let ranks = build_ranks(&merges);
    let vocab_ids = build_vocab_ids(&vocab);
    let byte_symbols = bytes_to_unicode();
    let re = Regex::new(GPT2_PATTERN).unwrap();
    let ids = encode(&prompt, &re, &ranks, &byte_symbols, &vocab_ids);
    let tokens: Vec<&str> = ids.iter().map(|&id| vocab[id].as_str()).collect();
    println!("Encoded ids: {ids:?}\n");
    println!("Encoded tokens: {tokens:?}\n");

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

// 257 base symbols (256 bytes plus <|endoftext|>) leave room for 23 merges.
const VOCAB_SIZE: usize = 280;

const TOY_CORPUS: [&str; 4] = [
    "This is the hugging face course",
    "This chapter is about tokenization",
    "This section shows several tokenizer algorithms",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
];

fn tokenize_bpe(corpus: &[&str], vocab_size: usize) -> (Vec<String>, Vec<(String, String)>) {
    // println!("Corpus: {corpus:?}\n");

    // Pre tokenize
    let mut word_freqs: HashMap<&str, u32> = HashMap::new();

    let re = Regex::new(GPT2_PATTERN).unwrap();

    // pre_tokenize step?
    for &text in corpus {
        let tokenize = pre_tokenize(&re, text);
        // Next step is the compute the frequency of each word in the tokenized corpus
        for word in tokenize {
            *word_freqs.entry(word).or_insert(0) += 1;
        }
    }

    // Sorting for printing
    // descending_sort(&word_freqs);

    // Byte-level: all 256 byte symbols start in the vocab whatever the corpus
    // holds, so no input can contain an unknown symbol.
    let byte_symbols = bytes_to_unicode();

    // alphabet.sort_unstable();
    // println!("Alphabet: {alphabet:?}\n");
    //
    let mut vocab: Vec<String> = vec!["<|endoftext|>".to_string()];
    vocab.extend(byte_symbols.iter().map(std::string::ToString::to_string));
    // println!("Extended Alphabet: {vocab:?}\n");

    // Split alphabet character to character to be able to start training
    // We split alphabet and not vocab as special characters are not standardized
    let mut splits: HashMap<&str, Vec<String>> = HashMap::new();
    for word in word_freqs.keys() {
        splits.insert(word, to_symbols(&byte_symbols, word));
    }
    // println!("Word[char] Dictionary: {splits:?}\n");

    let mut merges: Vec<(String, String)> = Vec::new();

    while vocab.len() < vocab_size {
        let pair_freqs = compute_pair_freqs(&splits, &word_freqs);
        // Highest count wins, ties go to the smallest pair (Reverse flips the
        // pair order), so the pick does not depend on HashMap iteration order.
        let Some((best_pair, _)) = pair_freqs
            .iter()
            .max_by_key(|(pair, freq)| (**freq, Reverse(*pair)))
        else {
            break;
        };
        let best_pair = best_pair.clone();
        let a = &best_pair.0;
        let b = &best_pair.1;

        // mutates splits slice
        merge_pairs(a, b, &mut splits);
        let pair_merged = format!("{a}{b}");
        merges.push(best_pair);
        vocab.push(pair_merged);
    }

    (vocab, merges)
}
fn encode_chunk(
    chunk: &str,
    ranks: &HashMap<(String, String), usize>,
    byte_symbols: &[char; 256],
) -> Vec<String> {
    let mut pieces = to_symbols(byte_symbols, chunk);
    // One key buffer for every lookup: clone_from reuses its allocation.
    let mut key: (String, String) = (String::new(), String::new());
    loop {
        // index, rank
        let mut best: Option<(usize, usize)> = None;

        for (i, w) in pieces.windows(2).enumerate() {
            key.0.clone_from(&w[0]);
            key.1.clone_from(&w[1]);
            if let Some(&rank) = ranks.get(&key)
                && best.is_none_or(|(_, best_rank)| rank < best_rank)
            {
                best = Some((i, rank));
            }
        }

        match best {
            Some((i, _)) => {
                let first = &pieces[i];
                let second = &pieces[i + 1];
                pieces[i] = format!("{first}{second}");
                pieces.remove(i + 1);
            }
            None => break,
        }
    }
    pieces
}

fn encode(
    text: &str,
    re: &Regex,
    ranks: &HashMap<(String, String), usize>,
    byte_symbols: &[char; 256],
    vocab_ids: &HashMap<String, usize>,
) -> Vec<usize> {
    let mut ids = Vec::new();
    for chunk in pre_tokenize(re, text) {
        for piece in encode_chunk(chunk, ranks, byte_symbols) {
            ids.push(vocab_ids[&piece]);
        }
    }
    ids
}

fn to_symbols(byte_symbols: &[char; 256], word: &str) -> Vec<String> {
    word.bytes()
        .map(|b| byte_symbols[usize::from(b)].to_string())
        .collect()
}

fn build_ranks(merges: &[(String, String)]) -> HashMap<(String, String), usize> {
    merges
        .iter()
        .enumerate()
        .map(|(rank, pair)| (pair.clone(), rank))
        .collect()
}

fn build_vocab_ids(vocab: &[String]) -> HashMap<String, usize> {
    vocab
        .iter()
        .enumerate()
        .map(|(id, token)| (token.clone(), id))
        .collect()
}

fn merge_pairs(a: &str, b: &str, splits: &mut HashMap<&str, Vec<String>>) {
    for split in splits.values_mut() {
        if split.len() < 2 {
            continue;
        }

        let mut i = 0;
        while i + 1 < split.len() {
            if split[i] == a && split[i + 1] == b {
                split[i] = format!("{a}{b}");
                split.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }
}

fn compute_pair_freqs(
    splits: &HashMap<&str, Vec<String>>,
    word_freqs: &HashMap<&str, u32>,
) -> HashMap<(String, String), u32> {
    let mut pair_freqs: HashMap<(String, String), u32> = HashMap::new();

    for (word, freq) in word_freqs {
        let split = &splits[word];
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

// GPT-2's byte -> printable char table, indexed by byte value. Bytes that
// already print cleanly keep their own char. The other 68 (controls, space,
// 0x7F-0xA0, 0xAD) are shifted up to U+0100.. in byte order, so space becomes
// 'Ġ' and newline 'Ċ'.
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

fn pre_tokenize<'a>(re: &Regex, text: &'a str) -> Vec<&'a str> {
    re.find_iter(text).map(|m| m.unwrap().as_str()).collect()
}

#[cfg(test)]
mod tests;
