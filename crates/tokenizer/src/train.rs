use crate::byte_level::{bytes_to_unicode, to_symbols};
use crate::pretokenize::pre_tokenize;
use fancy_regex::Regex;
use std::{cmp::Reverse, collections::HashMap};

// 257 base symbols (256 bytes plus <|endoftext|>) leave room for 23 merges.
pub const TOY_VOCAB_SIZE: usize = 280;

pub const TOY_CORPUS: [&str; 4] = [
    "This is the hugging face course",
    "This chapter is about tokenization",
    "This section shows several tokenizer algorithms",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
];

pub(crate) fn tokenize_bpe(
    corpus: &[&str],
    vocab_size: usize,
    re: &Regex,
) -> (Vec<String>, Vec<(String, String)>) {
    // println!("Corpus: {corpus:?}\n");

    // Pre tokenize
    let mut word_freqs: HashMap<&str, u32> = HashMap::new();

    // pre_tokenize step?
    for &text in corpus {
        let tokenize = pre_tokenize(re, text);
        // Next step is the compute the frequency of each word in the tokenized corpus
        for word in tokenize {
            *word_freqs.entry(word).or_insert(0) += 1;
        }
    }

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
