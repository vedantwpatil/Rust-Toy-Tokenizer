use std::{env, fs};
use tokenizer::{
    TOY_CORPUS, TOY_VOCAB_SIZE, Tokenizer, bytes_per_token, tokenize_by_byte, tokenize_by_char,
    tokenize_by_word,
};

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
    let tokenizer = Tokenizer::train(&corpus, TOY_VOCAB_SIZE);

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
    println!(
        "Tokenize with BPE\nVocab:{:?}\n\nMerges{:?}\n\n",
        tokenizer.vocab, tokenizer.merges
    );

    let ids = tokenizer.encode(&prompt);
    let tokens: Vec<&str> = ids.iter().map(|&id| tokenizer.vocab[id].as_str()).collect();
    println!("Encoded ids: {ids:?}\n");
    println!("Encoded tokens: {tokens:?}\n");

    let decoded_prompt = tokenizer.decode(&ids).unwrap();
    println!("Prompt after being encoded and decoded: {decoded_prompt:?}\n");

    // METRICS
    // The trained vocab is tiny, so GPT-2's published merges are the useful
    // comparison. They load as data; without the file that row is skipped.
    let trained_on = if corpus.len() == 1 {
        "the prompt".to_string()
    } else {
        format!("{} corpus lines", corpus.len())
    };
    let mut counts = vec![
        ("word".to_string(), word_by_word.len()),
        ("char".to_string(), char_by_char.len()),
        ("byte".to_string(), byte_by_byte.len()),
        (
            format!(
                "BPE, {} vocab, trained on {trained_on}",
                tokenizer.vocab.len()
            ),
            ids.len(),
        ),
    ];
    match fs::read_to_string(GPT2_MERGES_PATH) {
        Ok(merges_txt) => {
            let gpt2 = Tokenizer::from_gpt2_merges(&merges_txt);
            counts.push((
                format!("BPE, {} vocab, GPT-2's training", gpt2.vocab.len()),
                gpt2.encode(&prompt).len(),
            ));
        }
        Err(e) => println!("(no GPT-2 row: {GPT2_MERGES_PATH}: {e})"),
    }
    println!("Compression of the same {} byte prompt", prompt.len());
    for (name, n) in counts {
        let ratio = bytes_per_token(&prompt, n);
        println!("{n:>5} tokens, {ratio:>5.2} bytes/token  {name}");
    }

    Ok(())
}

fn take_input() -> Result<String, String> {
    env::args()
        .nth(1)
        .ok_or_else(|| "No prompt provided".to_string())
}

const GPT2_MERGES_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tokenizer/data/gpt2/merges.txt"
);
