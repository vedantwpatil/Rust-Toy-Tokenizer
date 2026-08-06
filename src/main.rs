use std::{env, vec};

fn main() -> Result<(), String> {
    println!("Hello, rust tokenizer!");

    // We need to take in input "prompt"
    let prompt = take_input()?;

    // Tokenize the prompt

    // Could have all of the different types of tokenizers here to see how they change for the same
    // prompt

    let char_by_char = tokenize_by_char(prompt.clone());
    let word_by_word = tokenize_by_word(prompt.clone());

    // DISPLAY
    println!("{prompt:?}");

    // Should consider running this into a llm or having some visual/demonstration of how each
    // tokenizer works and the performance of it. That might be something to do after the tokenizers
    // themselves as it would first require creating my own LLM to be able to feed a tokenized prompt
    // into which would be it's own project
    //
    // Some ideas on displays with just the tokenizer itself were to figure out the compresion ratio
    // it was able to determine and printing out the merged tokens to see the patterns it learned
    println!("{word_by_word:?}");
    println!("{char_by_char:?}");
    Ok(())
}

fn take_input() -> Result<String, String> {
    let args: Vec<String> = env::args().collect();
    match args.get(1) {
        Some(prompt) => Ok(prompt.to_string()),
        None => Err("No prompt provided".to_string()),
    }
}

fn tokenize_by_char(prompt: String) -> Vec<char> {
    let mut tokens: Vec<char> = vec![];

    for c in prompt.chars() {
        tokens.push(c);
    }
    tokens
}

fn tokenize_by_word(prompt: String) -> Vec<String> {
    let mut tokens: Vec<String> = vec![];

    // Should we go character by character and implement matching or is that overkill?
    // The logic would probably be duplicated for others
    //
    // I think it's more important to not and do something simpler that way we can effectively
    // compare it later down the line
    //
    // Considering a few approaches right now, either for loop where we look word by word (need to
    // figure out how to do that in rust)
    //
    // While loop where we go letter by letter and match for the types of characters we'd be
    // expecting, the matching seems elegant but overkill

    for word in prompt.split_whitespace() {
        tokens.push(word.to_string());
    }

    tokens
}
