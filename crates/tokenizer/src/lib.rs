mod baseline;
mod byte_level;
mod encode;
mod metrics;
mod pretokenize;
mod tokenizer;
mod train;

pub use baseline::{tokenize_by_byte, tokenize_by_char, tokenize_by_word};
pub use metrics::bytes_per_token;
pub use tokenizer::{ENDOFTEXT, Tokenizer};
pub use train::{TOY_CORPUS, TOY_VOCAB_SIZE};

#[cfg(test)]
mod tests;
