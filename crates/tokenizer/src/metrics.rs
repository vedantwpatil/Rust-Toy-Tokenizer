// Input bytes per token: higher means the tokenizer compresses more.
pub fn bytes_per_token(text: &str, n_tokens: usize) -> f64 {
    if n_tokens == 0 {
        return 0.0;
    }
    text.len() as f64 / n_tokens as f64
}
