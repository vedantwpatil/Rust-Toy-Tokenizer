use fancy_regex::Regex;

// GPT-2 regex match pattern
pub(crate) const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub(crate) fn pre_tokenize<'a>(re: &Regex, text: &'a str) -> Vec<&'a str> {
    re.find_iter(text).map(|m| m.unwrap().as_str()).collect()
}
