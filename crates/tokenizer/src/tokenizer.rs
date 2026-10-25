use crate::byte_level::{build_char_to_byte, bytes_to_unicode, prints_cleanly};
use crate::encode::{build_ranks, build_vocab_ids, encode_chunk};
use crate::pretokenize::{GPT2_PATTERN, pre_tokenize};
use crate::train::tokenize_bpe;
use fancy_regex::Regex;
use std::{cmp::Reverse, collections::HashMap, string::FromUtf8Error};

pub const ENDOFTEXT: &str = "<|endoftext|>";

// Everything encode and decode need, built once. An id is an index into
// `vocab`, so the layout is whatever the constructor chose: training puts
// <|endoftext|> first, the GPT-2 loader matches GPT-2's ids.
pub struct Tokenizer {
    pub vocab: Vec<String>,
    pub merges: Vec<(String, String)>,
    pub(crate) ranks: HashMap<(String, String), usize>,
    pub(crate) vocab_ids: HashMap<String, usize>,
    pub(crate) byte_symbols: [char; 256],
    pub(crate) char_to_byte: HashMap<char, u8>,
    pub(crate) pattern: Regex,
    // shortcut: linear scan per lookup, fine for a handful of special tokens
    pub(crate) specials: Vec<(String, usize)>,
}

impl Tokenizer {
    pub fn new(
        vocab: Vec<String>,
        merges: Vec<(String, String)>,
        pattern: &str,
        specials: Vec<(String, usize)>,
    ) -> Self {
        // An empty special would match at every position and never advance.
        assert!(specials.iter().all(|(text, _)| !text.is_empty()));
        let byte_symbols = bytes_to_unicode();
        Self {
            ranks: build_ranks(&merges),
            vocab_ids: build_vocab_ids(&vocab),
            char_to_byte: build_char_to_byte(&byte_symbols),
            pattern: Regex::new(pattern).unwrap(),
            byte_symbols,
            vocab,
            merges,
            specials,
        }
    }

    pub fn train(corpus: &[&str], vocab_size: usize) -> Tokenizer {
        let re = Regex::new(GPT2_PATTERN).unwrap();
        let (vocab, merges) = tokenize_bpe(corpus, vocab_size, &re);
        Self::new(
            vocab,
            merges,
            GPT2_PATTERN,
            vec![(ENDOFTEXT.to_string(), 0)],
        )
    }

    pub fn from_gpt2_merges(merges_txt: &str) -> Self {
        let byte_symbols = bytes_to_unicode();
        let mut vocab: Vec<String> = (0..=u8::MAX)
            .filter(|&b| prints_cleanly(b))
            .chain((0..=u8::MAX).filter(|&b| !prints_cleanly(b)))
            .map(|b| byte_symbols[usize::from(b)].to_string())
            .collect();
        let merges: Vec<(String, String)> = merges_txt
            .lines()
            .skip(1)
            .filter(|line| !line.is_empty())
            .map(|line| {
                let (a, b) = line.split_once(' ').unwrap();
                (a.to_string(), b.to_string())
            })
            .collect();
        vocab.extend(merges.iter().map(|(a, b)| format!("{a}{b}")));
        let endoftext_id = vocab.len();
        vocab.push(ENDOFTEXT.to_string());
        Tokenizer::new(
            vocab,
            merges,
            GPT2_PATTERN,
            vec![(ENDOFTEXT.to_string(), endoftext_id)],
        )
    }

    // Special tokens are cut out of the text first and emitted as their own id,
    // so they are never pre-tokenized or merged. Text between them is encoded
    // normally.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        let mut rest = text;
        // Earliest special in `rest`, the longest one if several start together.
        while let Some((at, len, id)) = self
            .specials
            .iter()
            .filter_map(|(special, id)| {
                rest.find(special.as_str())
                    .map(|at| (at, special.len(), *id))
            })
            .min_by_key(|&(at, len, _)| (at, Reverse(len)))
        {
            self.encode_plain(&rest[..at], &mut ids);
            ids.push(id);
            rest = &rest[at + len..];
        }
        self.encode_plain(rest, &mut ids);
        ids
    }

    fn encode_plain(&self, text: &str, ids: &mut Vec<usize>) {
        for chunk in pre_tokenize(&self.pattern, text) {
            for piece in encode_chunk(chunk, &self.ranks, &self.byte_symbols) {
                ids.push(self.vocab_ids[&piece]);
            }
        }
    }

    // Bytes are collected across all tokens and turned into a String once at
    // the end: one token can hold half of a multi-byte character.
    pub fn decode(&self, ids: &[usize]) -> Result<String, FromUtf8Error> {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            // A special's text is plain UTF-8, not byte symbols.
            if let Some((special, _)) = self
                .specials
                .iter()
                .find(|(_, special_id)| *special_id == id)
            {
                bytes.extend_from_slice(special.as_bytes());
                continue;
            }
            for chars in self.vocab[id].chars() {
                bytes.push(self.char_to_byte[&chars]);
            }
        }
        String::from_utf8(bytes)
    }
}
