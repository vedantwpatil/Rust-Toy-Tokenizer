# Research on Tokenizers

## What is a tokenizer?

The first question that we should look to answer is what is a tokenzier formally and how does it integrate with large language models?

Tokenizers are used to convert a prompt sent to a large language model into something that they're able to better interpret. The process of this is breaking up the single prompt into smaller groups of letters and characters called **tokens** in which it can use and numerically quantify as **vectors** to figure out a response to your prompt.

#### Why do we need tokenizers in our LLMS

Without tokenizers LLMs struggle to understand the prompt as it's not as evident for a machine to draw conclusions on how language and words groups together to form definitions. When thinking about things in the English language, the example that jumps out to me is __ (Something witty, I'm not clever enough to figure out right now). There are more technical reasons which definitely exist but I don't plan on getting bogged down in as they feel relatively overkill to look into

## How are tokenizers trained

They're trained by collecting a **text corpus** which is a large collection of spoken text. This in combination with a tokenization method is done to build up the vocabulary of the tokenizer and have it be able to break up common language into tokens for the llm to understand. A tokenizer with a poor vocabulary will struggle to break up the prompt in a way that a llm can meaningfully understand and will increase the amount of compute it takes to understand the same phrase.

## What are the different types of tokenizers?

### The simplest tokenizer

The most basic type of tokenizer would be just each individual character is a token

The pros of this is that we're able to simply understand what a token is, the issue is that it isn't very helpful in determining understanding of our text

### BPE (Byte Pair Encoding)

**Definition**:
Byte Pair Encoding training starts by computing the unique set of words used in the corpus (after the normalization and pre-tokenization steps are completed), then building the vocabulary by taking all the symbols used to write those words.

#### What does that actually mean?

First phrase that looks somewhat unfamiliar is corpus.

**Corpus Definition**: A corpus is the collection of text data used to train the tokenizer which helps it identify the most frequent pairs of tokens

What are the normalization and pre-tokenization steps?

**BPE Architecture**

We have three major steps  
normalization, pre-tokenization, splitting words into individual characters, merging the appropriate characters together depending on the rules learned

#### Technical Notes

##### i32 vs u32

Something to be aware of in our implementation is u32 vs i32 in this function particularly

```rust
fn compute_pair_freqs(
    splits: &HashMap<&str, Vec<String>>,
    word_freqs: &HashMap<&str, u32>,
) -> HashMap<(String, String), u32> {
    let mut pair_freqs: HashMap<(String, String), u32> = HashMap::new();

    for (word, freq) in word_freqs {
        let split = splits[word].clone();
        for window in split.windows(2) {
            let pair = (window[0].clone(), window[1].clone());
            *pair_freqs.entry(pair).or_insert(0) += freq;
        }
    }
    pair_freqs
}
```

Pair counts are `u32`, not `i32`. A count is never negative and `u32` doubles the range, but that only raises the limit: a pair count is bounded by the corpus size in bytes, so `u32` is safe up to about 4.29 GB. Overflow panics in a debug build and wraps silently in a release build. The code change and the overflow behaviour are in [I32_VS_U32.md](I32_VS_U32.md).

##### Chars vs Bytes

We now use a byte lookup table instead of letters found in our corpus as our vocab

```rust
let byte_symbols = bytes_to_unicode();
let mut vocab: Vec<String> = vec!["<|endoftext|>".to_string()];
vocab.extend(byte_symbols.iter().map(std::string::ToString::to_string));

let mut splits: HashMap<&str, Vec<String>> = HashMap::new();
for word in word_freqs.keys() {
    let vec_of_chars: Vec<String> = word
        .bytes()
        .map(|b| byte_symbols[usize::from(b)].to_string())
        .collect();
    splits.insert(word, vec_of_chars);
}
```

The Hugging Face course starts BPE from the characters found in the corpus. This project starts from the 256 byte values, each mapped to a symbol through `bytes_to_unicode`. That removes the need for an unknown token, makes the base vocabulary fixed, and makes training deterministic. The before and after code, examples, and the reasons for the mapping table are in [CHARS_VS_BYTES.md](CHARS_VS_BYTES.md).

## How do large language models use different tokenizers for different purposes?

## How do you implement a tokenizer?

It can't just be a simple algorithm like breaking up a prompt word by word can it?

## Are more or less tokens better?

Something I'm referencing as I'm learning more about this
[blog post](https://rishijeet.github.io/blog/from-text-to-tokens-the-complete-guide-to-tokenization-in-llms/)

[hugging face post](https://deepwiki.com/huggingface/tokenizers/5.1-bpe-training)

[secondary hugging face post](https://huggingface.co/learn/llm-course/chapter6/5)

Great [video ](https://youtu.be/_xM8scs4_x4) explaining the current state of LLM inference providers. Helps explain the question of what is the llm inference service that is provided
