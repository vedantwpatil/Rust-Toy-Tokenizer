# Rust Tokenizer

For my analysis on tokenizers and how I went about my approach [analysis](./docs/RESEARCH.md)

Plan to implement multiple types of tokenizers and experiment with "outdated" tokenizers before building a [BPE](./docs/RESEARCH.md#bpe-byte-pair-encoding) tokenizer from scratch

The main goal of this is to be able to learn about how the inference pipeline works in a lot of common machine learning pipelines. The way that we'll learn this is by first creating a tokenizer, then creating the respective components around the tokenizer to feed into a model to do inference and then finally use this inference pipeline to create my own harness. Everything should be created from scratch and in rust with writing some bindings to python to be able to call some of the crates for performance testing against other inference pipelines.

### Personal Notes

This might be apart of a greater ai chain but currently just focusing on creating a tokenizer in rust that I could use to feed to a llm. Seems like a fun problem to work on where you take in a input string "prompt" and split it up into words and then could use this to get a better understanding of how inference pipelines work and develop a better understanding of what a harness is.

I'm interested in finding out what makes a good tokenizer and what the algorithm the best tokenizers implement are
