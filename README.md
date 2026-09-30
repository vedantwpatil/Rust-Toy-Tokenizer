# Rust Tokenizer

I think I have a big misunderstanding in what a tokenizer is and need to address that before I go on to attempt implementing it. 3b1b save me

This might be apart of a greater ai chain but currently just focusing on creating a tokenizer in rust that I could use to feed to a llm. Seems like a fun problem to work on where you take in a input string "prompt" and split it up into words

I'm interested in finding out what makes a good tokenizer and what the algorithm the best tokenizers implement are

For my analysis on tokenizers and how I went about my approach [RESEARCH.md]

The tokenizer that I decided to implement and focus the most around is [BPE](/RESEARCH.md#bpe-byte-pair-encoding) there is novelty in the others but I wanted to have something fun to work on from a performance aspect and thought it lined up well with the overall tech/ai atmosphere. So a good combination of my interests and the way tech is going

The final goal is to build the entire inference pipeline myself from scratch in rust and have it be performant. After building the entire inference pipeline we can then search into optimizing specific stages for specific tasks
