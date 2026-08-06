# Research on Tokenizers

## What is a tokenizer?

The first question that we should look to answer is what is a tokenzier formally and how does it integrate with large language models?

Tokenizers are used to convert a prompt sent to a large language model into something that they're able to better interpret. The process of this is breaking up the single prompt into smaller groups of letters and characters called **tokens** in which it can use and numerically quantify as **vectors** to figure out a response to your prompt.

#### Why do we need tokenizers in our LLMS

Without tokenizers LLMs struggle to understand the prompt as it's not as evident for a machine to draw conclusions on how language and words groups together to form definitions. When thinking about things in the English language, the example that jumps out to me is __ (Something witty, I'm not clever enough to figure out right now). There are more technical reasons which definitely exist but I don't plan on getting bogged down in as they feel relatively overkill to look into

## How are tokenizers trained

They're trained by collecting a **text corpus** which is a large collection of spoken text. This in combination with a tokenization method is done to build up the vocabulary of the tokenizer and have it be able to break up common language into tokens for the llm to understand. A tokenizer with a poor vocabulary will struggle to break up the prompt in a way that a llm can meaningfully understand and will increase the amount of compute it takes to understand the same phrase.

## What are the different types of tokenizers?

#### The simplest tokenizer

The most basic type of tokenizer would be just each individual character is a token

The pros of this is that we're able to simply understand what a token is, the issue is that it isn't very helpful in determining understanding of our text

#### BPE (Byte Pair Encoding)

## How do large language models use different tokenizers for different purposes?

## How do you implement a tokenizer?

It can't just be a simple algorithm like breaking up a prompt word by word can it?

Something I'm referencing as I'm learning more about this
[blog post](https://rishijeet.github.io/blog/from-text-to-tokens-the-complete-guide-to-tokenization-in-llms/)
