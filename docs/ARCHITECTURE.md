# Architecture

Tokenizer: Is the encode phase in LLMs
Consists of 3 main stages
normalizer -> pretokenizer -> model

On a fixed corpus we don't need a normalizer as we can directly modify the corups to fit the constraints we have but it is valuable for more general datasets
