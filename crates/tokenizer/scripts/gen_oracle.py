"""Regenerates data/gpt2/oracle_cases.in, the expected GPT-2 ids the Rust
oracle test compares against, and data/gpt2/bench_expected.in, the token count
and id hash the speed benchmarks check their output against. Needs
`pip install tiktoken`; run from crates/tokenizer with
`python3 -I scripts/gen_oracle.py`.

The output is a Rust expression (`include!`d by src/tests.rs), so there is no
JSON parser on the Rust side. Non-printable-ASCII chars are written as
\\u{..} escapes so the file survives any editor.
"""

import tiktoken

ENDOFTEXT = "<|endoftext|>"

CASES = [
    "",
    "x",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "This is the hugging face course",
    # contractions, curly apostrophe, stray apostrophes
    "I've been told they'll say we're done, he's gone; she'd go. I'm sure.",
    "rock'n'roll 's 't 're O'Brien’s DON'T",
    # numbers
    "It's 3:45pm on 2024-10-24, call 555-0123. 1234567890 3.14159 1e-10",
    # code
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
    'fn main() {\n\tprintln!("{}", 1 + 2);\n}\n',
    "https://example.com/a/b?c=d&e=f#g",
    "camelCaseIdentifier snake_case_name HELLO WORLD",
    # whitespace: leading, trailing, runs, tabs, CRLF
    " ",
    "   ",
    "  leading",
    "trailing  ",
    "a  b   c",
    "\n\n\nthree newlines",
    "tabs\t\there",
    "\r\n windows\r\n",
    "end \n",
    "word \n word",
    # repeats
    "a" * 50,
    "!!!!!!!!!! ????",
    # non-ASCII
    "café naïve résumé",
    "日本語のテキスト",
    "Привет, мир!",
    "नमस्ते",
    "emoji \U0001F642\U0001F680\U0001F44D\U0001F3FD",
    "\U0001D514\U0001D52B\U0001D526\U0001D520\U0001D52C\U0001D521\U0001D522",
    # odd whitespace and control characters
    "a b",
    "x​y",
    "­",
    "\x00\x01 \x7f",
    # long-ish prose
    "Four score and seven years ago our fathers brought forth on this continent, "
    "a new nation, conceived in Liberty, and dedicated to the proposition that "
    "all men are created equal.\n\nNow we are engaged in a great civil war, "
    "testing whether that nation, or any nation so conceived and so dedicated, "
    "can long endure.",
    # special token: recognised and never merged
    ENDOFTEXT,
    "before" + ENDOFTEXT + "after",
    ENDOFTEXT + ENDOFTEXT,
    "line one\n" + ENDOFTEXT + "\nline two",
]


def rust_str(s):
    out = []
    for ch in s:
        if 0x20 <= ord(ch) < 0x7F and ch not in '"\\':
            out.append(ch)
        else:
            out.append("\\u{%x}" % ord(ch))
    return '"' + "".join(out) + '"'


enc = tiktoken.get_encoding("gpt2")
lines = ["&["]
for text in CASES:
    ids = enc.encode(text, allowed_special={ENDOFTEXT})
    assert enc.decode(ids) == text
    lines.append("    (%s, &[%s])," % (rust_str(text), ", ".join(map(str, ids))))
lines.append("]")

with open("data/gpt2/oracle_cases.in", "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote %d cases" % len(CASES))


# Speed benchmarks in src/tests.rs encode these two texts and compare the
# number of ids and an FNV-1a hash of them with tiktoken's. BENCH_REPEATS and
# both text rules must match the Rust side.
BENCH_REPEATS = 50
BENCH_TEXTS = ["\n".join(CASES) * BENCH_REPEATS, "a" * 4000]


def fnv(ids):
    h = 0xCBF29CE484222325
    for i in ids:
        h = ((h ^ i) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


expected = []
for text in BENCH_TEXTS:
    ids = enc.encode(text, allowed_special={ENDOFTEXT})
    expected.append("(%d, 0x%x)" % (len(ids), fnv(ids)))
with open("data/gpt2/bench_expected.in", "w") as f:
    f.write("[%s]\n" % ", ".join(expected))
print("wrote bench expectations")
