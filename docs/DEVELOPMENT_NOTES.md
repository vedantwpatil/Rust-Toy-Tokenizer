## Lifetimes

For this function

```rust
fn pre_tokenize<'a>(re: &Regex, text: &'a str) -> Vec<&'a str> {}
```

The compiler doesn't need to know how long these references live. The caller's data decides that. It needs to know which input the returned references borrow from. With two reference parameters and no `self`, lifetime elision can't infer that, so the signature has to say it. A lifetime is a region of the program, from where the borrow is created to its last use. It is not a lexical scope.

Lifetimes don't tie data together and don't affect when anything is dropped. Owners drop at the end of their scope, in reverse declaration order. A lifetime adds one constraint: the owner of the borrowed data must outlive every use of the borrow.

`'a` on `text` and on the returned elements tells the caller that every `&str` in the returned `Vec` borrows from `text`. So `text`'s owner must stay alive until the last use of the `Vec`. The reverse is not true: dropping the `Vec` does nothing to `text`. The `Vec` owns its heap buffer and only its elements borrow, which is why `'a` sits on `&'a str` and not on the `Vec`. `re` has its own separate, elided lifetime. The output doesn't borrow from it, so the caller can drop `re` early.

The signature is a contract the compiler checks from both sides. It checks the body against the contract, and it checks every caller against it without reading the body. The body can keep this promise because `fancy_regex::Match<'t>::as_str` returns `&'t str`, tied to the searched text and not to the regex.
