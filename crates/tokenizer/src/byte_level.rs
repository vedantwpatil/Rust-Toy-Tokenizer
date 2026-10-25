use std::collections::HashMap;

pub(crate) fn to_symbols(byte_symbols: &[char; 256], word: &str) -> Vec<String> {
    word.bytes()
        .map(|b| byte_symbols[usize::from(b)].to_string())
        .collect()
}

pub(crate) fn prints_cleanly(b: u8) -> bool {
    matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF)
}

// GPT-2's byte -> printable char table, indexed by byte value. Bytes that
// already print cleanly keep their own char. The other 68 (controls, space,
// 0x7F-0xA0, 0xAD) are shifted up to U+0100.. in byte order, so space becomes
// 'Ġ' and newline 'Ċ'.
pub(crate) fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut shifted = 0;
    for b in 0..=u8::MAX {
        table[usize::from(b)] = if prints_cleanly(b) {
            char::from(b)
        } else {
            let c = char::from_u32(256 + shifted).expect("below the surrogate range");
            shifted += 1;
            c
        };
    }
    table
}

pub(crate) fn build_char_to_byte(bytes_symbols: &[char; 256]) -> HashMap<char, u8> {
    let mut inverse_table: HashMap<char, u8> = HashMap::new();
    for (index, chars) in bytes_symbols.iter().enumerate() {
        inverse_table.insert(*chars, index as u8);
    }
    inverse_table
}
