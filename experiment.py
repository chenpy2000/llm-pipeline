"""
experiment.py — Validate that the BPE tokenizer works correctly.

Tests:
  1. Roundtrip: encode(text) -> decode -> should recover original text
  2. Vocab size matches requested size
  3. Special tokens get their own IDs and survive roundtrip
  4. Every byte (0-255) is in the vocab (full byte coverage)
  5. Merge count equals vocab_size - 256 - len(special_tokens)
  6. Encoding unseen text still works (no KeyError)
  7. Save/load roundtrip preserves encode/decode behavior
  8. Empty string encodes to empty list
  9. Unicode / multibyte text roundtrips correctly
"""

import os
import tempfile
from tokenizer import Tokenizer


def run_tests():
    # ---- Training corpus ----
    train_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
        "Peter Piper picked a peck of pickled peppers.",
        "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.",
        "The rain in Spain stays mainly in the plain.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "A stitch in time saves nine.",
        "Actions speak louder than words.",
    ] * 10  # repeat so there's enough frequency for merges

    special_tokens = ["<|endoftext|>", "<|padding|>"]
    vocab_size = 300  # small for fast testing

    print("Training tokenizer...")
    tok = Tokenizer.train(train_texts, vocab_size, special_tokens)
    print(f"  Done. Vocab size = {tok.vocab_size}, Merges = {len(tok.merges)}\n")

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}  {detail}")
            failed += 1

    # --- Test 1: Vocab size ---
    print("[Test 1] Vocab size")
    check("vocab_size matches", tok.vocab_size == vocab_size,
          f"expected {vocab_size}, got {tok.vocab_size}")

    # --- Test 2: Merge count ---
    print("[Test 2] Merge count")
    expected_merges = vocab_size - 256 - len(special_tokens)
    check("merge count", len(tok.merges) == expected_merges,
          f"expected {expected_merges}, got {len(tok.merges)}")

    # --- Test 3: Full byte coverage ---
    print("[Test 3] Byte coverage")
    all_bytes_present = all(bytes([i]) in tok.bytes_to_id for i in range(256))
    check("all 256 bytes in vocab", all_bytes_present)

    # --- Test 4: Special tokens have IDs ---
    print("[Test 4] Special tokens")
    for st in special_tokens:
        has_id = st.encode("utf-8") in tok.bytes_to_id
        check(f"'{st}' in vocab", has_id)

    # --- Test 5: Roundtrip on training data ---
    print("[Test 5] Roundtrip (training text)")
    for text in train_texts[:5]:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        check(f"roundtrip '{text[:40]}...'", decoded == text,
              f"got '{decoded[:60]}'")

    # --- Test 6: Roundtrip with special tokens ---
    print("[Test 6] Roundtrip with special tokens")
    text_with_special = "Hello world<|endoftext|>Second document<|padding|>Third"
    ids = tok.encode(text_with_special)
    decoded = tok.decode(ids)
    check("special token roundtrip", decoded == text_with_special,
          f"got '{decoded}'")

    # Check that special tokens get exactly one ID each
    eot_id = tok.bytes_to_id[b"<|endoftext|>"]
    pad_id = tok.bytes_to_id[b"<|padding|>"]
    check("<|endoftext|> is single token", ids.count(eot_id) == 1)
    check("<|padding|> is single token", ids.count(pad_id) == 1)

    # --- Test 7: Unseen text (no crash) ---
    print("[Test 7] Unseen text")
    unseen = "Supercalifragilisticexpialidocious 12345 !@#$%"
    ids = tok.encode(unseen)
    decoded = tok.decode(ids)
    check("unseen text roundtrip", decoded == unseen,
          f"got '{decoded}'")

    # --- Test 8: Unicode / multibyte ---
    print("[Test 8] Unicode text")
    unicode_texts = [
        "café résumé naïve",
        "日本語テスト",
        "Привет мир",
        "emoji: 🚀🔥✨",
        "mixed: hello世界café🎉",
    ]
    for ut in unicode_texts:
        ids = tok.encode(ut)
        decoded = tok.decode(ids)
        check(f"unicode '{ut[:30]}'", decoded == ut,
              f"got '{decoded}'")

    # --- Test 9: Empty string ---
    print("[Test 9] Empty string")
    check("empty encode", tok.encode("") == [])
    check("empty decode", tok.decode([]) == "")

    # --- Test 10: Save / Load roundtrip ---
    print("[Test 10] Save/Load")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        tok.save(tmp_path)
        tok2 = Tokenizer.load(tmp_path)

        check("loaded vocab_size", tok2.vocab_size == tok.vocab_size)
        check("loaded merges count", len(tok2.merges) == len(tok.merges))

        # Encode/decode with loaded tokenizer should match
        test_str = "The quick brown fox<|endoftext|>jumps over"
        ids1 = tok.encode(test_str)
        ids2 = tok2.encode(test_str)
        check("save/load encode match", ids1 == ids2,
              f"original={ids1[:10]}... loaded={ids2[:10]}...")

        decoded2 = tok2.decode(ids2)
        check("save/load decode match", decoded2 == test_str)
    finally:
        os.remove(tmp_path)

    # --- Test 11: Encoding is deterministic ---
    print("[Test 11] Determinism")
    det_text = "deterministic encoding test"
    ids_a = tok.encode(det_text)
    ids_b = tok.encode(det_text)
    check("same input -> same output", ids_a == ids_b)

    # --- Test 12: IDs are valid vocab indices ---
    print("[Test 12] Valid IDs")
    long_text = " ".join(train_texts[:3]) + "<|endoftext|>" + unseen
    ids = tok.encode(long_text)
    all_valid = all(i in tok.vocab for i in ids)
    check("all encoded IDs exist in vocab", all_valid)

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed == 0:
        print("All tests passed!")
    else:
        print("Some tests failed.")
    print(f"{'='*50}")

    return failed == 0


if __name__ == "__main__":
    run_tests()
